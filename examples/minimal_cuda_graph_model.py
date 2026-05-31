import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import gate_fusion as gf


class TinyVQC(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int, device, dtype):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.ry_angles = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.rz_angles = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.cry_angles = nn.Parameter(torch.randn(n_layers, n_qubits - 1) * 0.1)

        cry_edges = [(q, q + 1) for q in range(n_qubits - 1)]
        self.cry_plans = gf.compute_two_qubit_plans(
            cry_edges, n_qubits, device, dtype, commute_fn=gf.cry_commute,
        )
        toff_triples = [(0, 1, 2)]
        self.toff_plans = gf.compute_three_qubit_plans(
            toff_triples, n_qubits, device, dtype, commute_fn=gf.toffoli_commute,
        )
        toff_re, toff_im = gf.build_toffoli_8x8(1, device, dtype)
        self.register_buffer('toff_re', toff_re)
        self.register_buffer('toff_im', toff_im)

    def forward(self, amp: torch.Tensor) -> torch.Tensor:
        B = amp.shape[0]
        nq = self.n_qubits
        re = amp.reshape(B, *([2] * nq))
        im = torch.zeros_like(re)

        for ll in range(self.n_layers):
            G_re, G_im = gf.build_ry_rz_gates(self.ry_angles[ll], self.rz_angles[ll])
            re, im = gf.apply_su2_layer(re, im, G_re, G_im)
            for plan in self.cry_plans:
                cry_a = self.cry_angles[ll].index_select(0, plan['k_idx'])
                re, im = gf.apply_disjoint_cry_layer(re, im, plan, cry_a)
            for plan in self.toff_plans:
                re, im = gf.apply_disjoint_three_qubit_layer(
                    re, im, plan, self.toff_re, self.toff_im,
                )
        return (re * re + im * im).reshape(B, -1)


def _make_inputs(n_qubits, B, device, dtype):
    amp = torch.rand(B, 1 << n_qubits, device=device, dtype=dtype)
    amp = amp / amp.sum(dim=1, keepdim=True)
    target = torch.rand_like(amp); target = target / target.sum(dim=1, keepdim=True)
    return amp, target


def _time(fn, iters, sync_cuda):
    if sync_cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync_cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def benchmark_cuda(n_qubits, n_layers, B, iters):
    device = torch.device('cuda'); dtype = torch.float32
    print(f"\n=== CUDA ({torch.cuda.get_device_name(0)}): nq={n_qubits}, layers={n_layers}, B={B} ===")
    torch.manual_seed(0)
    model = TinyVQC(n_qubits, n_layers, device, dtype).to(device)
    amp, target = _make_inputs(n_qubits, B, device, dtype)

    def fwd_eager():
        with torch.no_grad():
            return model(amp)
    for _ in range(3): fwd_eager()
    t_eager_fwd = _time(fwd_eager, iters, sync_cuda=True)

    model_g = TinyVQC(n_qubits, n_layers, device, dtype).to(device)
    model_g.load_state_dict(model.state_dict())
    graphed_fwd = torch.cuda.make_graphed_callables(model_g, (amp.clone(),), num_warmup_iters=3)
    t_graph_fwd = _time(lambda: graphed_fwd(amp), iters, sync_cuda=True)

    print(f"  forward (no_grad):")
    print(f"    eager:       {t_eager_fwd*1000:7.2f} ms/call")
    print(f"    cuda-graph:  {t_graph_fwd*1000:7.2f} ms/call    speedup: {t_eager_fwd/t_graph_fwd:5.2f}x")

    model_t = TinyVQC(n_qubits, n_layers, device, dtype).to(device)
    opt = torch.optim.Adam(model_t.parameters(), lr=1e-3)
    def step():
        opt.zero_grad(set_to_none=True)
        loss = (model_t(amp) - target).pow(2).mean()
        loss.backward(); opt.step()
    for _ in range(3): step()
    t_eager_step = _time(step, iters, sync_cuda=True)

    model_tg = TinyVQC(n_qubits, n_layers, device, dtype).to(device)
    opt_g = torch.optim.Adam(model_tg.parameters(), lr=1e-3)
    graphed_t = torch.cuda.make_graphed_callables(model_tg, (amp.clone(),), num_warmup_iters=3)
    def step_g():
        opt_g.zero_grad(set_to_none=True)
        loss = (graphed_t(amp) - target).pow(2).mean()
        loss.backward(); opt_g.step()
    for _ in range(3): step_g()
    t_graph_step = _time(step_g, iters, sync_cuda=True)

    print(f"  training step (fwd + bwd + opt):")
    print(f"    eager:       {t_eager_step*1000:7.2f} ms/step")
    print(f"    cuda-graph:  {t_graph_step*1000:7.2f} ms/step    speedup: {t_eager_step/t_graph_step:5.2f}x")


def _try_torch_compile(model, sample_input):
    compiled = torch.compile(model, fullgraph=False, mode="default")
    t0 = time.perf_counter()
    try:
        if isinstance(sample_input, tuple):
            with torch.no_grad():
                compiled(*sample_input)
        else:
            with torch.no_grad():
                compiled(sample_input)
    except Exception as e:
        return None, str(e).splitlines()[-1]
    return compiled, time.perf_counter() - t0


def benchmark_cpu(n_qubits, n_layers, B, iters):
    device = torch.device('cpu'); dtype = torch.float32
    print(f"\n=== CPU: nq={n_qubits}, layers={n_layers}, B={B}  (threads={torch.get_num_threads()}) ===")
    torch.manual_seed(0)
    model = TinyVQC(n_qubits, n_layers, device, dtype).to(device)
    amp, target = _make_inputs(n_qubits, B, device, dtype)

    def fwd_eager():
        with torch.no_grad():
            return model(amp)
    for _ in range(3): fwd_eager()
    t_eager_fwd = _time(fwd_eager, iters, sync_cuda=False)

    print(f"  forward (no_grad):")
    print(f"    eager:                  {t_eager_fwd*1000:7.2f} ms/call")

    model_c = TinyVQC(n_qubits, n_layers, device, dtype).to(device)
    model_c.load_state_dict(model.state_dict())
    compiled, info = _try_torch_compile(model_c, amp)
    if compiled is None:
        print(f"    torch.compile:          skipped ({info})")
    else:
        def fwd_compiled():
            with torch.no_grad():
                return compiled(amp)
        for _ in range(2): fwd_compiled()
        t_compiled_fwd = _time(fwd_compiled, iters, sync_cuda=False)
        print(f"    torch.compile:          {t_compiled_fwd*1000:7.2f} ms/call    "
              f"speedup: {t_eager_fwd/t_compiled_fwd:5.2f}x  (compile cost: {info:.1f}s)")

    model_t = TinyVQC(n_qubits, n_layers, device, dtype).to(device)
    opt = torch.optim.Adam(model_t.parameters(), lr=1e-3)
    def step():
        opt.zero_grad(set_to_none=True)
        loss = (model_t(amp) - target).pow(2).mean()
        loss.backward(); opt.step()
    for _ in range(3): step()
    t_eager_step = _time(step, iters, sync_cuda=False)

    print(f"  training step (fwd + bwd + opt):")
    print(f"    eager:                  {t_eager_step*1000:7.2f} ms/step")

    model_tc = TinyVQC(n_qubits, n_layers, device, dtype).to(device)
    opt_c = torch.optim.Adam(model_tc.parameters(), lr=1e-3)
    compiled_t = torch.compile(model_tc, fullgraph=False, mode="default")
    def step_c():
        opt_c.zero_grad(set_to_none=True)
        loss = (compiled_t(amp) - target).pow(2).mean()
        loss.backward(); opt_c.step()
    try:
        t0 = time.perf_counter()
        step_c(); step_c()
        compile_step_cost = time.perf_counter() - t0
        for _ in range(2): step_c()
        t_compiled_step = _time(step_c, iters, sync_cuda=False)
        print(f"    torch.compile:          {t_compiled_step*1000:7.2f} ms/step    "
              f"speedup: {t_eager_step/t_compiled_step:5.2f}x  (compile cost: {compile_step_cost:.1f}s)")
    except Exception as e:
        print(f"    torch.compile:          skipped ({str(e).splitlines()[-1]})")


def benchmark_pennylane(n_qubits, n_layers, B, iters):
    try:
        import pennylane as qml
    except ImportError:
        print("\n=== PennyLane: skipped (not installed) ===")
        return
    print(f"\n=== PennyLane lightning.qubit: nq={n_qubits}, layers={n_layers}, B={B} ===")

    torch.manual_seed(0)
    ry = (torch.randn(n_layers, n_qubits) * 0.1).numpy()
    rz = (torch.randn(n_layers, n_qubits) * 0.1).numpy()
    cry = (torch.randn(n_layers, n_qubits - 1) * 0.1).numpy()
    amps = torch.rand(B, 1 << n_qubits)
    amps = (amps / amps.sum(dim=1, keepdim=True)).numpy()

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(init_state):
        qml.StatePrep(init_state, wires=range(n_qubits), normalize=False)
        for ll in range(n_layers):
            for q in range(n_qubits):
                qml.RY(ry[ll, q], wires=q)
                qml.RZ(rz[ll, q], wires=q)
            for q in range(n_qubits - 1):
                qml.CRY(cry[ll, q], wires=[q, q + 1])
            qml.Toffoli(wires=[0, 1, 2])
        return qml.probs(wires=range(n_qubits))

    for _ in range(3):
        _ = circuit(amps[0])

    t0 = time.perf_counter()
    for _ in range(iters):
        for b in range(B):
            _ = circuit(amps[b])
    t_per_batch = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters * B):
        _ = circuit(amps[0])
    t_per_call = (time.perf_counter() - t0) / (iters * B)

    print(f"  per-circuit eval (B=1):          {t_per_call*1000:7.2f} ms/call")
    print(f"  batch of {B} sequential evals:     {t_per_batch*1000:7.2f} ms  ({t_per_batch*1000/B:6.2f} ms/sample)")


if __name__ == "__main__":
    n_qubits, n_layers, B, iters = 8, 3, 32, 50
    benchmark_cpu(n_qubits, n_layers, B, iters)
    if torch.cuda.is_available():
        benchmark_cuda(n_qubits, n_layers, B, iters)
    benchmark_pennylane(n_qubits, n_layers, B, iters)

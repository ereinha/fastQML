import sys
import time

import torch
import torchquantum as tq
import torchquantum.functional as tqf

sys.path.insert(0, 'examples')
from minimal_cuda_graph_model import TinyVQC, _make_inputs

DEVICE = torch.device('cuda')
NQ, NL, B, ITERS = 8, 3, 32, 200


class TQModel(torch.nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ry_angles = torch.nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.rz_angles = torch.nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.cry_angles = torch.nn.Parameter(torch.randn(n_layers, n_qubits - 1) * 0.1)

    def forward(self, amp):
        bsz = amp.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=amp.device)
        qdev.states = amp.to(torch.complex64).reshape(bsz, *([2] * self.n_qubits))
        for ll in range(self.n_layers):
            for q in range(self.n_qubits):
                tqf.ry(qdev, wires=q, params=self.ry_angles[ll, q].expand(bsz))
                tqf.rz(qdev, wires=q, params=self.rz_angles[ll, q].expand(bsz))
            for q in range(self.n_qubits - 1):
                tqf.cry(qdev, wires=[q, q + 1], params=self.cry_angles[ll, q].expand(bsz))
            tqf.ccnot(qdev, wires=[0, 1, 2])
        s = qdev.states.reshape(bsz, -1)
        return (s.real ** 2 + s.imag ** 2)


def _time(fn, iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == '__main__':
    torch.manual_seed(0)
    fq = TinyVQC(NQ, NL, DEVICE, torch.float32).to(DEVICE)
    tqm = TQModel(NQ, NL).to(DEVICE)
    with torch.no_grad():
        tqm.ry_angles.copy_(fq.ry_angles)
        tqm.rz_angles.copy_(fq.rz_angles)
        tqm.cry_angles.copy_(fq.cry_angles)
    amp, target = _make_inputs(NQ, B, DEVICE, torch.float32)

    with torch.no_grad():
        p_fq = fq(amp)
        p_tq = tqm(amp)
    d = (p_fq - p_tq).abs().max().item()
    print(f"cross-check |probs_fastqml - probs_torchquantum| = {d:.2e}  "
          f"({'PASS' if d < 1e-5 else 'FAIL'})")

    def tq_fwd():
        with torch.no_grad():
            tqm(amp)
    for _ in range(5):
        tq_fwd()
    t_fwd = _time(tq_fwd, ITERS)

    opt = torch.optim.Adam(tqm.parameters(), lr=1e-3)

    def tq_step():
        opt.zero_grad(set_to_none=True)
        loss = ((tqm(amp) - target) ** 2).sum()
        loss.backward()
        opt.step()
    for _ in range(5):
        tq_step()
    t_step = _time(tq_step, ITERS)

    print(f"=== TorchQuantum {tq.__version__} (CUDA eager, batched): nq={NQ}, layers={NL}, B={B} ===")
    print(f"  forward (no_grad):            {t_fwd:8.2f} ms/call   ({t_fwd / B:.3f} ms/sample)")
    print(f"  training step (fwd+bwd+opt):  {t_step:8.2f} ms/step")

    try:
        tqc = TQModel(NQ, NL).to(DEVICE)
        tqc.load_state_dict(tqm.state_dict())
        comp = torch.compile(tqc, dynamic=False)
        with torch.no_grad():
            for _ in range(5):
                comp(amp)
        def comp_fwd():
            with torch.no_grad():
                comp(amp)
        t_c = _time(comp_fwd, ITERS)
        opt_c = torch.optim.Adam(comp.parameters(), lr=1e-3)
        def comp_step():
            opt_c.zero_grad(set_to_none=True)
            loss = ((comp(amp) - target) ** 2).sum()
            loss.backward()
            opt_c.step()
        for _ in range(5):
            comp_step()
        t_cs = _time(comp_step, ITERS)
        print(f"  forward (torch.compile):      {t_c:8.2f} ms/call   ({t_c / B:.3f} ms/sample)")
        print(f"  training step (torch.compile):{t_cs:8.2f} ms/step")
    except Exception as e:
        print(f"  torch.compile:                failed ({type(e).__name__})")

    try:
        tqg = TQModel(NQ, NL).to(DEVICE)
        tqg.load_state_dict(tqm.state_dict())
        graphed = torch.cuda.make_graphed_callables(tqg, (amp.clone(),), num_warmup_iters=3)
        t_g = _time(lambda: graphed(amp), ITERS)
        print(f"  forward (cuda-graph):         {t_g:8.2f} ms/call   ({t_g / B:.3f} ms/sample)")
    except Exception as e:
        print(f"  forward (cuda-graph):         capture failed ({type(e).__name__})")

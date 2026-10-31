import torch
import pennylane as qml
import numpy as np

from fastqml import gate_fusion as gf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
TOL = 1e-10


def make_random_state(nq, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    re = torch.randn(1, *([2] * nq), device=DEVICE, dtype=DTYPE, generator=g)
    im = torch.randn(1, *([2] * nq), device=DEVICE, dtype=DTYPE, generator=g)
    nm = torch.sqrt((re.reshape(1, -1)**2 + im.reshape(1, -1)**2).sum(dim=1, keepdim=True))
    re = re / nm.reshape(1, *([1] * nq))
    im = im / nm.reshape(1, *([1] * nq))
    return re, im


def to_pennylane_state(re, im):
    flat_re = re.reshape(-1).cpu().numpy().astype(np.complex128)
    flat_im = im.reshape(-1).cpu().numpy().astype(np.complex128)
    return flat_re + 1j * flat_im


def run_pennylane(nq, ops_fn, init_re, init_im):
    init_state = to_pennylane_state(init_re, init_im)
    dev = qml.device("default.qubit", wires=nq)
    with qml.queuing.QueuingManager.stop_recording():
        ops = ops_fn()

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(init_state, wires=range(nq), normalize=False)
        for op in ops:
            qml.apply(op)
        return qml.state()

    return circuit()


def compare(label, re_ours, im_ours, ref_state, tol=TOL):
    ours_complex = re_ours.reshape(-1).cpu().numpy() + 1j * im_ours.reshape(-1).cpu().numpy()
    diff = np.abs(ours_complex - ref_state).max()
    status = "OK" if diff < tol else "FAIL"
    print(f"  {status}  {label:50s}  max |diff| = {diff:.3e}")
    return diff < tol


def test_single_qubit_layer():
    nq = 5
    torch.manual_seed(42)
    re0, im0 = make_random_state(nq, seed=0)
    ry_angles = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.5
    rz_angles = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.5

    def ops():
        ops_list = []
        for q in range(nq):
            ops_list.append(qml.RY(ry_angles[q].item(), wires=q))
            ops_list.append(qml.RZ(rz_angles[q].item(), wires=q))
        return ops_list
    ref = run_pennylane(nq, ops, re0, im0)

    G_re, G_im = gf.build_ry_rz_gates(ry_angles, rz_angles)
    re_t, im_t = gf.apply_su2_layer(re0.clone(), im0.clone(), G_re, G_im)
    compare("apply_su2_layer (RY then RZ per qubit)", re_t, im_t, ref)


def test_individual_single_qubit_builders():
    nq = 4
    re0, im0 = make_random_state(nq, seed=1)

    def apply_single(G2_re, G2_im, q):
        I_re = torch.eye(2, device=DEVICE, dtype=DTYPE).unsqueeze(0).expand(nq, 2, 2).contiguous()
        I_im = torch.zeros(nq, 2, 2, device=DEVICE, dtype=DTYPE)
        I_re = I_re.clone(); I_im = I_im.clone()
        I_re[q] = G2_re
        I_im[q] = G2_im
        return gf.apply_su2_layer(re0.clone(), im0.clone(), I_re, I_im)

    cases = [
        ("RY", lambda: gf.build_ry_gates(torch.tensor([0.7], device=DEVICE, dtype=DTYPE)), [qml.RY(0.7, wires=1)], 1),
        ("RZ", lambda: gf.build_rz_gates(torch.tensor([1.2], device=DEVICE, dtype=DTYPE)), [qml.RZ(1.2, wires=2)], 2),
        ("RX", lambda: gf.build_rx_gates(torch.tensor([0.9], device=DEVICE, dtype=DTYPE)), [qml.RX(0.9, wires=0)], 0),
        ("PhaseShift", lambda: gf.build_phase_shift_gates(torch.tensor([1.1], device=DEVICE, dtype=DTYPE)),
         [qml.PhaseShift(1.1, wires=3)], 3),
        ("Hadamard", lambda: gf.build_h_gates(1, DEVICE, DTYPE), [qml.Hadamard(wires=2)], 2),
        ("PauliX", lambda: gf.build_x_gates(1, DEVICE, DTYPE), [qml.PauliX(wires=0)], 0),
        ("PauliY", lambda: gf.build_y_gates(1, DEVICE, DTYPE), [qml.PauliY(wires=1)], 1),
        ("PauliZ", lambda: gf.build_z_gates(1, DEVICE, DTYPE), [qml.PauliZ(wires=3)], 3),
        ("S",        lambda: gf.build_s_gates(1, DEVICE, DTYPE), [qml.S(wires=2)], 2),
        ("T",        lambda: gf.build_t_gates(1, DEVICE, DTYPE), [qml.T(wires=1)], 1),
    ]
    for name, builder, qml_ops, q in cases:
        G_re_one, G_im_one = builder()
        G2_re = G_re_one[0] if G_re_one.dim() == 3 else G_re_one
        G2_im = G_im_one[0] if G_im_one.dim() == 3 else G_im_one
        re_t, im_t = apply_single(G2_re, G2_im, q)
        ref = run_pennylane(nq, lambda: qml_ops, re0, im0)
        compare(f"single-qubit {name} on q={q}", re_t, im_t, ref)


def test_rot():
    nq = 4
    re0, im0 = make_random_state(nq, seed=2)
    phi   = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.0
    theta = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.0
    omega = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.0
    G_re, G_im = gf.build_rot_gates(phi, theta, omega)

    def ops():
        return [qml.Rot(phi[q].item(), theta[q].item(), omega[q].item(), wires=q) for q in range(nq)]
    ref = run_pennylane(nq, ops, re0, im0)
    re_t, im_t = gf.apply_su2_layer(re0.clone(), im0.clone(), G_re, G_im)
    compare("Rot(phi,theta,omega) per qubit layer", re_t, im_t, ref)


def test_two_qubit_gates_single():
    nq = 5
    re0, im0 = make_random_state(nq, seed=3)

    def apply_via_plan(builder_fn, n_pairs_args, edges, qml_op_factory):
        plans = gf.compute_two_qubit_plans(edges, nq, DEVICE, DTYPE, commute_fn=gf.cry_commute)
        re_t, im_t = re0.clone(), im0.clone()
        for plan in plans:
            G_re, G_im = builder_fn(*n_pairs_args(plan))
            re_t, im_t = gf.apply_disjoint_two_qubit_layer(re_t, im_t, plan, G_re, G_im)
        ref = run_pennylane(nq, qml_op_factory, re0, im0)
        return re_t, im_t, ref

    theta = 0.85; phi = 1.2
    parametric_cases = [
        ("CRY",    gf.build_cry_4x4,    lambda p: (torch.tensor([theta], device=DEVICE, dtype=DTYPE),),  [(0, 2)],
            lambda: [qml.CRY(theta, wires=[0, 2])]),
        ("CRX",    gf.build_crx_4x4,    lambda p: (torch.tensor([theta], device=DEVICE, dtype=DTYPE),),  [(1, 3)],
            lambda: [qml.CRX(theta, wires=[1, 3])]),
        ("CRZ",    gf.build_crz_4x4,    lambda p: (torch.tensor([theta], device=DEVICE, dtype=DTYPE),),  [(2, 4)],
            lambda: [qml.CRZ(theta, wires=[2, 4])]),
        ("CPhase", gf.build_cphase_4x4, lambda p: (torch.tensor([phi],   device=DEVICE, dtype=DTYPE),),  [(0, 3)],
            lambda: [qml.ControlledPhaseShift(phi, wires=[0, 3])]),
        ("IsingZZ", gf.build_ising_zz_4x4, lambda p: (torch.tensor([theta], device=DEVICE, dtype=DTYPE),), [(1, 4)],
            lambda: [qml.IsingZZ(theta, wires=[1, 4])]),
        ("IsingXX", gf.build_ising_xx_4x4, lambda p: (torch.tensor([theta], device=DEVICE, dtype=DTYPE),), [(0, 4)],
            lambda: [qml.IsingXX(theta, wires=[0, 4])]),
        ("IsingYY", gf.build_ising_yy_4x4, lambda p: (torch.tensor([theta], device=DEVICE, dtype=DTYPE),), [(2, 3)],
            lambda: [qml.IsingYY(theta, wires=[2, 3])]),
    ]
    for name, b, args_fn, edges, qml_fac in parametric_cases:
        re_t, im_t, ref = apply_via_plan(b, args_fn, edges, qml_fac)
        compare(f"2q parametric {name} on {edges[0]}", re_t, im_t, ref)

    const_cases = [
        ("CNOT", gf.build_cnot_4x4, [(0, 2)], lambda: [qml.CNOT(wires=[0, 2])]),
        ("CY",   gf.build_cy_4x4,   [(1, 4)], lambda: [qml.CY(wires=[1, 4])]),
        ("CZ",   gf.build_cz_4x4,   [(0, 3)], lambda: [qml.CZ(wires=[0, 3])]),
        ("SWAP", gf.build_swap_4x4, [(1, 3)], lambda: [qml.SWAP(wires=[1, 3])]),
        ("ISWAP",gf.build_iswap_4x4,[(2, 4)], lambda: [qml.ISWAP(wires=[2, 4])]),
    ]
    for name, b, edges, qml_fac in const_cases:
        plans = gf.compute_two_qubit_plans(edges, nq, DEVICE, DTYPE, commute_fn=gf.cry_commute)
        re_t, im_t = re0.clone(), im0.clone()
        for plan in plans:
            G_re, G_im = b(plan['n_pairs'], DEVICE, DTYPE)
            re_t, im_t = gf.apply_disjoint_two_qubit_layer(re_t, im_t, plan, G_re, G_im)
        ref = run_pennylane(nq, qml_fac, re0, im0)
        compare(f"2q constant {name} on {edges[0]}", re_t, im_t, ref)


def test_multi_pair_disjoint_color():
    nq = 6
    re0, im0 = make_random_state(nq, seed=4)
    edges = [(0, 1), (2, 3), (4, 5)]
    angles = torch.tensor([0.3, 0.9, 1.4], device=DEVICE, dtype=DTYPE)

    plans = gf.compute_two_qubit_plans(edges, nq, DEVICE, DTYPE, commute_fn=gf.cry_commute)
    assert len(plans) == 1, f"expected 1 color, got {len(plans)}"
    G_re, G_im = gf.build_cry_4x4(angles)
    re_t, im_t = gf.apply_disjoint_two_qubit_layer(re0.clone(), im0.clone(), plans[0], G_re, G_im)

    def ops():
        return [qml.CRY(angles[i].item(), wires=list(e)) for i, e in enumerate(edges)]
    ref = run_pennylane(nq, ops, re0, im0)
    compare("3 disjoint CRYs in one color (n_pairs=3)", re_t, im_t, ref)


def test_chain_cry_multiple_colors():
    nq = 5
    re0, im0 = make_random_state(nq, seed=5)
    edges = [(1, 2), (2, 3), (3, 4)]
    angles = torch.tensor([0.6, 0.9, 1.1], device=DEVICE, dtype=DTYPE)

    plans = gf.compute_two_qubit_plans(edges, nq, DEVICE, DTYPE, commute_fn=gf.cry_commute)
    re_t, im_t = re0.clone(), im0.clone()
    for plan in plans:
        a = angles.index_select(0, plan['k_idx'])
        G_re, G_im = gf.build_cry_4x4(a)
        re_t, im_t = gf.apply_disjoint_two_qubit_layer(re_t, im_t, plan, G_re, G_im)

    def ops():
        return [qml.CRY(angles[i].item(), wires=list(e)) for i, e in enumerate(edges)]
    ref = run_pennylane(nq, ops, re0, im0)
    compare(f"chain CRY (3 edges, {len(plans)} colors)", re_t, im_t, ref)


def test_diagonal_packing_with_cphase():
    nq = 5
    re0, im0 = make_random_state(nq, seed=6)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    phis = torch.tensor([0.3, 0.5, 0.7, 0.9, 1.1], device=DEVICE, dtype=DTYPE)

    plans = gf.compute_two_qubit_plans(edges, nq, DEVICE, DTYPE, commute_fn=gf.diagonal_commute)
    re_t, im_t = re0.clone(), im0.clone()
    for plan in plans:
        a = phis.index_select(0, plan['k_idx'])
        G_re, G_im = gf.build_cphase_4x4(a)
        re_t, im_t = gf.apply_disjoint_two_qubit_layer(re_t, im_t, plan, G_re, G_im)

    def ops():
        return [qml.ControlledPhaseShift(phis[i].item(), wires=list(e)) for i, e in enumerate(edges)]
    ref = run_pennylane(nq, ops, re0, im0)
    compare(f"5 overlapping CPhase ({len(plans)} colors via diag commute)", re_t, im_t, ref)


def test_single_qubit_layer_chunked():
    nq = 9
    re0, im0 = make_random_state(nq, seed=7)
    ry_angles = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.5
    rz_angles = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.5

    def ops():
        ops_list = []
        for q in range(nq):
            ops_list.append(qml.RY(ry_angles[q].item(), wires=q))
            ops_list.append(qml.RZ(rz_angles[q].item(), wires=q))
        return ops_list
    ref = run_pennylane(nq, ops, re0, im0)

    G_re, G_im = gf.build_ry_rz_gates(ry_angles, rz_angles)
    re_t, im_t = gf.apply_su2_layer(re0.clone(), im0.clone(), G_re, G_im)
    n_win = len(gf._su2_windows(nq, gf.MAX_WINDOW))
    compare(f"apply_su2_layer nq={nq} chunked ({n_win} windows)", re_t, im_t, ref)


def test_windowed_color_with_long_span_pair():
    nq = 9
    re0, im0 = make_random_state(nq, seed=8)
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 8)]
    angles = torch.randn(len(edges), device=DEVICE, dtype=DTYPE) * 1.5

    plans = gf.compute_two_qubit_plans(edges, nq, DEVICE, DTYPE, commute_fn=gf.cry_commute)
    re_t, im_t = re0.clone(), im0.clone()
    for plan in plans:
        a = angles.index_select(0, plan['k_idx'])
        G_re, G_im = gf.build_cry_4x4(a)
        re_t, im_t = gf.apply_disjoint_two_qubit_layer(re_t, im_t, plan, G_re, G_im)

    def ops():
        return [qml.CRY(angles[i].item(), wires=list(e)) for i, e in enumerate(edges)]
    ref = run_pennylane(nq, ops, re0, im0)
    n_win = sum(len(p['windows']) for p in plans)
    n_long = sum(len(p['long']['windows']) for p in plans if 'long' in p)
    compare(f"4 adjacent CRYs + CRY(1,8) ({n_win} windows, {n_long} via permute)", re_t, im_t, ref)


def test_diagonal_builders():
    nq = 6
    re0, im0 = make_random_state(nq, seed=9)
    th = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.5
    pairs = [(0, 1), (1, 2), (3, 5), (4, 0)]
    ph = torch.randn(len(pairs), device=DEVICE, dtype=DTYPE) * 1.5

    cases = [
        ("build_rz_diag", gf.build_rz_diag(th, nq),
         lambda: [qml.RZ(th[q].item(), wires=q) for q in range(nq)]),
        ("build_phase_shift_diag", gf.build_phase_shift_diag(th, nq),
         lambda: [qml.PhaseShift(th[q].item(), wires=q) for q in range(nq)]),
        ("build_ising_zz_diag (4 overlapping pairs)", gf.build_ising_zz_diag(ph, pairs, nq),
         lambda: [qml.IsingZZ(ph[i].item(), wires=list(p)) for i, p in enumerate(pairs)]),
        ("build_cphase_diag (4 overlapping pairs)", gf.build_cphase_diag(ph, pairs, nq),
         lambda: [qml.ControlledPhaseShift(ph[i].item(), wires=list(p)) for i, p in enumerate(pairs)]),
        ("build_cz_diag (4 overlapping pairs)", gf.build_cz_diag(pairs, nq, DEVICE, DTYPE),
         lambda: [qml.CZ(wires=list(p)) for p in pairs]),
    ]
    for name, (d_re, d_im), ops in cases:
        re_t, im_t = gf.apply_diagonal_layer(re0.clone(), im0.clone(), d_re, d_im)
        ref = run_pennylane(nq, ops, re0, im0)
        compare(name, re_t, im_t, ref)


def test_fused_sequence():
    nq = 9
    re0, im0 = make_random_state(nq, seed=11)
    ry = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.5
    rz = torch.randn(nq, device=DEVICE, dtype=DTYPE) * 1.5
    cry = torch.randn(nq - 1, device=DEVICE, dtype=DTYPE) * 1.5
    crx = torch.randn(1, device=DEVICE, dtype=DTYPE) * 1.5

    G_re, G_im = gf.build_ry_rz_gates(ry, rz)
    C_re, C_im = gf.build_cry_4x4(cry)
    T_re, T_im = gf.build_toffoli_8x8(1, DEVICE, DTYPE)
    X_re, X_im = gf.build_crx_4x4(crx)
    gates = ([(('su2', q), (q,)) for q in range(nq)] + [(('cry', q), (q, q + 1)) for q in range(nq - 1)]
             + [('toff', (0, 1, 2)), ('crx', (0, nq - 1))])
    mats = {('su2', q): (G_re[q], G_im[q]) for q in range(nq)}
    mats.update({('cry', q): (C_re[q], C_im[q]) for q in range(nq - 1)})
    mats['toff'] = (T_re[0], T_im[0])
    mats['crx'] = (X_re[0], X_im[0])
    plan = gf.compile_sequence(gates, nq, DEVICE, DTYPE)
    re_t, im_t = gf.apply_sequence(re0.clone(), im0.clone(), plan, mats)

    def ops():
        ops_list = []
        for q in range(nq):
            ops_list.append(qml.RY(ry[q].item(), wires=q))
            ops_list.append(qml.RZ(rz[q].item(), wires=q))
        ops_list += [qml.CRY(cry[q].item(), wires=[q, q + 1]) for q in range(nq - 1)]
        ops_list.append(qml.Toffoli(wires=[0, 1, 2]))
        ops_list.append(qml.CRX(crx[0].item(), wires=[0, nq - 1]))
        return ops_list
    ref = run_pennylane(nq, ops, re0, im0)
    compare(f"fused VQC layer + CRX(0,{nq - 1}) nq={nq} ({plan['n_windows']} windows, {plan['n_long']} long)",
            re_t, im_t, ref)


def test_toffoli_single():
    nq = 5
    re0, im0 = make_random_state(nq, seed=10)
    triple = (0, 2, 4)

    plans = gf.compute_three_qubit_plans([triple], nq, DEVICE, DTYPE)
    assert len(plans) == 1 and plans[0]['n_triples'] == 1
    G_re, G_im = gf.build_toffoli_8x8(1, DEVICE, DTYPE)
    re_t, im_t = gf.apply_disjoint_three_qubit_layer(
        re0.clone(), im0.clone(), plans[0], G_re, G_im)

    def ops():
        return [qml.Toffoli(wires=list(triple))]
    ref = run_pennylane(nq, ops, re0, im0)
    compare(f"Toffoli on {triple}", re_t, im_t, ref)


def test_cswap_single():
    nq = 5
    re0, im0 = make_random_state(nq, seed=11)
    triple = (1, 2, 4)

    plans = gf.compute_three_qubit_plans([triple], nq, DEVICE, DTYPE,
                                          commute_fn=gf.cswap_commute)
    G_re, G_im = gf.build_cswap_8x8(1, DEVICE, DTYPE)
    re_t, im_t = gf.apply_disjoint_three_qubit_layer(
        re0.clone(), im0.clone(), plans[0], G_re, G_im)

    def ops():
        return [qml.CSWAP(wires=list(triple))]
    ref = run_pennylane(nq, ops, re0, im0)
    compare(f"CSWAP on {triple}", re_t, im_t, ref)


def test_multi_toffoli_disjoint():
    nq = 7
    re0, im0 = make_random_state(nq, seed=12)
    triples = [(0, 1, 2), (3, 4, 5)]

    plans = gf.compute_three_qubit_plans(triples, nq, DEVICE, DTYPE)
    assert len(plans) == 1, f"expected 1 color, got {len(plans)}"
    G_re, G_im = gf.build_toffoli_8x8(2, DEVICE, DTYPE)
    re_t, im_t = gf.apply_disjoint_three_qubit_layer(
        re0.clone(), im0.clone(), plans[0], G_re, G_im)

    def ops():
        return [qml.Toffoli(wires=list(t)) for t in triples]
    ref = run_pennylane(nq, ops, re0, im0)
    compare(f"2 disjoint Toffolis (n_triples=2)", re_t, im_t, ref)


def test_overlapping_toffoli_multi_color():
    nq = 5
    re0, im0 = make_random_state(nq, seed=13)
    triples = [(0, 1, 2), (1, 2, 3)]
    plans = gf.compute_three_qubit_plans(triples, nq, DEVICE, DTYPE)

    re_t, im_t = re0.clone(), im0.clone()
    for plan in plans:
        n = plan['n_triples']
        G_re, G_im = gf.build_toffoli_8x8(n, DEVICE, DTYPE)
        re_t, im_t = gf.apply_disjoint_three_qubit_layer(re_t, im_t, plan, G_re, G_im)

    def ops():
        return [qml.Toffoli(wires=list(t)) for t in triples]
    ref = run_pennylane(nq, ops, re0, im0)
    compare(f"chain Toffoli (2 edges, {len(plans)} colors)", re_t, im_t, ref)


def test_per_batch_su2_chain():
    nq = 5
    B = 4
    g = torch.Generator(device=DEVICE).manual_seed(7)
    re0 = torch.randn(B, *([2] * nq), device=DEVICE, dtype=DTYPE, generator=g)
    im0 = torch.randn(B, *([2] * nq), device=DEVICE, dtype=DTYPE, generator=g)
    nm = torch.sqrt((re0.reshape(B, -1)**2 + im0.reshape(B, -1)**2).sum(dim=1, keepdim=True))
    re0 = re0 / nm.reshape(B, *([1] * nq))
    im0 = im0 / nm.reshape(B, *([1] * nq))
    angles = torch.randn(B, nq, device=DEVICE, dtype=DTYPE, generator=g) * 1.5

    G_re, G_im = gf.build_ry_per_batch(angles)
    re_t, im_t = gf.apply_per_batch_su2_chain(re0.clone(), im0.clone(), G_re, G_im)

    all_match = True
    for b in range(B):
        re_b = re0[b:b+1]
        im_b = im0[b:b+1]
        def ops():
            return [qml.RY(angles[b, q].item(), wires=q) for q in range(nq)]
        ref = run_pennylane(nq, ops, re_b, im_b)
        ours_b = re_t[b].reshape(-1).cpu().numpy() + 1j * im_t[b].reshape(-1).cpu().numpy()
        diff = np.abs(ours_b - ref).max()
        if diff > TOL:
            all_match = False
            print(f"  FAIL  per_batch_chain batch {b}: max |diff| = {diff:.3e}")
    if all_match:
        print(f"  OK    apply_per_batch_su2_chain (B={B}, nq={nq}): all batches match")


if __name__ == "__main__":
    print("=== Single-qubit gates ===")
    test_individual_single_qubit_builders()
    test_rot()
    test_single_qubit_layer()
    print()
    print("=== 2-qubit gates (single, via 1-color plan) ===")
    test_two_qubit_gates_single()
    print()
    print("=== 2-qubit gates (multi-pair colors) ===")
    test_multi_pair_disjoint_color()
    test_chain_cry_multiple_colors()
    test_diagonal_packing_with_cphase()
    print()
    print("=== Windowed layers (nq > MAX_WINDOW) ===")
    test_single_qubit_layer_chunked()
    test_windowed_color_with_long_span_pair()
    print()
    print("=== Diagonal layers ===")
    test_diagonal_builders()
    print()
    print("=== Time-fused sequences ===")
    test_fused_sequence()
    print()
    print("=== 3-qubit gates (Toffoli, CSWAP) ===")
    test_toffoli_single()
    test_cswap_single()
    test_multi_toffoli_disjoint()
    test_overlapping_toffoli_multi_color()
    print()
    print("=== Per-batch ===")
    test_per_batch_su2_chain()

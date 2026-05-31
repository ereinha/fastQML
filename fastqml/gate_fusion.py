import math
from typing import Callable, Tuple

import torch
try:
    import torch._inductor.config as _ic
    _ic.triton.cudagraphs = False
except Exception:
    pass


def build_ry_gates(angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    G_re = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    return G_re, torch.zeros_like(G_re)


def build_rz_gates(angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    z = torch.zeros_like(c)
    G_re = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    G_im = torch.stack([torch.stack([-s, z], -1), torch.stack([z, s], -1)], -2)
    return G_re, G_im


def build_rx_gates(angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    z = torch.zeros_like(c)
    G_re = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    G_im = torch.stack([torch.stack([z, -s], -1), torch.stack([-s, z], -1)], -2)
    return G_re, G_im


def build_phase_shift_gates(phis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c, s = torch.cos(phis), torch.sin(phis)
    one = torch.ones_like(c); z = torch.zeros_like(c)
    G_re = torch.stack([torch.stack([one, z], -1), torch.stack([z,  c], -1)], -2)
    G_im = torch.stack([torch.stack([z,   z], -1), torch.stack([z,  s], -1)], -2)
    return G_re, G_im


def build_rot_gates(phi: torch.Tensor, theta: torch.Tensor,
                    omega: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    Gr_phi, Gi_phi = build_rz_gates(phi)
    Gr_th,  Gi_th  = build_ry_gates(theta)
    Gr_om,  Gi_om  = build_rz_gates(omega)
    a_re = torch.einsum('nij,njk->nik', Gr_th, Gr_phi) - torch.einsum('nij,njk->nik', Gi_th, Gi_phi)
    a_im = torch.einsum('nij,njk->nik', Gr_th, Gi_phi) + torch.einsum('nij,njk->nik', Gi_th, Gr_phi)
    out_re = torch.einsum('nij,njk->nik', Gr_om, a_re) - torch.einsum('nij,njk->nik', Gi_om, a_im)
    out_im = torch.einsum('nij,njk->nik', Gr_om, a_im) + torch.einsum('nij,njk->nik', Gi_om, a_re)
    return out_re, out_im


def build_ry_rz_gates(ry_angles: torch.Tensor,
                      rz_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cy, sy = torch.cos(ry_angles / 2), torch.sin(ry_angles / 2)
    cz, sz = torch.cos(rz_angles / 2), torch.sin(rz_angles / 2)
    G_re = torch.stack([
        torch.stack([cz * cy, -cz * sy], -1),
        torch.stack([cz * sy,  cz * cy], -1),
    ], -2)
    G_im = torch.stack([
        torch.stack([-sz * cy,  sz * sy], -1),
        torch.stack([ sz * sy,  sz * cy], -1),
    ], -2)
    return G_re, G_im


def _const_2x2(re_vals, im_vals, nq, device, dtype):
    G_re = torch.tensor(re_vals, device=device, dtype=dtype).unsqueeze(0).expand(nq, 2, 2)
    G_im = torch.tensor(im_vals, device=device, dtype=dtype).unsqueeze(0).expand(nq, 2, 2)
    return G_re.contiguous(), G_im.contiguous()


def build_x_gates(nq, device, dtype):
    return _const_2x2([[0, 1], [1, 0]], [[0, 0], [0, 0]], nq, device, dtype)


def build_y_gates(nq, device, dtype):
    return _const_2x2([[0, 0], [0, 0]], [[0, -1], [1, 0]], nq, device, dtype)


def build_z_gates(nq, device, dtype):
    return _const_2x2([[1, 0], [0, -1]], [[0, 0], [0, 0]], nq, device, dtype)


def build_h_gates(nq, device, dtype):
    s = 1.0 / math.sqrt(2.0)
    return _const_2x2([[s, s], [s, -s]], [[0, 0], [0, 0]], nq, device, dtype)


def build_s_gates(nq, device, dtype):
    return _const_2x2([[1, 0], [0, 0]], [[0, 0], [0, 1]], nq, device, dtype)


def build_t_gates(nq, device, dtype):
    c = math.cos(math.pi / 4); s = math.sin(math.pi / 4)
    return _const_2x2([[1, 0], [0, c]], [[0, 0], [0, s]], nq, device, dtype)


def build_ry_per_batch(angles):
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    G_re = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    return G_re, torch.zeros_like(G_re)


def build_rz_per_batch(angles):
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    z = torch.zeros_like(c)
    G_re = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    G_im = torch.stack([torch.stack([-s, z], -1), torch.stack([z, s], -1)], -2)
    return G_re, G_im


def build_rx_per_batch(angles):
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    z = torch.zeros_like(c)
    G_re = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    G_im = torch.stack([torch.stack([z, -s], -1), torch.stack([-s, z], -1)], -2)
    return G_re, G_im


def build_ry_rx_per_batch(ry_angles, rx_angles):
    cy, sy = torch.cos(ry_angles / 2), torch.sin(ry_angles / 2)
    cx, sx = torch.cos(rx_angles / 2), torch.sin(rx_angles / 2)
    G_re = torch.stack([
        torch.stack([cx * cy, -cx * sy], -1),
        torch.stack([cx * sy,  cx * cy], -1),
    ], -2)
    G_im = torch.stack([
        torch.stack([-sx * sy, -sx * cy], -1),
        torch.stack([-sx * cy,  sx * sy], -1),
    ], -2)
    return G_re, G_im


def apply_su2_layer(re: torch.Tensor, im: torch.Tensor,
                    G_re: torch.Tensor, G_im: torch.Tensor):
    B = re.shape[0]
    nq = G_re.shape[0]
    U_re, U_im = G_re[0], G_im[0]
    for q in range(1, nq):
        Cre, Cim = G_re[q], G_im[q]
        new_re = torch.kron(U_re, Cre) - torch.kron(U_im, Cim)
        new_im = torch.kron(U_re, Cim) + torch.kron(U_im, Cre)
        U_re, U_im = new_re, new_im
    re_flat = re.reshape(B, -1)
    im_flat = im.reshape(B, -1)
    Ut_re, Ut_im = U_re.T, U_im.T
    new_re = re_flat @ Ut_re - im_flat @ Ut_im
    new_im = re_flat @ Ut_im + im_flat @ Ut_re
    return new_re.reshape(re.shape), new_im.reshape(im.shape)


def apply_per_batch_su2_chain(re, im, G_re, G_im):
    nq = G_re.shape[1]
    for q in range(nq):
        re = re.movedim(q + 1, -1)
        im = im.movedim(q + 1, -1)
        Gr, Gi = G_re[:, q], G_im[:, q]
        new_re = torch.einsum('bij,b...j->b...i', Gr, re) - torch.einsum('bij,b...j->b...i', Gi, im)
        new_im = torch.einsum('bij,b...j->b...i', Gr, im) + torch.einsum('bij,b...j->b...i', Gi, re)
        re = new_re.movedim(-1, q + 1)
        im = new_im.movedim(-1, q + 1)
    return re, im


def _zero_4x4(n_pairs, device, dtype):
    return torch.zeros(n_pairs, 4, 4, device=device, dtype=dtype)


def build_cry_4x4(angles):
    n = angles.shape[0]
    dev, dt = angles.device, angles.dtype
    c, s = torch.cos(angles / 2), torch.sin(angles / 2)
    G_re = _zero_4x4(n, dev, dt)
    G_re[:, 0, 0] = 1.0; G_re[:, 1, 1] = 1.0
    G_re[:, 2, 2] = c;   G_re[:, 2, 3] = -s
    G_re[:, 3, 2] = s;   G_re[:, 3, 3] = c
    return G_re, _zero_4x4(n, dev, dt)


def build_crx_4x4(angles):
    n = angles.shape[0]
    dev, dt = angles.device, angles.dtype
    c, s = torch.cos(angles / 2), torch.sin(angles / 2)
    G_re = _zero_4x4(n, dev, dt)
    G_im = _zero_4x4(n, dev, dt)
    G_re[:, 0, 0] = 1.0; G_re[:, 1, 1] = 1.0
    G_re[:, 2, 2] = c;   G_re[:, 3, 3] = c
    G_im[:, 2, 3] = -s;  G_im[:, 3, 2] = -s
    return G_re, G_im


def build_crz_4x4(angles):
    n = angles.shape[0]
    dev, dt = angles.device, angles.dtype
    c, s = torch.cos(angles / 2), torch.sin(angles / 2)
    G_re = _zero_4x4(n, dev, dt)
    G_im = _zero_4x4(n, dev, dt)
    G_re[:, 0, 0] = 1.0; G_re[:, 1, 1] = 1.0
    G_re[:, 2, 2] = c;   G_re[:, 3, 3] = c
    G_im[:, 2, 2] = -s;  G_im[:, 3, 3] = s
    return G_re, G_im


def build_cphase_4x4(phis):
    n = phis.shape[0]
    dev, dt = phis.device, phis.dtype
    c, s = torch.cos(phis), torch.sin(phis)
    G_re = _zero_4x4(n, dev, dt)
    G_im = _zero_4x4(n, dev, dt)
    G_re[:, 0, 0] = 1.0; G_re[:, 1, 1] = 1.0; G_re[:, 2, 2] = 1.0
    G_re[:, 3, 3] = c;   G_im[:, 3, 3] = s
    return G_re, G_im


def build_ising_zz_4x4(angles):
    n = angles.shape[0]
    dev, dt = angles.device, angles.dtype
    c, s = torch.cos(angles / 2), torch.sin(angles / 2)
    G_re = _zero_4x4(n, dev, dt)
    G_im = _zero_4x4(n, dev, dt)
    G_re[:, 0, 0] = c;  G_im[:, 0, 0] = -s
    G_re[:, 1, 1] = c;  G_im[:, 1, 1] =  s
    G_re[:, 2, 2] = c;  G_im[:, 2, 2] =  s
    G_re[:, 3, 3] = c;  G_im[:, 3, 3] = -s
    return G_re, G_im


def build_ising_xx_4x4(angles):
    n = angles.shape[0]
    dev, dt = angles.device, angles.dtype
    c, s = torch.cos(angles / 2), torch.sin(angles / 2)
    G_re = _zero_4x4(n, dev, dt); G_im = _zero_4x4(n, dev, dt)
    G_re[:, 0, 0] = c; G_re[:, 1, 1] = c; G_re[:, 2, 2] = c; G_re[:, 3, 3] = c
    G_im[:, 0, 3] = -s; G_im[:, 3, 0] = -s
    G_im[:, 1, 2] = -s; G_im[:, 2, 1] = -s
    return G_re, G_im


def build_ising_yy_4x4(angles):
    n = angles.shape[0]
    dev, dt = angles.device, angles.dtype
    c, s = torch.cos(angles / 2), torch.sin(angles / 2)
    G_re = _zero_4x4(n, dev, dt); G_im = _zero_4x4(n, dev, dt)
    G_re[:, 0, 0] = c; G_re[:, 1, 1] = c; G_re[:, 2, 2] = c; G_re[:, 3, 3] = c
    G_im[:, 0, 3] =  s; G_im[:, 3, 0] =  s
    G_im[:, 1, 2] = -s; G_im[:, 2, 1] = -s
    return G_re, G_im


def _const_4x4(re_vals, im_vals, n_pairs, device, dtype):
    G_re = torch.tensor(re_vals, device=device, dtype=dtype).unsqueeze(0).expand(n_pairs, 4, 4)
    G_im = torch.tensor(im_vals, device=device, dtype=dtype).unsqueeze(0).expand(n_pairs, 4, 4)
    return G_re.contiguous(), G_im.contiguous()


def build_cnot_4x4(n_pairs, device, dtype):
    z = [[0.] * 4 for _ in range(4)]
    re = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    return _const_4x4(re, z, n_pairs, device, dtype)


def build_cy_4x4(n_pairs, device, dtype):
    re = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    im = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]
    return _const_4x4(re, im, n_pairs, device, dtype)


def build_cz_4x4(n_pairs, device, dtype):
    re = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    z = [[0.] * 4 for _ in range(4)]
    return _const_4x4(re, z, n_pairs, device, dtype)


def build_swap_4x4(n_pairs, device, dtype):
    re = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    z = [[0.] * 4 for _ in range(4)]
    return _const_4x4(re, z, n_pairs, device, dtype)


def build_iswap_4x4(n_pairs, device, dtype):
    re = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    im = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    return _const_4x4(re, im, n_pairs, device, dtype)


def cry_commute(e1, e2):
    a, b = e1
    c, d = e2
    return not (a == d or b == c)


def diagonal_commute(e1, e2):
    return True


def symmetric_commute(e1, e2):
    a, b = e1; c, d = e2
    return not ({a, b} & {c, d})


def greedy_edge_coloring(edges, commute_fn: Callable = cry_commute):
    colors = []
    for k, edge in enumerate(edges):
        qa, qb = edge
        placed = None
        for c in range(len(colors)):
            disjoint = all(qa not in e and qb not in e for _, e in colors[c])
            if not disjoint:
                continue
            ok = True
            for c2 in range(c + 1, len(colors)):
                for _, e_other in colors[c2]:
                    if not commute_fn(edge, e_other):
                        ok = False; break
                if not ok: break
            if ok:
                placed = c; break
        if placed is not None:
            colors[placed].append((k, edge))
        else:
            colors.append([(k, edge)])
    return colors


def identity_pad_gates(G_re, G_im, qubit_indices, nq):
    device, dtype = G_re.device, G_re.dtype
    eye2 = torch.eye(2, device=device, dtype=dtype)
    zero2 = torch.zeros(2, 2, device=device, dtype=dtype)
    qubit_set = set(qubit_indices)
    re_slabs, im_slabs = [], []
    g = 0
    for q in range(nq):
        if q in qubit_set:
            re_slabs.append(G_re[g]); im_slabs.append(G_im[g]); g += 1
        else:
            re_slabs.append(eye2); im_slabs.append(zero2)
    return torch.stack(re_slabs, dim=0), torch.stack(im_slabs, dim=0)


def apply_disjoint_cry_layer(re, im, plan, angles):
    G_re, G_im = build_cry_4x4(angles)
    return apply_disjoint_two_qubit_layer(re, im, plan, G_re, G_im)


def _const_8x8(re_vals, im_vals, n_triples, device, dtype):
    G_re = torch.tensor(re_vals, device=device, dtype=dtype).unsqueeze(0).expand(n_triples, 8, 8)
    G_im = torch.tensor(im_vals, device=device, dtype=dtype).unsqueeze(0).expand(n_triples, 8, 8)
    return G_re.contiguous(), G_im.contiguous()


def build_toffoli_8x8(n_triples, device, dtype):
    re = [[0.0] * 8 for _ in range(8)]
    for i in range(6):
        re[i][i] = 1.0
    re[6][7] = 1.0
    re[7][6] = 1.0
    im = [[0.0] * 8 for _ in range(8)]
    return _const_8x8(re, im, n_triples, device, dtype)


def build_cswap_8x8(n_triples, device, dtype):
    re = [[0.0] * 8 for _ in range(8)]
    for i in range(4):
        re[i][i] = 1.0
    re[4][4] = 1.0
    re[5][6] = 1.0
    re[6][5] = 1.0
    re[7][7] = 1.0
    im = [[0.0] * 8 for _ in range(8)]
    return _const_8x8(re, im, n_triples, device, dtype)


def toffoli_commute(t1, t2):
    c11, c12, t1_ = t1
    c21, c22, t2_ = t2
    return not (t1_ in (c21, c22) or t2_ in (c11, c12))


def cswap_commute(t1, t2):
    return not (set(t1) & set(t2))


def greedy_triple_coloring(triples, commute_fn: Callable = toffoli_commute):
    colors = []
    for k, triple in enumerate(triples):
        q_set = set(triple)
        placed = None
        for c in range(len(colors)):
            disjoint = all(not (q_set & set(t)) for _, t in colors[c])
            if not disjoint:
                continue
            ok = True
            for c2 in range(c + 1, len(colors)):
                for _, t_other in colors[c2]:
                    if not commute_fn(triple, t_other):
                        ok = False; break
                if not ok: break
            if ok:
                placed = c; break
        if placed is not None:
            colors[placed].append((k, triple))
        else:
            colors.append([(k, triple)])
    return colors


_triple_plan_cache: dict = {}


def compute_three_qubit_plans(triples, nq: int, device, dtype,
                              commute_fn: Callable = toffoli_commute):
    if len(triples) == 0:
        return []
    key = (tuple(triples), nq, str(device), str(dtype), commute_fn.__name__)
    if key in _triple_plan_cache:
        return _triple_plan_cache[key]
    colors = greedy_triple_coloring(triples, commute_fn)
    plans = []
    for color in colors:
        k_indices = [k for k, _ in color]
        color_triples = [t for _, t in color]
        tripled = [q for t in color_triples for q in t]
        unpaired = [q for q in range(nq) if q not in set(tripled)]
        new_order = tripled + unpaired
        perm = [0] + [q + 1 for q in new_order]
        inv_perm = [0] * (nq + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        plans.append({
            'n_triples': len(color_triples),
            'n_unpaired': len(unpaired),
            'dim': 1 << nq,
            'k_idx': torch.tensor(k_indices, dtype=torch.long, device=device),
            'perm': tuple(perm),
            'inv_perm': tuple(inv_perm),
            'triples': color_triples,
        })
    _triple_plan_cache[key] = plans
    return plans


def apply_disjoint_three_qubit_layer(re, im, plan, G_re, G_im):
    n_triples = plan['n_triples']
    n_unpaired = plan['n_unpaired']
    DIM = plan['dim']
    B = re.shape[0]
    device, dtype = re.device, re.dtype

    U_re = G_re[0]; U_im = G_im[0]
    for k in range(1, n_triples):
        Cre, Cim = G_re[k], G_im[k]
        new_re = torch.kron(U_re, Cre) - torch.kron(U_im, Cim)
        new_im = torch.kron(U_re, Cim) + torch.kron(U_im, Cre)
        U_re, U_im = new_re, new_im
    if n_unpaired > 0:
        I2 = torch.eye(2, device=device, dtype=dtype)
        for _ in range(n_unpaired):
            U_re = torch.kron(U_re, I2)
            U_im = torch.kron(U_im, I2)

    re_p = re.permute(plan['perm']).contiguous()
    im_p = im.permute(plan['perm']).contiguous()
    re_flat = re_p.reshape(B, DIM)
    im_flat = im_p.reshape(B, DIM)
    Ut_re, Ut_im = U_re.T, U_im.T
    new_re = re_flat @ Ut_re - im_flat @ Ut_im
    new_im = re_flat @ Ut_im + im_flat @ Ut_re
    out_shape = re_p.shape
    new_re = new_re.reshape(out_shape).permute(plan['inv_perm']).contiguous()
    new_im = new_im.reshape(out_shape).permute(plan['inv_perm']).contiguous()
    return new_re, new_im


_plan_cache: dict = {}


def compute_two_qubit_plans(edges, nq: int, device, dtype,
                            commute_fn: Callable = cry_commute):
    if len(edges) == 0:
        return []
    key = (tuple(edges), nq, str(device), str(dtype), commute_fn.__name__)
    if key in _plan_cache:
        return _plan_cache[key]
    colors = greedy_edge_coloring(edges, commute_fn)
    plans = []
    for color in colors:
        k_indices = [k for k, _ in color]
        color_edges = [e for _, e in color]
        paired = [q for qa, qb in color_edges for q in (qa, qb)]
        unpaired = [q for q in range(nq) if q not in set(paired)]
        new_order = paired + unpaired
        perm = [0] + [q + 1 for q in new_order]
        inv_perm = [0] * (nq + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        plans.append({
            'n_pairs': len(color_edges),
            'n_unpaired': len(unpaired),
            'dim': 1 << nq,
            'k_idx': torch.tensor(k_indices, dtype=torch.long, device=device),
            'perm': tuple(perm),
            'inv_perm': tuple(inv_perm),
            'edges': color_edges,
        })
    _plan_cache[key] = plans
    return plans


def apply_disjoint_two_qubit_layer(re, im, plan, G_re, G_im):
    n_pairs = plan['n_pairs']
    n_unpaired = plan['n_unpaired']
    DIM = plan['dim']
    B = re.shape[0]
    device, dtype = re.device, re.dtype

    U_re = G_re[0]; U_im = G_im[0]
    for k in range(1, n_pairs):
        Cre, Cim = G_re[k], G_im[k]
        new_re = torch.kron(U_re, Cre) - torch.kron(U_im, Cim)
        new_im = torch.kron(U_re, Cim) + torch.kron(U_im, Cre)
        U_re, U_im = new_re, new_im
    if n_unpaired > 0:
        I2 = torch.eye(2, device=device, dtype=dtype)
        for _ in range(n_unpaired):
            U_re = torch.kron(U_re, I2)
            U_im = torch.kron(U_im, I2)

    re_p = re.permute(plan['perm']).contiguous()
    im_p = im.permute(plan['perm']).contiguous()
    re_flat = re_p.reshape(B, DIM)
    im_flat = im_p.reshape(B, DIM)
    Ut_re, Ut_im = U_re.T, U_im.T
    new_re = re_flat @ Ut_re - im_flat @ Ut_im
    new_im = re_flat @ Ut_im + im_flat @ Ut_re
    out_shape = re_p.shape
    new_re = new_re.reshape(out_shape).permute(plan['inv_perm']).contiguous()
    new_im = new_im.reshape(out_shape).permute(plan['inv_perm']).contiguous()
    return new_re, new_im

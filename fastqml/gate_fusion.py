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


MAX_WINDOW = 7

def _kron_chain(G_re, G_im):
    U_re, U_im = G_re[0], G_im[0]
    for q in range(1, G_re.shape[0]):
        C_re, C_im = G_re[q], G_im[q]
        U_re, U_im = (torch.kron(U_re, C_re) - torch.kron(U_im, C_im),
                      torch.kron(U_re, C_im) + torch.kron(U_im, C_re))
    return U_re, U_im


try:
    from . import window_triton as _wt
except Exception:
    _wt = None


USE_TRITON_WINDOW = True

def _apply_window(re, im, U_re, U_im, start, width):

    if USE_TRITON_WINDOW and _wt is not None and _wt.usable(re, width):
        return _wt.apply_window(re, im, U_re, U_im, start, width)
    B = re.shape[0]
    nq = re.dim() - 1
    hi, lo = 1 << start, 1 << (nq - start - width)
    dim = 1 << width
    shape = re.shape
    if lo == 1 or lo < 16:
        if lo == 1:
            r, i = re.reshape(B * hi, dim), im.reshape(B * hi, dim)
        else:
            r = re.reshape(B * hi, dim, lo).transpose(1, 2).reshape(B * hi * lo, dim)
            i = im.reshape(B * hi, dim, lo).transpose(1, 2).reshape(B * hi * lo, dim)
        Ut_re, Ut_im = U_re.T, U_im.T
        new_re = torch.addmm(r @ Ut_re, i, Ut_im, alpha=-1.0)
        new_im = torch.addmm(r @ Ut_im, i, Ut_re)
        if lo > 1:
            new_re = new_re.reshape(B * hi, lo, dim).transpose(1, 2)
            new_im = new_im.reshape(B * hi, lo, dim).transpose(1, 2)
    else:
        r, i = re.reshape(B * hi, dim, lo), im.reshape(B * hi, dim, lo)
        Ue_re, Ue_im = U_re.expand(B * hi, dim, dim), U_im.expand(B * hi, dim, dim)
        new_re = torch.baddbmm(torch.bmm(Ue_re, r), Ue_im, i, alpha=-1.0)
        new_im = torch.baddbmm(torch.bmm(Ue_im, r), Ue_re, i)
    return new_re.reshape(shape), new_im.reshape(shape)


_SINGLE_WINDOW_UP_TO = 8

def _su2_windows(nq, max_window, single_up_to=None):
    if single_up_to is None:
        single_up_to = _SINGLE_WINDOW_UP_TO
    if nq <= max(max_window, single_up_to):
        return [(0, nq)]
    n_win = -(-nq // max_window)
    base, extra = divmod(nq, n_win)
    windows, start = [], 0
    for width in [base] * (n_win - extra) + [base + 1] * extra:
        windows.append((start, width))
        start += width
    return windows


def apply_su2_layer(re: torch.Tensor, im: torch.Tensor,
                    G_re: torch.Tensor, G_im: torch.Tensor, max_window: int = None):
    nq = G_re.shape[0]
    single_up_to = _SINGLE_WINDOW_UP_TO
    if USE_TRITON_WINDOW and _wt is not None and _wt.usable(re, _wt.MIN_WIDTH):
        single_up_to = min(single_up_to, _wt.MAX_WIDTH)
    for start, width in _su2_windows(nq, max_window or MAX_WINDOW, single_up_to):
        U_re, U_im = _kron_chain(G_re[start:start + width], G_im[start:start + width])
        re, im = _apply_window(re, im, U_re, U_im, start, width)
    return re, im



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


def _diagonal_from_phases(ph_re, ph_im, qubit_tuples, nq):
    d_re = d_im = None
    for k, qs in enumerate(qubit_tuples):
        g = len(qs)
        order = sorted(range(g), key=lambda i: qs[i])
        shape = [1] * nq
        for q in qs:
            shape[q] = 2
        p_re = ph_re[k].reshape((2,) * g).permute(order).reshape(shape)
        p_im = ph_im[k].reshape((2,) * g).permute(order).reshape(shape)
        if d_re is None:
            d_re, d_im = p_re, p_im
        else:
            d_re, d_im = d_re * p_re - d_im * p_im, d_re * p_im + d_im * p_re
    full = (2,) * nq
    return d_re.expand(full).reshape(-1), d_im.expand(full).reshape(-1)


def apply_diagonal_layer(re, im, d_re, d_im):
    shape = re.shape
    B = shape[0]
    r, i = re.reshape(B, -1), im.reshape(B, -1)
    return (r * d_re - i * d_im).reshape(shape), (r * d_im + i * d_re).reshape(shape)


def build_rz_diag(angles, nq):
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    return _diagonal_from_phases(torch.stack([c, c], -1), torch.stack([-s, s], -1),
                                 [(q,) for q in range(nq)], nq)


def build_phase_shift_diag(phis, nq):
    c, s = torch.cos(phis), torch.sin(phis)
    one, z = torch.ones_like(c), torch.zeros_like(c)
    return _diagonal_from_phases(torch.stack([one, c], -1), torch.stack([z, s], -1),
                                 [(q,) for q in range(nq)], nq)


def build_ising_zz_diag(angles, pairs, nq):
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    return _diagonal_from_phases(torch.stack([c, c, c, c], -1), torch.stack([-s, s, s, -s], -1),
                                 [tuple(p) for p in pairs], nq)


def build_cphase_diag(phis, pairs, nq):
    c, s = torch.cos(phis), torch.sin(phis)
    one, z = torch.ones_like(c), torch.zeros_like(c)
    return _diagonal_from_phases(torch.stack([one, one, one, c], -1), torch.stack([z, z, z, s], -1),
                                 [tuple(p) for p in pairs], nq)


def build_cz_diag(pairs, nq, device, dtype):
    n = len(pairs)
    ph_re = torch.tensor([1.0, 1.0, 1.0, -1.0], device=device, dtype=dtype).expand(n, 4)
    return _diagonal_from_phases(ph_re, torch.zeros(n, 4, device=device, dtype=dtype),
                                 [tuple(p) for p in pairs], nq)



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


_plan_cache: dict = {}
_triple_plan_cache: dict = {}


def _embed_gate(G_re, G_im, qubits, start, width, eye):
    others = [q for q in range(start, start + width) if q not in qubits]
    m = len(others)
    if m:
        G_re, G_im = torch.kron(G_re, eye[m]), torch.kron(G_im, eye[m])
    order = list(qubits) + others
    perm = [order.index(q) for q in range(start, start + width)]
    if perm != list(range(width)):
        full = perm + [p + width for p in perm]
        dim = 1 << width
        G_re = G_re.reshape((2,) * (2 * width)).permute(full).reshape(dim, dim)
        G_im = G_im.reshape((2,) * (2 * width)).permute(full).reshape(dim, dim)
    return G_re, G_im


def _window_operator(get, win, eye):
    parts = []
    for block in win['blocks']:
        if block[0] == 'gap':
            parts.append((eye[block[1]], None))
            continue
        _, bs, bw, gates = block
        W_re = W_im = None
        for gi, qs in gates:
            E_re, E_im = _embed_gate(*get(gi), qs, bs, bw, eye)
            if W_re is None:
                W_re, W_im = E_re, E_im
            else:
                W_re, W_im = E_re @ W_re - E_im @ W_im, E_re @ W_im + E_im @ W_re
        parts.append((W_re, W_im))
    U_re, U_im = parts[0]
    if U_im is None:
        U_im = torch.zeros_like(U_re)
    for P_re, P_im in parts[1:]:
        if P_im is None:
            U_re, U_im = torch.kron(U_re, P_re), torch.kron(U_im, P_re)
        else:
            U_re, U_im = (torch.kron(U_re, P_re) - torch.kron(U_im, P_im),
                          torch.kron(U_re, P_im) + torch.kron(U_im, P_re))
    return U_re, U_im


MIN_WINDOW = 4

def _runs(gates):
    runs = []
    for gi, qs in sorted(gates, key=lambda g: min(g[1])):
        lo, hi = min(qs), max(qs)
        if runs and lo <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], hi)
            runs[-1][2].append((gi, qs))
        else:
            runs.append([lo, hi, [(gi, qs)]])
    return runs


def _window_blocks(gates, start, end):
    blocks, pos = [], start
    for bs, be, rg in _runs(gates):
        if bs > pos:
            blocks.append(('gap', bs - pos))
        blocks.append(('gates', bs, be - bs + 1, rg))
        pos = be + 1
    if end + 1 > pos:
        blocks.append(('gap', end + 1 - pos))
    return blocks


def _pad_window(start, end, nq):
    pad = min(MIN_WINDOW, nq) - (end - start + 1)
    pad_right = min(max(pad, 0), nq - 1 - end)
    pad_left = max(pad, 0) - pad_right
    return start - pad_left, end + pad_right


def _window_plan(gates, max_window, nq):
    gates = sorted(gates, key=lambda g: min(g[1]))
    short = [g for g in gates if max(g[1]) - min(g[1]) + 1 <= max_window]
    long_ = [g for g in gates if max(g[1]) - min(g[1]) + 1 > max_window]
    packed = []
    for gi, qs in short:
        lo, hi = min(qs), max(qs)
        if packed and max(packed[-1][1], hi) - packed[-1][0] + 1 <= max_window:
            packed[-1][1] = max(packed[-1][1], hi)
            packed[-1][2].append((gi, qs))
        else:
            packed.append([lo, hi, [(gi, qs)]])
    windows = []
    for start, end, gs in packed:
        start, end = _pad_window(start, end, nq)
        windows.append({'start': start, 'width': end - start + 1, 'blocks': _window_blocks(gs, start, end)})
    return windows, long_


def _color_plan(color, nq, device, dtype, max_window):
    local = [(i, tuple(qs)) for i, (_, qs) in enumerate(color)]
    max_window = max(max_window, max(len(qs) for _, qs in local))
    windows, long_ = _window_plan(local, max_window, nq)
    plan = {'windows': windows}
    if long_:
        front = [q for _, qs in long_ for q in qs]
        new_order = front + [q for q in range(nq) if q not in set(front)]
        perm = [0] + [q + 1 for q in new_order]
        inv_perm = [0] * (nq + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        pos = {q: j for j, q in enumerate(new_order)}
        moved = [(gi, tuple(pos[q] for q in qs)) for gi, qs in long_]
        lw, rem = _window_plan(moved, max_window, nq)
        assert not rem
        plan['long'] = {'perm': tuple(perm), 'inv_perm': tuple(inv_perm), 'windows': lw}
    plan['eye'] = {m: torch.eye(1 << m, device=device, dtype=dtype) for m in range(1, max_window + 1)}
    return plan


def _apply_windowed(re, im, plan, G_re, G_im):
    eye = plan['eye']
    get = lambda gi: (G_re[gi], G_im[gi])
    for win in plan['windows']:
        U_re, U_im = _window_operator(get, win, eye)
        re, im = _apply_window(re, im, U_re, U_im, win['start'], win['width'])
    lp = plan.get('long')
    if lp:
        re, im = re.permute(lp['perm']).contiguous(), im.permute(lp['perm']).contiguous()
        for win in lp['windows']:
            U_re, U_im = _window_operator(get, win, eye)
            re, im = _apply_window(re, im, U_re, U_im, win['start'], win['width'])
        re, im = re.permute(lp['inv_perm']).contiguous(), im.permute(lp['inv_perm']).contiguous()
    return re, im


def compute_two_qubit_plans(edges, nq: int, device, dtype,
                            commute_fn: Callable = cry_commute, max_window: int = None):
    if len(edges) == 0:
        return []
    K = max_window or MAX_WINDOW
    key = (tuple(edges), nq, str(device), str(dtype), commute_fn.__name__, K)
    if key in _plan_cache:
        return _plan_cache[key]
    plans = []
    for color in greedy_edge_coloring(edges, commute_fn):
        plan = _color_plan(color, nq, device, dtype, K)
        plan.update({
            'n_pairs': len(color),
            'dim': 1 << nq,
            'k_idx': torch.tensor([k for k, _ in color], dtype=torch.long, device=device),
            'edges': [e for _, e in color],
        })
        plans.append(plan)
    _plan_cache[key] = plans
    return plans


def apply_disjoint_two_qubit_layer(re, im, plan, G_re, G_im):
    return _apply_windowed(re, im, plan, G_re, G_im)


def compute_three_qubit_plans(triples, nq: int, device, dtype,
                              commute_fn: Callable = toffoli_commute, max_window: int = None):
    if len(triples) == 0:
        return []
    K = max_window or MAX_WINDOW
    key = (tuple(triples), nq, str(device), str(dtype), commute_fn.__name__, K)
    if key in _triple_plan_cache:
        return _triple_plan_cache[key]
    plans = []
    for color in greedy_triple_coloring(triples, commute_fn):
        plan = _color_plan(color, nq, device, dtype, K)
        plan.update({
            'n_triples': len(color),
            'dim': 1 << nq,
            'k_idx': torch.tensor([k for k, _ in color], dtype=torch.long, device=device),
            'triples': [t for _, t in color],
        })
        plans.append(plan)
    _triple_plan_cache[key] = plans
    return plans


def apply_disjoint_three_qubit_layer(re, im, plan, G_re, G_im):
    return _apply_windowed(re, im, plan, G_re, G_im)


_seq_plan_cache: dict = {}


def _cmatmul(A_re, A_im, B_re, B_im):
    d = A_re.shape[0]
    P = torch.cat([torch.cat([A_re, -A_im], 1), torch.cat([A_im, A_re], 1)], 0)
    C = P @ torch.cat([B_re, B_im], 0)
    return C[:d], C[d:]


def _fused_operator(get, win, eye):
    start, end = win['start'], win['start'] + win['width'] - 1
    gates = win['gates']
    op, i = None, 0
    while i < len(gates):
        run, used = [gates[i]], set(gates[i][1])
        i += 1
        while i < len(gates) and not (set(gates[i][1]) & used):
            run.append(gates[i])
            used |= set(gates[i][1])
            i += 1
        R = _window_operator(get, {'blocks': _window_blocks(run, start, end)}, eye)
        op = R if op is None else _cmatmul(R[0], R[1], op[0], op[1])
    return op


def _fuse_plan(gates, nq, max_window):
    open_, steps = [], []
    order = 0

    def emit(ws):
        for w in sorted(ws, key=lambda w: w['order']):
            open_.remove(w)
            s, e = _pad_window(w['start'], w['end'], nq)
            steps.append({'kind': 'window', 'start': s, 'width': e - s + 1, 'gates': w['gates']})

    for gid, qs in gates:
        lo, hi = min(qs), max(qs)
        if hi - lo + 1 > max_window:
            emit(list(open_))
            steps.append({'kind': 'long', 'gate': (gid, qs)})
            continue
        touching = [w for w in open_ if not (hi < w['start'] or lo > w['end'])]
        ns = min([lo] + [w['start'] for w in touching])
        ne = max([hi] + [w['end'] for w in touching])
        if ne - ns + 1 <= max_window:
            emit([w for w in open_ if w not in touching and not (ne < w['start'] or ns > w['end'])])
            merged = {'start': ns, 'end': ne, 'order': order, 'gates': []}
            for w in sorted(touching, key=lambda w: w['order']):
                merged['gates'] += w['gates']
                open_.remove(w)
            merged['gates'].append((gid, qs))
            open_.append(merged)
        else:
            inside = [w for w in touching if lo <= w['start'] and w['end'] <= hi]
            emit([w for w in touching if w not in inside])
            new = {'start': lo, 'end': hi, 'order': order, 'gates': []}
            for w in sorted(inside, key=lambda w: w['order']):
                new['gates'] += w['gates']
                open_.remove(w)
            new['gates'].append((gid, qs))
            open_.append(new)
        order += 1
    emit(list(open_))
    return steps


def compile_sequence(gates, nq: int, device, dtype, max_window: int = None):
    K = max_window or MAX_WINDOW
    gates = [(gid, tuple(qs)) for gid, qs in gates]
    key = (tuple(gates), nq, str(device), str(dtype), K)
    if key in _seq_plan_cache:
        return _seq_plan_cache[key]
    steps = _fuse_plan(gates, nq, K)
    for s in steps:
        if s['kind'] == 'long':
            s['plan'] = _color_plan([s['gate']], nq, device, dtype, K)
    plan = {
        'steps': steps,
        'nq': nq,
        'eye': {m: torch.eye(1 << m, device=device, dtype=dtype) for m in range(1, max(K, MIN_WINDOW) + 1)},
        'n_windows': sum(s['kind'] == 'window' for s in steps),
        'n_long': sum(s['kind'] == 'long' for s in steps),
    }
    _seq_plan_cache[key] = plan
    return plan


def apply_sequence(re, im, plan, mats):
    get = lambda gid: mats[gid]
    eye = plan['eye']
    for s in plan['steps']:
        if s['kind'] == 'window':
            U_re, U_im = _fused_operator(get, s, eye)
            re, im = _apply_window(re, im, U_re, U_im, s['start'], s['width'])
        else:
            G_re, G_im = mats[s['gate'][0]]
            re, im = _apply_windowed(re, im, s['plan'], G_re[None], G_im[None])
    return re, im

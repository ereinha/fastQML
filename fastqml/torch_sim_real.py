import torch
try:
    import torch._inductor.config as _ic
    _ic.triton.cudagraphs = False
except Exception:
    pass


def _bc(val, ndim):
    if val.dim() == 0:
        return val
    return val.reshape(-1, *([1] * ndim))


def apply_ry(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    c = _bc(torch.cos(theta / 2), nq - 1)
    s = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re = torch.stack([c * re0 - s * re1, s * re0 + c * re1], dim=-1)
    new_im = torch.stack([c * im0 - s * im1, s * im0 + c * im1], dim=-1)
    return new_re.movedim(-1, qubit + 1), new_im.movedim(-1, qubit + 1)


def apply_rz(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    ct = _bc(torch.cos(theta / 2), nq - 1)
    st = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re0 = ct * re0 + st * im0
    new_im0 = ct * im0 - st * re0
    new_re1 = ct * re1 - st * im1
    new_im1 = ct * im1 + st * re1
    return (torch.stack([new_re0, new_re1], dim=-1).movedim(-1, qubit + 1),
            torch.stack([new_im0, new_im1], dim=-1).movedim(-1, qubit + 1))


def apply_rx(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    c = _bc(torch.cos(theta / 2), nq - 1)
    s = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re0 = c * re0 + s * im1
    new_im0 = c * im0 - s * re1
    new_re1 = s * im0 + c * re1
    new_im1 = -s * re0 + c * im1
    return (torch.stack([new_re0, new_re1], dim=-1).movedim(-1, qubit + 1),
            torch.stack([new_im0, new_im1], dim=-1).movedim(-1, qubit + 1))


def apply_izz(re, im, theta, q0, q1, nq):
    z = torch.tensor([1.0, -1.0], dtype=re.dtype, device=re.device)
    s0 = [1] * (nq + 1); s0[q0 + 1] = 2
    s1 = [1] * (nq + 1); s1[q1 + 1] = 2
    zz = z.reshape(s0) * z.reshape(s1)
    angle = _bc(theta / 2, nq) * zz
    ct = torch.cos(angle)
    st = torch.sin(angle)
    new_re = ct * re + st * im
    new_im = ct * im - st * re
    return new_re, new_im


def _two_qubit_perm(control, target, nq):
    target_axes = [control + 1, target + 1]
    others = [a for a in range(1, nq + 1) if a not in target_axes]
    perm = [0] + others + target_axes
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return perm, inv


def apply_cry(re, im, theta, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    G = torch.stack([
        torch.stack([c, -s]),
        torch.stack([s,  c]),
    ])
    Gt = G.T
    re_c0 = re[..., 0:1, :]; im_c0 = im[..., 0:1, :]
    re_c1 = re[..., 1:2, :]; im_c1 = im[..., 1:2, :]
    new_re_c1 = re_c1 @ Gt
    new_im_c1 = im_c1 @ Gt
    out_re = torch.cat([re_c0, new_re_c1], dim=-2)
    out_im = torch.cat([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_x(re, im, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    new_re = torch.stack([re[..., 1], re[..., 0]], dim=-1)
    new_im = torch.stack([im[..., 1], im[..., 0]], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_y(re, im, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re = torch.stack([im1, -im0], dim=-1)
    new_im = torch.stack([-re1, re0], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_z(re, im, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    new_re = torch.stack([re[..., 0], -re[..., 1]], dim=-1)
    new_im = torch.stack([im[..., 0], -im[..., 1]], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_h(re, im, q, nq):
    inv_sqrt2 = 0.7071067811865475
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re = torch.stack([(re0 + re1) * inv_sqrt2, (re0 - re1) * inv_sqrt2], dim=-1)
    new_im = torch.stack([(im0 + im1) * inv_sqrt2, (im0 - im1) * inv_sqrt2], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_s(re, im, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    new_re = torch.stack([re[..., 0], -im[..., 1]], dim=-1)
    new_im = torch.stack([im[..., 0], re[..., 1]], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_t(re, im, q, nq):
    inv_sqrt2 = 0.7071067811865475
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    re1, im1 = re[..., 1], im[..., 1]
    new_re1 = (re1 - im1) * inv_sqrt2
    new_im1 = (re1 + im1) * inv_sqrt2
    new_re = torch.stack([re[..., 0], new_re1], dim=-1)
    new_im = torch.stack([im[..., 0], new_im1], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_phase_shift(re, im, phi, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    c = _bc(torch.cos(phi), nq - 1)
    s = _bc(torch.sin(phi), nq - 1)
    re1, im1 = re[..., 1], im[..., 1]
    new_re1 = c * re1 - s * im1
    new_im1 = c * im1 + s * re1
    new_re = torch.stack([re[..., 0], new_re1], dim=-1)
    new_im = torch.stack([im[..., 0], new_im1], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_rot(re, im, phi, theta, omega, q, nq):
    re, im = apply_rz(re, im, phi, q, nq)
    re, im = apply_ry(re, im, theta, q, nq)
    re, im = apply_rz(re, im, omega, q, nq)
    return re, im


def apply_cnot(re, im, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    new_re_c1 = torch.stack([re_c1[..., 1], re_c1[..., 0]], dim=-1)
    new_im_c1 = torch.stack([im_c1[..., 1], im_c1[..., 0]], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


apply_cx = apply_cnot


def apply_cy(re, im, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_c1 = torch.stack([im_t1, -im_t0], dim=-1)
    new_im_c1 = torch.stack([-re_t1, re_t0], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_cz(re, im, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    new_re_c1 = torch.stack([re_c1[..., 0], -re_c1[..., 1]], dim=-1)
    new_im_c1 = torch.stack([im_c1[..., 0], -im_c1[..., 1]], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_swap(re, im, q0, q1, nq):
    perm, inv = _two_qubit_perm(q0, q1, nq)
    re = re.permute(perm); im = im.permute(perm)
    out_re = re.transpose(-1, -2)
    out_im = im.transpose(-1, -2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_iswap(re, im, q0, q1, nq):
    perm, inv = _two_qubit_perm(q0, q1, nq)
    re = re.permute(perm); im = im.permute(perm)
    re_00 = re[..., 0, 0]; re_01 = re[..., 0, 1]
    re_10 = re[..., 1, 0]; re_11 = re[..., 1, 1]
    im_00 = im[..., 0, 0]; im_01 = im[..., 0, 1]
    im_10 = im[..., 1, 0]; im_11 = im[..., 1, 1]
    new_re_10, new_im_10 = -im_01, re_01
    new_re_01, new_im_01 = -im_10, re_10
    row0_re = torch.stack([re_00, new_re_01], dim=-1)
    row1_re = torch.stack([new_re_10, re_11], dim=-1)
    row0_im = torch.stack([im_00, new_im_01], dim=-1)
    row1_im = torch.stack([new_im_10, im_11], dim=-1)
    out_re = torch.stack([row0_re, row1_re], dim=-2)
    out_im = torch.stack([row0_im, row1_im], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_crx(re, im, theta, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(theta / 2), nq - 2)
    s = _bc(torch.sin(theta / 2), nq - 2)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_t0 = c * re_t0 + s * im_t1
    new_im_t0 = c * im_t0 - s * re_t1
    new_re_t1 = s * im_t0 + c * re_t1
    new_im_t1 = -s * re_t0 + c * im_t1
    new_re_c1 = torch.stack([new_re_t0, new_re_t1], dim=-1)
    new_im_c1 = torch.stack([new_im_t0, new_im_t1], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_crz(re, im, theta, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    ct = _bc(torch.cos(theta / 2), nq - 2)
    st = _bc(torch.sin(theta / 2), nq - 2)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_t0 = ct * re_t0 + st * im_t0
    new_im_t0 = ct * im_t0 - st * re_t0
    new_re_t1 = ct * re_t1 - st * im_t1
    new_im_t1 = ct * im_t1 + st * re_t1
    new_re_c1 = torch.stack([new_re_t0, new_re_t1], dim=-1)
    new_im_c1 = torch.stack([new_im_t0, new_im_t1], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_cphase(re, im, phi, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(phi), nq - 2)
    s = _bc(torch.sin(phi), nq - 2)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_t1 = c * re_t1 - s * im_t1
    new_im_t1 = c * im_t1 + s * re_t1
    new_re_c1 = torch.stack([re_t0, new_re_t1], dim=-1)
    new_im_c1 = torch.stack([im_t0, new_im_t1], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_ixx(re, im, theta, q0, q1, nq):
    perm, inv = _two_qubit_perm(q0, q1, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(theta / 2), nq - 2)
    s = _bc(torch.sin(theta / 2), nq - 2)
    re_00 = re[..., 0, 0]; re_01 = re[..., 0, 1]
    re_10 = re[..., 1, 0]; re_11 = re[..., 1, 1]
    im_00 = im[..., 0, 0]; im_01 = im[..., 0, 1]
    im_10 = im[..., 1, 0]; im_11 = im[..., 1, 1]
    new_re_00 = c * re_00 + s * im_11; new_im_00 = c * im_00 - s * re_11
    new_re_11 = c * re_11 + s * im_00; new_im_11 = c * im_11 - s * re_00
    new_re_01 = c * re_01 + s * im_10; new_im_01 = c * im_01 - s * re_10
    new_re_10 = c * re_10 + s * im_01; new_im_10 = c * im_10 - s * re_01
    row0_re = torch.stack([new_re_00, new_re_01], dim=-1)
    row1_re = torch.stack([new_re_10, new_re_11], dim=-1)
    row0_im = torch.stack([new_im_00, new_im_01], dim=-1)
    row1_im = torch.stack([new_im_10, new_im_11], dim=-1)
    out_re = torch.stack([row0_re, row1_re], dim=-2)
    out_im = torch.stack([row0_im, row1_im], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_iyy(re, im, theta, q0, q1, nq):
    perm, inv = _two_qubit_perm(q0, q1, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(theta / 2), nq - 2)
    s = _bc(torch.sin(theta / 2), nq - 2)
    re_00 = re[..., 0, 0]; re_01 = re[..., 0, 1]
    re_10 = re[..., 1, 0]; re_11 = re[..., 1, 1]
    im_00 = im[..., 0, 0]; im_01 = im[..., 0, 1]
    im_10 = im[..., 1, 0]; im_11 = im[..., 1, 1]
    new_re_00 = c * re_00 - s * im_11; new_im_00 = c * im_00 + s * re_11
    new_re_11 = c * re_11 - s * im_00; new_im_11 = c * im_11 + s * re_00
    new_re_01 = c * re_01 + s * im_10; new_im_01 = c * im_01 - s * re_10
    new_re_10 = c * re_10 + s * im_01; new_im_10 = c * im_10 - s * re_01
    row0_re = torch.stack([new_re_00, new_re_01], dim=-1)
    row1_re = torch.stack([new_re_10, new_re_11], dim=-1)
    row0_im = torch.stack([new_im_00, new_im_01], dim=-1)
    row1_im = torch.stack([new_im_10, new_im_11], dim=-1)
    out_re = torch.stack([row0_re, row1_re], dim=-2)
    out_im = torch.stack([row0_im, row1_im], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def _move_three_to_end(re, im, q1, q2, q3, nq):
    target_axes = [q1 + 1, q2 + 1, q3 + 1]
    others = [a for a in range(1, nq + 1) if a not in target_axes]
    perm = [0] + others + target_axes
    return re.permute(perm), im.permute(perm), perm


def _invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def apply_toffoli(re, im, c1, c2, target, nq):
    re_p, im_p, perm = _move_three_to_end(re, im, c1, c2, target, nq)
    re_00 = re_p[..., 0, 0, :]; im_00 = im_p[..., 0, 0, :]
    re_01 = re_p[..., 0, 1, :]; im_01 = im_p[..., 0, 1, :]
    re_10 = re_p[..., 1, 0, :]; im_10 = im_p[..., 1, 0, :]
    re_11 = re_p[..., 1, 1, :]; im_11 = im_p[..., 1, 1, :]
    re_11_new = re_11.flip(-1)
    im_11_new = im_11.flip(-1)
    row0 = torch.stack([re_00, re_01], dim=-2)
    row1 = torch.stack([re_10, re_11_new], dim=-2)
    new_re = torch.stack([row0, row1], dim=-3)
    row0i = torch.stack([im_00, im_01], dim=-2)
    row1i = torch.stack([im_10, im_11_new], dim=-2)
    new_im = torch.stack([row0i, row1i], dim=-3)
    return new_re.permute(_invert_perm(perm)), new_im.permute(_invert_perm(perm))


apply_ccx = apply_toffoli


def apply_cswap(re, im, control, target1, target2, nq):
    re_p, im_p, perm = _move_three_to_end(re, im, control, target1, target2, nq)
    re_c0 = re_p[..., 0, :, :]; im_c0 = im_p[..., 0, :, :]
    re_c1 = re_p[..., 1, :, :]; im_c1 = im_p[..., 1, :, :]
    new_re_c1 = re_c1.transpose(-1, -2)
    new_im_c1 = im_c1.transpose(-1, -2)
    new_re = torch.stack([re_c0, new_re_c1], dim=-3)
    new_im = torch.stack([im_c0, new_im_c1], dim=-3)
    return new_re.permute(_invert_perm(perm)), new_im.permute(_invert_perm(perm))


def _amp_to_state_real(amp, nq):
    B = amp.shape[0]
    norm = amp.norm(dim=1, keepdim=True).clamp(min=1e-12)
    normed = (amp / norm).reshape(B, *([2] * nq))
    return normed, torch.zeros_like(normed)


def apply_block_encoding_complex(amp, rot_layers, cry_anchor, cry_data, rot_final,
                                 nq, eps=1e-12):
    B = amp.shape[0]
    nq_total = nq + 1

    norm_in = amp.norm(dim=1, keepdim=True).clamp(min=eps)
    amp_n = amp / norm_in

    re_data = amp_n.reshape(B, *([2] * nq))
    zero = torch.zeros_like(re_data)
    re = torch.stack([re_data, zero], dim=-1)
    im = torch.zeros_like(re)

    L = rot_layers.shape[0]
    for layer in range(L):
        for q in range(nq_total):
            phi, theta, omega = rot_layers[layer, q, 0], rot_layers[layer, q, 1], rot_layers[layer, q, 2]
            re, im = apply_rot(re, im, phi, theta, omega, q, nq_total)
        for q in range(nq):
            re, im = apply_cry(re, im, cry_anchor[layer, q], nq, q, nq_total)
        for q in range(nq - 1):
            re, im = apply_cry(re, im, cry_data[layer, q], q, q + 1, nq_total)

    for q in range(nq_total):
        phi, theta, omega = rot_final[q, 0], rot_final[q, 1], rot_final[q, 2]
        re, im = apply_rot(re, im, phi, theta, omega, q, nq_total)

    re_post = re[..., 0]
    im_post = im[..., 0]

    norm_sq = (re_post * re_post + im_post * im_post).reshape(B, -1).sum(dim=1)
    norm = torch.sqrt(norm_sq.clamp(min=eps * eps)).reshape(-1, *([1] * nq))
    return re_post / norm, im_post / norm


def hadamard_test_overlap(re_Q, im_Q, re_K, im_K, nq):
    inv_sqrt2 = 0.7071067811865475
    re_joint = torch.stack([re_Q, re_K], dim=1) * inv_sqrt2
    im_joint = torch.stack([im_Q, im_K], dim=1) * inv_sqrt2

    nq_total = nq + 1
    re_joint, im_joint = apply_h(re_joint, im_joint, q=0, nq=nq_total)

    re_h0 = re_joint[:, 0]
    im_h0 = im_joint[:, 0]
    p0 = (re_h0 * re_h0 + im_h0 * im_h0).reshape(re_h0.shape[0], -1).sum(dim=1)
    return 2.0 * p0 - 1.0


def apply_block_encoding_real(amp, ry_layers, cry_anchor, cry_data, ry_final, nq,
                              eps=1e-12):
    B = amp.shape[0]
    nq_total = nq + 1

    norm_in = amp.norm(dim=1, keepdim=True).clamp(min=eps)
    amp_n = amp / norm_in

    re_data = amp_n.reshape(B, *([2] * nq))
    zero = torch.zeros_like(re_data)
    re = torch.stack([re_data, zero], dim=-1)
    im = torch.zeros_like(re)

    L = ry_layers.shape[0]
    for layer in range(L):
        for q in range(nq_total):
            re, im = apply_ry(re, im, ry_layers[layer, q], q, nq_total)
        for q in range(nq):
            re, im = apply_cry(re, im, cry_anchor[layer, q], nq, q, nq_total)
        for q in range(nq - 1):
            re, im = apply_cry(re, im, cry_data[layer, q], q, q + 1, nq_total)

    for q in range(nq_total):
        re, im = apply_ry(re, im, ry_final[q], q, nq_total)

    re_post = re[..., 0]
    flat = re_post.reshape(B, -1)
    norm_out = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    return flat / norm_out


def _state_to_probs_real(re, im):
    B = re.shape[0]
    return (re * re + im * im).reshape(B, -1)


def _renormalize_real(re, im, nq):
    B = re.shape[0]
    norm_sq = (re * re + im * im).reshape(B, -1).sum(dim=1)
    norm = torch.sqrt(norm_sq.clamp(min=1e-24)).reshape(-1, *([1] * nq))
    return re / norm, im / norm

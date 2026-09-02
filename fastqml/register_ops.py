from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False


def _stride_of(target, nq):
    return 1 << (nq - 1 - target)


def _check(re, angles, target, select_bits, nq):
    D = re.shape[-1]
    if D != (1 << nq):
        raise ValueError(f"flat state dim {D} != 2^{nq}")
    stride = _stride_of(target, nq)
    S = 1 << select_bits
    if S > stride:
        raise ValueError("select bits must lie strictly below the target qubit")
    if angles.shape != (re.shape[0], S):
        raise ValueError(f"angles must have shape (B, {S})")
    return stride, S


if HAVE_TRITON:
    @triton.jit
    def _mux_fwd_kernel(vr_ptr, vi_ptr, cy_ptr, sy_ptr, cz_ptr, sz_ptr,
                        npairs, dim, half, stride, nsel,
                        HAS_RZ: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BLOCK + tl.arange(0, BLOCK)
        mask = p < npairs
        config = p // half
        rem = p - config * half
        group = rem // stride
        s = rem - group * stride
        idx0 = config * dim + group * (2 * stride) + s
        idx1 = idx0 + stride
        aidx = config * nsel + s % nsel
        cy = tl.load(cy_ptr + aidx, mask=mask)
        sy = tl.load(sy_ptr + aidx, mask=mask)
        v0r = tl.load(vr_ptr + idx0, mask=mask); v1r = tl.load(vr_ptr + idx1, mask=mask)
        v0i = tl.load(vi_ptr + idx0, mask=mask); v1i = tl.load(vi_ptr + idx1, mask=mask)
        u0r = cy * v0r - sy * v1r; u1r = sy * v0r + cy * v1r
        u0i = cy * v0i - sy * v1i; u1i = sy * v0i + cy * v1i
        if HAS_RZ:
            cz = tl.load(cz_ptr + aidx, mask=mask)
            sz = tl.load(sz_ptr + aidx, mask=mask)
            n0r = cz * u0r + sz * u0i; n0i = cz * u0i - sz * u0r
            n1r = cz * u1r - sz * u1i; n1i = cz * u1i + sz * u1r
        else:
            n0r = u0r; n0i = u0i; n1r = u1r; n1i = u1i
        tl.store(vr_ptr + idx0, n0r, mask=mask); tl.store(vr_ptr + idx1, n1r, mask=mask)
        tl.store(vi_ptr + idx0, n0i, mask=mask); tl.store(vi_ptr + idx1, n1i, mask=mask)

    @triton.jit
    def _mux_bwd_kernel(vr_ptr, vi_ptr, gr_ptr, gi_ptr, cy_ptr, sy_ptr, cz_ptr, sz_ptr,
                        gty_ptr, gtz_ptr, npairs, dim, half, stride, nsel,
                        HAS_RZ: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BLOCK + tl.arange(0, BLOCK)
        mask = p < npairs
        config = p // half
        rem = p - config * half
        group = rem // stride
        s = rem - group * stride
        idx0 = config * dim + group * (2 * stride) + s
        idx1 = idx0 + stride
        aidx = config * nsel + s % nsel
        cy = tl.load(cy_ptr + aidx, mask=mask)
        sy = tl.load(sy_ptr + aidx, mask=mask)
        v0r = tl.load(vr_ptr + idx0, mask=mask); v1r = tl.load(vr_ptr + idx1, mask=mask)
        v0i = tl.load(vi_ptr + idx0, mask=mask); v1i = tl.load(vi_ptr + idx1, mask=mask)
        g0r = tl.load(gr_ptr + idx0, mask=mask); g1r = tl.load(gr_ptr + idx1, mask=mask)
        g0i = tl.load(gi_ptr + idx0, mask=mask); g1i = tl.load(gi_ptr + idx1, mask=mask)
        u0r = cy * v0r - sy * v1r; u1r = sy * v0r + cy * v1r
        u0i = cy * v0i - sy * v1i; u1i = sy * v0i + cy * v1i
        if HAS_RZ:
            cz = tl.load(cz_ptr + aidx, mask=mask)
            sz = tl.load(sz_ptr + aidx, mask=mask)
            a0r = cz * g0r - sz * g0i; a0i = sz * g0r + cz * g0i
            a1r = cz * g1r + sz * g1i; a1i = -sz * g1r + cz * g1i
            gtz = 0.5 * (g0r * (-sz * u0r + cz * u0i) + g0i * (-sz * u0i - cz * u0r)
                         + g1r * (-sz * u1r - cz * u1i) + g1i * (-sz * u1i + cz * u1r))
            tl.atomic_add(gtz_ptr + aidx, gtz, mask=mask)
        else:
            a0r = g0r; a0i = g0i; a1r = g1r; a1i = g1i
        gty = 0.5 * (a0r * (-sy * v0r - cy * v1r) + a0i * (-sy * v0i - cy * v1i)
                     + a1r * (cy * v0r - sy * v1r) + a1i * (cy * v0i - sy * v1i))
        tl.atomic_add(gty_ptr + aidx, gty, mask=mask)
        gv0r = cy * a0r + sy * a1r; gv1r = -sy * a0r + cy * a1r
        gv0i = cy * a0i + sy * a1i; gv1i = -sy * a0i + cy * a1i
        tl.store(gr_ptr + idx0, gv0r, mask=mask); tl.store(gr_ptr + idx1, gv1r, mask=mask)
        tl.store(gi_ptr + idx0, gv0i, mask=mask); tl.store(gi_ptr + idx1, gv1i, mask=mask)

    def _mux_launch_fwd(vr, vi, cy, sy, cz, sz, stride, nsel):
        B, dim = vr.shape
        half = dim // 2
        npairs = B * half
        grid = lambda meta: (triton.cdiv(npairs, meta['BLOCK']),)
        has_rz = cz is not None
        _mux_fwd_kernel[grid](vr, vi, cy, sy, cz if has_rz else cy, sz if has_rz else sy,
                              npairs, dim, half, stride, nsel, 1 if has_rz else 0, BLOCK=1024)

    def _mux_launch_bwd(vr, vi, gr, gi, cy, sy, cz, sz, gty, gtz, stride, nsel):
        B, dim = vr.shape
        half = dim // 2
        npairs = B * half
        grid = lambda meta: (triton.cdiv(npairs, meta['BLOCK']),)
        has_rz = cz is not None
        _mux_bwd_kernel[grid](vr, vi, gr, gi, cy, sy, cz if has_rz else cy, sz if has_rz else sy,
                              gty, gtz if has_rz else gty, npairs, dim, half, stride, nsel,
                              1 if has_rz else 0, BLOCK=1024)


def _mux_eager(vr, vi, ty, tz, stride, S):
    B, D = vr.shape
    G = D // (2 * stride)
    R = stride // S
    cy = torch.cos(ty / 2).view(B, 1, 1, S)
    sy = torch.sin(ty / 2).view(B, 1, 1, S)
    xr = vr.view(B, G, 2, R, S)
    xi = vi.view(B, G, 2, R, S)
    v0r, v1r = xr[:, :, 0], xr[:, :, 1]
    v0i, v1i = xi[:, :, 0], xi[:, :, 1]
    u0r = cy * v0r - sy * v1r; u1r = sy * v0r + cy * v1r
    u0i = cy * v0i - sy * v1i; u1i = sy * v0i + cy * v1i
    if tz is not None:
        cz = torch.cos(tz / 2).view(B, 1, 1, S)
        sz = torch.sin(tz / 2).view(B, 1, 1, S)
        n0r = cz * u0r + sz * u0i; n0i = cz * u0i - sz * u0r
        n1r = cz * u1r - sz * u1i; n1i = cz * u1i + sz * u1r
    else:
        n0r, n0i, n1r, n1i = u0r, u0i, u1r, u1i
    out_r = torch.stack([n0r, n1r], dim=2).reshape(B, D)
    out_i = torch.stack([n0i, n1i], dim=2).reshape(B, D)
    return out_r, out_i


class _MuxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vr, vi, ty, tz, stride, S):
        cy = torch.cos(ty / 2).contiguous()
        sy = torch.sin(ty / 2).contiguous()
        cz = torch.cos(tz / 2).contiguous() if tz is not None else None
        sz = torch.sin(tz / 2).contiguous() if tz is not None else None
        out_r = vr.contiguous().clone()
        out_i = vi.contiguous().clone()
        _mux_launch_fwd(out_r, out_i, cy, sy, cz, sz, stride, S)
        ctx.save_for_backward(vr, vi, cy, sy,
                              cz if cz is not None else vr.new_zeros(0),
                              sz if sz is not None else vr.new_zeros(0))
        ctx.stride = stride
        ctx.S = S
        ctx.has_rz = tz is not None
        return out_r, out_i

    @staticmethod
    def backward(ctx, g_r, g_i):
        vr, vi, cy, sy, cz, sz = ctx.saved_tensors
        has_rz = ctx.has_rz
        gr = g_r.contiguous().clone()
        gi = g_i.contiguous().clone()
        B = vr.shape[0]
        gty = torch.zeros(B, ctx.S, device=vr.device, dtype=vr.dtype)
        gtz = torch.zeros(B, ctx.S, device=vr.device, dtype=vr.dtype) if has_rz else None
        _mux_launch_bwd(vr.contiguous(), vi.contiguous(), gr, gi, cy, sy,
                        cz if has_rz else None, sz if has_rz else None,
                        gty, gtz, ctx.stride, ctx.S)
        return gr, gi, gty, gtz, None, None


def apply_mux_ry(re, im, angles, target, nq, select_bits):
    stride, S = _check(re, angles, target, select_bits, nq)
    if HAVE_TRITON and re.is_cuda:
        return _MuxFn.apply(re, im, angles, None, stride, S)
    return _mux_eager(re, im, angles, None, stride, S)


def apply_mux_ry_rz(re, im, ry_angles, rz_angles, target, nq, select_bits):
    stride, S = _check(re, ry_angles, target, select_bits, nq)
    _check(re, rz_angles, target, select_bits, nq)
    if HAVE_TRITON and re.is_cuda:
        return _MuxFn.apply(re, im, ry_angles, rz_angles, stride, S)
    return _mux_eager(re, im, ry_angles, rz_angles, stride, S)


def apply_mux_ry_(re, im, angles, target, nq, select_bits):
    stride, S = _check(re, angles, target, select_bits, nq)
    if HAVE_TRITON and re.is_cuda and not torch.is_grad_enabled():
        cy = torch.cos(angles / 2).contiguous()
        sy = torch.sin(angles / 2).contiguous()
        _mux_launch_fwd(re, im, cy, sy, None, None, stride, S)
        return re, im
    out_r, out_i = _mux_eager(re, im, angles, None, stride, S)
    re.copy_(out_r); im.copy_(out_i)
    return re, im


def apply_mux_ry_rz_(re, im, ry_angles, rz_angles, target, nq, select_bits):
    stride, S = _check(re, ry_angles, target, select_bits, nq)
    if HAVE_TRITON and re.is_cuda and not torch.is_grad_enabled():
        cy = torch.cos(ry_angles / 2).contiguous()
        sy = torch.sin(ry_angles / 2).contiguous()
        cz = torch.cos(rz_angles / 2).contiguous()
        sz = torch.sin(rz_angles / 2).contiguous()
        _mux_launch_fwd(re, im, cy, sy, cz, sz, stride, S)
        return re, im
    out_r, out_i = _mux_eager(re, im, ry_angles, rz_angles, stride, S)
    re.copy_(out_r); im.copy_(out_i)
    return re, im


def compose_dense(mats: list) -> Tuple[torch.Tensor, torch.Tensor]:
    U_re, U_im = mats[0]
    for M_re, M_im in mats[1:]:
        new_re = M_re @ U_re - M_im @ U_im
        new_im = M_re @ U_im + M_im @ U_re
        U_re, U_im = new_re, new_im
    return U_re, U_im


def apply_dense(re, im, U_re, U_im):
    Ut_re, Ut_im = U_re.T, U_im.T
    new_re = re @ Ut_re - im @ Ut_im
    new_im = re @ Ut_im + im @ Ut_re
    return new_re, new_im

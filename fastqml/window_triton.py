import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

MIN_WIDTH = 4
MAX_WIDTH = 7
FWD_CFG = {16: (64, 4, 16), 32: (64, 8, 32), 64: (64, 4, 16), 128: (32, 8, 16)}

if HAVE_TRITON:
    @triton.jit
    def _win_fwd_kernel(xr_ptr, xi_ptr, ur_ptr, ui_ptr, or_ptr, oi_ptr, R, lo, dim_lo,
                        DIM: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr,
                        PREC: tl.constexpr):
        pid = tl.program_id(0)
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < R
        base = (r // lo) * dim_lo + (r % lo)
        n = tl.arange(0, DIM)
        acc_r = tl.zeros((BLOCK_R, DIM), dtype=tl.float32)
        acc_i = tl.zeros((BLOCK_R, DIM), dtype=tl.float32)
        for k0 in range(0, DIM, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            xoff = base[:, None] + k[None, :] * lo
            xr = tl.load(xr_ptr + xoff, mask=rmask[:, None], other=0.0)
            xi = tl.load(xi_ptr + xoff, mask=rmask[:, None], other=0.0)
            uoff = n[None, :] * DIM + k[:, None]
            utr = tl.load(ur_ptr + uoff)
            uti = tl.load(ui_ptr + uoff)
            acc_r = tl.dot(xr, utr, acc_r, input_precision=PREC)
            acc_r = tl.dot(xi, -uti, acc_r, input_precision=PREC)
            acc_i = tl.dot(xr, uti, acc_i, input_precision=PREC)
            acc_i = tl.dot(xi, utr, acc_i, input_precision=PREC)
        ooff = base[:, None] + n[None, :] * lo
        tl.store(or_ptr + ooff, acc_r, mask=rmask[:, None])
        tl.store(oi_ptr + ooff, acc_i, mask=rmask[:, None])


def _geometry(re, start, width):
    B = re.shape[0]
    nq = re.dim() - 1
    hi, dim, lo = 1 << start, 1 << width, 1 << (nq - start - width)
    return B * hi * lo, lo, dim


def _prec():
    return "tf32" if torch.backends.cuda.matmul.allow_tf32 else "tf32x3"


def _fwd(re, im, U_re, U_im, start, width):
    R, lo, dim = _geometry(re, start, width)
    out_re, out_im = torch.empty_like(re), torch.empty_like(im)
    block_r, warps, block_k = FWD_CFG[dim]
    grid = (triton.cdiv(R, block_r),)
    _win_fwd_kernel[grid](re, im, U_re, U_im, out_re, out_im, R, lo, dim * lo,
                          DIM=dim, BLOCK_R=block_r, BLOCK_K=block_k, PREC=_prec(),
                          num_warps=warps)
    return out_re, out_im


GRAD_U_CHUNK = 1 << 25


def _accumulate(dU, xr, xi, gr, gi):
    if dU is None:
        dU = (gr.T @ xr, gi.T @ xr)
    else:
        dU[0].addmm_(gr.T, xr)
        dU[1].addmm_(gi.T, xr)
    dU[0].addmm_(gi.T, xi)
    dU[1].addmm_(gr.T, xi, alpha=-1.0)
    return dU


def _grad_u(re, im, g_re, g_im, start, width):
    R, lo, dim = _geometry(re, start, width)
    if lo == 1:
        return _accumulate(None, *(t.reshape(R, dim) for t in (re, im, g_re, g_im)))
    slabs = R // lo
    Xr, Xi, Gr, Gi = (t.reshape(slabs, dim, lo) for t in (re, im, g_re, g_im))
    if lo >= 32:
        step = max(1, GRAD_U_CHUNK // (dim * dim))
        dU_re = dU_im = None
        for s0 in range(0, slabs, step):
            sl = slice(s0, s0 + step)
            xr_t, xi_t = Xr[sl].transpose(1, 2), Xi[sl].transpose(1, 2)
            p_re = torch.bmm(Gr[sl], xr_t).sum(0) + torch.bmm(Gi[sl], xi_t).sum(0)
            p_im = torch.bmm(Gi[sl], xr_t).sum(0) - torch.bmm(Gr[sl], xi_t).sum(0)
            dU_re = p_re if dU_re is None else dU_re.add_(p_re)
            dU_im = p_im if dU_im is None else dU_im.add_(p_im)
        return dU_re, dU_im
    step = max(1, GRAD_U_CHUNK // (dim * lo))
    dU = None
    for s0 in range(0, slabs, step):
        sl = slice(s0, s0 + step)
        dU = _accumulate(dU, *(t[sl].transpose(1, 2).reshape(-1, dim) for t in (Xr, Xi, Gr, Gi)))
    return dU


class _WindowFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, re, im, U_re, U_im, start, width):
        re, im = re.contiguous(), im.contiguous()
        U_re, U_im = U_re.contiguous(), U_im.contiguous()
        ctx.save_for_backward(re, im, U_re, U_im)
        ctx.geom = (start, width)
        return _fwd(re, im, U_re, U_im, start, width)

    @staticmethod
    def backward(ctx, g_re, g_im):
        re, im, U_re, U_im = ctx.saved_tensors
        start, width = ctx.geom
        g_re, g_im = g_re.contiguous(), g_im.contiguous()
        gx_re = gx_im = gU_re = gU_im = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            gx_re, gx_im = _fwd(g_re, g_im, U_re.T.contiguous(), (-U_im.T).contiguous(), start, width)
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            gU_re, gU_im = _grad_u(re, im, g_re, g_im, start, width)
        return gx_re, gx_im, gU_re, gU_im, None, None


def usable(re, width):
    return HAVE_TRITON and re.is_cuda and re.dtype == torch.float32 and MIN_WIDTH <= width <= MAX_WIDTH


def apply_window(re, im, U_re, U_im, start, width):
    return _WindowFn.apply(re, im, U_re, U_im, start, width)

import torch

from fastqml import gate_fusion as gf
from fastqml import window_triton as wt

if not (torch.cuda.is_available() and wt.HAVE_TRITON):
    print("SKIP: window_triton needs CUDA + Triton")
    raise SystemExit(0)

dev = torch.device("cuda")
torch.manual_seed(0)
n_ok = n_total = 0


def run(nq, B, start, width, tf32):
    global n_ok, n_total
    torch.backends.cuda.matmul.allow_tf32 = tf32
    re = torch.randn(B, *([2] * nq), device=dev, requires_grad=True)
    im = torch.randn(B, *([2] * nq), device=dev, requires_grad=True)
    dim = 1 << width
    Q, _ = torch.linalg.qr(torch.randn(dim, dim, device=dev) + 1j * torch.randn(dim, dim, device=dev))
    U_re = Q.real.contiguous().requires_grad_(True)
    U_im = Q.imag.contiguous().requires_grad_(True)
    g_re, g_im = torch.randn_like(re), torch.randn_like(im)

    def both(use_triton):
        gf.USE_TRITON_WINDOW = use_triton
        for t in (re, im, U_re, U_im):
            t.grad = None
        o_re, o_im = gf._apply_window(re, im, U_re, U_im, start, width)
        (o_re * g_re + o_im * g_im).sum().backward()
        return o_re.detach(), o_im.detach(), re.grad.clone(), im.grad.clone(), U_re.grad.clone(), U_im.grad.clone()
    ref, out = both(False), both(True)
    err = max(((x - y).norm() / y.norm()).item() for x, y in zip(out, ref))
    tol = 2e-3 if tf32 else 2e-5
    n_total += 1
    n_ok += err < tol
    print(f"  {'OK' if err < tol else 'FAIL'}  nq={nq:2d} B={B:4d} window=({start},{width}) "
          f"tf32={str(tf32):5s}  max rel err (out, grad_x, grad_U) = {err:.1e}")


print("=== window_triton vs matmul path ===")
for nq, B in ((8, 3), (10, 5), (12, 2), (14, 1)):
    for start, width in ((0, 4), (0, 5), (2, 6), (nq - 7, 7), (nq - 5, 5), (3, 5)):
        if 0 <= start and start + width <= nq:
            run(nq, B, start, width, tf32=False)
run(12, 64, 0, 6, tf32=True)
run(12, 64, 6, 6, tf32=True)
run(10, 4096, 3, 4, tf32=True)
gf.USE_TRITON_WINDOW = True
print(f"\n{n_ok}/{n_total} passed")

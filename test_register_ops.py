import torch

from fastqml import register_ops as ro
from fastqml import circulant as cc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = []


def report(name, diff, tol):
    ok = diff < tol
    RESULTS.append(ok)
    print(f"  {'OK ' if ok else 'FAIL'} {name:52s} max |diff| = {diff:.2e}")


def dense_mux_reference(ty, tz, target, nq, select_bits, dtype):
    D = 1 << nq
    B = ty.shape[0]
    stride = 1 << (nq - 1 - target)
    S = 1 << select_bits
    U = torch.zeros(B, D, D, dtype=torch.complex128, device=ty.device)
    for b in range(B):
        for idx0 in range(D):
            if idx0 & stride:
                continue
            idx1 = idx0 + stride
            sel = idx0 % S
            a_y = float(ty[b, sel])
            cy, sy = torch.cos(torch.tensor(a_y / 2)), torch.sin(torch.tensor(a_y / 2))
            g = torch.tensor([[cy, -sy], [sy, cy]], dtype=torch.complex128)
            if tz is not None:
                a_z = float(tz[b, sel])
                ph = torch.exp(torch.tensor(1j * a_z / 2, dtype=torch.complex128))
                g = torch.diag(torch.stack([ph.conj(), ph])) @ g
            U[b, idx0, idx0] = g[0, 0]; U[b, idx0, idx1] = g[0, 1]
            U[b, idx1, idx0] = g[1, 0]; U[b, idx1, idx1] = g[1, 1]
    return U


def test_mux(nq=5, B=3, target=1, select_bits=2, with_rz=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = 1 << nq
    S = 1 << select_bits
    re = torch.randn(B, D, generator=g).to(DEVICE)
    im = torch.randn(B, D, generator=g).to(DEVICE)
    ty = torch.randn(B, S, generator=g).to(DEVICE)
    tz = torch.randn(B, S, generator=g).to(DEVICE) if with_rz else None
    U = dense_mux_reference(ty.cpu(), tz.cpu() if with_rz else None, target, nq, select_bits, torch.complex128)
    v = torch.complex(re.cpu().double(), im.cpu().double())
    ref = torch.einsum('bij,bj->bi', U, v)
    if with_rz:
        outr, outi = ro.apply_mux_ry_rz(re, im, ty, tz, target, nq, select_bits)
        ipr, ipi = ro.apply_mux_ry_rz_(re.clone(), im.clone(), ty, tz, target, nq, select_bits)
    else:
        outr, outi = ro.apply_mux_ry(re, im, ty, target, nq, select_bits)
        ipr, ipi = ro.apply_mux_ry_(re.clone(), im.clone(), ty, target, nq, select_bits)
    d1 = max((outr.cpu().double() - ref.real).abs().max().item(),
             (outi.cpu().double() - ref.imag).abs().max().item())
    d2 = max((ipr - outr).abs().max().item(), (ipi - outi).abs().max().item())
    tag = "ry_rz" if with_rz else "ry"
    report(f"mux_{tag} vs dense reference (nq={nq}, t={target}, S={S})", d1, 1e-5)
    report(f"mux_{tag} in-place == out-of-place", d2, 1e-6)


def test_mux_eager_vs_triton(nq=8, B=64, target=1, select_bits=6, seed=1):
    if not (ro.HAVE_TRITON and DEVICE.type == "cuda"):
        print("  SKIP triton-vs-eager (triton unavailable)")
        return
    g = torch.Generator().manual_seed(seed)
    D = 1 << nq
    S = 1 << select_bits
    re = torch.randn(B, D, generator=g).to(DEVICE)
    im = torch.randn(B, D, generator=g).to(DEVICE)
    ty = torch.randn(B, S, generator=g).to(DEVICE)
    tz = torch.randn(B, S, generator=g).to(DEVICE)
    stride = 1 << (nq - 1 - target)
    er, ei = ro._mux_eager(re, im, ty, tz, stride, S)
    tr, ti = ro.apply_mux_ry_rz(re, im, ty, tz, target, nq, select_bits)
    d = max((er - tr).abs().max().item(), (ei - ti).abs().max().item())
    report(f"triton fwd == eager fwd (nq={nq}, S={S})", d, 1e-5)


def test_mux_autograd(nq=6, B=8, target=1, select_bits=3, seed=2):
    g = torch.Generator().manual_seed(seed)
    D = 1 << nq
    S = 1 << select_bits
    stride = 1 << (nq - 1 - target)
    re0 = torch.randn(B, D, generator=g).to(DEVICE)
    im0 = torch.randn(B, D, generator=g).to(DEVICE)
    ty0 = torch.randn(B, S, generator=g).to(DEVICE)
    tz0 = torch.randn(B, S, generator=g).to(DEVICE)
    w = torch.randn(B, D, generator=g).to(DEVICE)

    def loss_of(fn):
        re = re0.clone().requires_grad_(True)
        im = im0.clone().requires_grad_(True)
        ty = ty0.clone().requires_grad_(True)
        tz = tz0.clone().requires_grad_(True)
        outr, outi = fn(re, im, ty, tz)
        loss = (w * outr).sum() + (w * outi * 0.5).sum()
        loss.backward()
        return [t.grad.clone() for t in (re, im, ty, tz)]

    ge = loss_of(lambda re, im, ty, tz: ro._mux_eager(re, im, ty, tz, stride, S))
    gt = loss_of(lambda re, im, ty, tz: ro.apply_mux_ry_rz(re, im, ty, tz, target, nq, select_bits))
    names = ["d_re", "d_im", "d_ty", "d_tz"]
    for n, a, b in zip(names, ge, gt):
        report(f"mux_ry_rz autograd {n} (triton bwd vs eager autograd)",
               (a - b).abs().max().item(), 1e-4)


def test_compose_dense(nq=5, B=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    D = 1 << nq
    mats = []
    for _ in range(3):
        q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.complex128, generator=g))
        mats.append((q.real.float().to(DEVICE), q.imag.float().to(DEVICE)))
    re = torch.randn(B, D, generator=g).to(DEVICE)
    im = torch.randn(B, D, generator=g).to(DEVICE)
    U_re, U_im = ro.compose_dense(mats)
    r1, i1 = ro.apply_dense(re, im, U_re, U_im)
    r2, i2 = re, im
    for M_re, M_im in mats:
        r2, i2 = ro.apply_dense(r2, i2, M_re, M_im)
    d = max((r1 - r2).abs().max().item(), (i1 - i2).abs().max().item())
    report("compose_dense == sequential apply_dense", d, 1e-3)


def random_unitary_blocks(L1, L2, A, seed):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(L1, L2, A, A, dtype=torch.complex128, generator=g)
    q, _ = torch.linalg.qr(m)
    return q


def test_circulant(L1=4, L2=4, A=4, B=16, seed=4):
    N = L1 * L2
    plan = cc.circulant_plan(L1, L2, DEVICE)
    Uk = random_unitary_blocks(L1, L2, A, seed).to(DEVICE)
    Uk_re, Uk_im = Uk.real.float(), Uk.imag.float()
    M_re, M_im = cc.circulant_dense(Uk_re, Uk_im, plan)
    M = torch.complex(M_re.double(), M_im.double())
    uerr = (M.conj().T @ M - torch.eye(A * N, dtype=torch.complex128, device=DEVICE)).abs().max().item()
    report(f"circulant_dense unitary for unitary U(k) (L={L1}x{L2}, A={A})", uerr, 1e-4)
    g = torch.Generator().manual_seed(seed + 1)
    re = torch.randn(B, A * N, generator=g).to(DEVICE)
    im = torch.randn(B, A * N, generator=g).to(DEVICE)
    r1, i1 = ro.apply_dense(re, im, M_re, M_im)
    r2, i2 = cc.circulant_apply_k(re, im, Uk_re, Uk_im, plan)
    d = max((r1 - r2).abs().max().item(), (i1 - i2).abs().max().item())
    report("circulant dense apply == k-space apply", d, 1e-3)
    sh_r = re.view(B, A, L1, L2).roll(shifts=(1, 2), dims=(2, 3)).reshape(B, A * N)
    sh_i = im.view(B, A, L1, L2).roll(shifts=(1, 2), dims=(2, 3)).reshape(B, A * N)
    r3, i3 = ro.apply_dense(sh_r, sh_i, M_re, M_im)
    r4 = r1.view(B, A, L1, L2).roll(shifts=(1, 2), dims=(2, 3)).reshape(B, A * N)
    i4 = i1.view(B, A, L1, L2).roll(shifts=(1, 2), dims=(2, 3)).reshape(B, A * N)
    d = max((r3 - r4).abs().max().item(), (i3 - i4).abs().max().item())
    report("circulant commutes with lattice translations", d, 1e-3)


if __name__ == "__main__":
    print("=== register_ops: multiplexed rotations ===")
    test_mux(with_rz=True)
    test_mux(with_rz=False)
    test_mux(nq=8, B=4, target=0, select_bits=6, with_rz=True, seed=5)
    test_mux_eager_vs_triton()
    test_mux_autograd()
    print("=== register_ops: dense composition ===")
    test_compose_dense()
    print("=== circulant ===")
    test_circulant()
    n_ok = sum(RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} passed")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)

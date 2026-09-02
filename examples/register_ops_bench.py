import torch

from fastqml import register_ops as ro
from fastqml import circulant as cc
from fastqml import gate_fusion as gf

DEVICE = torch.device("cuda")


def time_fn(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def bench_mux(B=8192, nq=8, target=1, select_bits=6):
    D = 1 << nq
    S = 1 << select_bits
    stride = 1 << (nq - 1 - target)
    g = torch.Generator().manual_seed(0)
    re = torch.randn(B, D, generator=g).to(DEVICE)
    im = torch.randn(B, D, generator=g).to(DEVICE)
    ty = torch.randn(B, S, generator=g).to(DEVICE)
    tz = torch.randn(B, S, generator=g).to(DEVICE)

    def via_triton():
        with torch.no_grad():
            ro.apply_mux_ry_rz_(re, im, ty, tz, target, nq, select_bits)

    def via_eager():
        ro._mux_eager(re, im, ty, tz, stride, S)

    def via_per_batch_chain():
        G = D // (2 * stride)
        R = stride // S
        BP = B * S
        r = re.view(B, G, 2, R, S).permute(0, 4, 2, 1, 3).reshape(BP, 2, G, R)
        i = im.view(B, G, 2, R, S).permute(0, 4, 2, 1, 3).reshape(BP, 2, G, R)
        cy = torch.cos(ty / 2).reshape(BP)
        sy = torch.sin(ty / 2).reshape(BP)
        cz = torch.cos(tz / 2).reshape(BP)
        sz = torch.sin(tz / 2).reshape(BP)
        Gr = torch.stack([torch.stack([cz * cy, -cz * sy], -1),
                          torch.stack([cz * sy, cz * cy], -1)], -2).unsqueeze(1)
        Gi = torch.stack([torch.stack([-sz * cy, sz * sy], -1),
                          torch.stack([sz * sy, sz * cy], -1)], -2).unsqueeze(1)
        r, i = gf.apply_per_batch_su2_chain(r, i, Gr, Gi)
        r.reshape(B, S, 2, G, R).permute(0, 3, 2, 4, 1).reshape(B, D)
        i.reshape(B, S, 2, G, R).permute(0, 3, 2, 4, 1).reshape(B, D)

    comp_eager = torch.compile(ro._mux_eager, dynamic=False)

    def via_compiled():
        comp_eager(re, im, ty, tz, stride, S)

    print(f"=== multiplexed RyRz: B={B}, nq={nq}, target={target}, {S} select values ===")
    if ro.HAVE_TRITON:
        print(f"  register_ops triton (in-place):     {time_fn(via_triton):8.3f} ms")
    print(f"  register_ops eager fallback:        {time_fn(via_eager):8.3f} ms")
    print(f"  fallback + torch.compile:           {time_fn(via_compiled):8.3f} ms")
    print(f"  per-batch su2 chain (batch fold):   {time_fn(via_per_batch_chain):8.3f} ms")


def bench_circulant(L, A, B):
    N = L * L
    plan = cc.circulant_plan(L, L, DEVICE)
    g = torch.Generator().manual_seed(1)
    m = torch.randn(L, L, A, A, dtype=torch.complex128, generator=g)
    q, _ = torch.linalg.qr(m)
    Uk_re, Uk_im = q.real.float().to(DEVICE), q.imag.float().to(DEVICE)
    M_re, M_im = cc.circulant_dense(Uk_re, Uk_im, plan)
    re = torch.randn(B, A * N, generator=g).to(DEVICE)
    im = torch.randn(B, A * N, generator=g).to(DEVICE)
    t_dense = time_fn(lambda: ro.apply_dense(re, im, M_re, M_im))
    t_k = time_fn(lambda: cc.circulant_apply_k(re, im, Uk_re, Uk_im, plan))
    comp_k = torch.compile(cc.circulant_apply_k, dynamic=False)
    t_kc = time_fn(lambda: comp_k(re, im, Uk_re, Uk_im, plan))
    print(f"=== circulant block: {L}x{L} lattice, A={A} (dim={A * N}), B={B} ===")
    print(f"  dense apply (4 GEMMs):              {t_dense:8.3f} ms")
    print(f"  k-space apply (DFT + blockmul):     {t_k:8.3f} ms")
    print(f"  k-space + torch.compile:            {t_kc:8.3f} ms")


def bench_compose(nq=8, B=8192, n_layers=8):
    D = 1 << nq
    g = torch.Generator().manual_seed(2)
    mats = []
    for _ in range(n_layers):
        m = torch.randn(D, D, dtype=torch.complex128, generator=g)
        q, _ = torch.linalg.qr(m)
        mats.append((q.real.float().to(DEVICE), q.imag.float().to(DEVICE)))
    U_re, U_im = ro.compose_dense(mats)
    re = torch.randn(B, D, generator=g).to(DEVICE)
    im = torch.randn(B, D, generator=g).to(DEVICE)

    def sequential():
        r, i = re, im
        for M_re, M_im in mats:
            r, i = ro.apply_dense(r, i, M_re, M_im)

    t_seq = time_fn(sequential)
    t_one = time_fn(lambda: ro.apply_dense(re, im, U_re, U_im))
    print(f"=== compose_dense: {n_layers} fixed layers, nq={nq}, B={B} ===")
    print(f"  sequential layer applies:           {t_seq:8.3f} ms")
    print(f"  composed once, single apply:        {t_one:8.3f} ms   ({t_seq / t_one:.1f}x)")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    bench_mux()
    bench_circulant(L=8, A=4, B=8192)
    bench_circulant(L=16, A=4, B=1024)
    bench_compose()

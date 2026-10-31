import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("FASTQML_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))))
from fastqml import window_triton as wt  # noqa: E402

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


cases = [(14, 4096, 0, 7), (14, 4096, 7, 7), (12, 4096, 0, 6), (12, 4096, 6, 6), (12, 4096, 7, 4), (12, 4096, 8, 4),
         (10, 32768, 3, 5)]
cfgs = [(16, 4), (32, 4), (32, 8), (64, 4), (64, 8), (128, 8)]
for nq, B, start, width in cases:
    dim = 1 << width
    re = torch.randn(B, *([2] * nq), device=dev)
    im = torch.randn(B, *([2] * nq), device=dev)
    U_re, U_im = torch.randn(dim, dim, device=dev), torch.randn(dim, dim, device=dev)
    bytes_moved = 4 * re.numel() * 4
    print(f"nq={nq} B={B} window=({start},{width}) dim={dim} lo={1 << (nq - start - width)}:")
    best = None
    for block_r, warps in cfgs:
        for block_k in sorted({min(dim, 16), min(dim, 32), min(dim, 64)}):
            wt.FWD_CFG[dim] = (block_r, warps, block_k)
            try:
                t = bench(lambda: wt._fwd(re, im, U_re, U_im, start, width))
            except Exception as e:
                print(f"    R={block_r:3d} w={warps} K={block_k:2d}: {type(e).__name__}: {str(e)[:70]}")
                continue
            gbs = bytes_moved / t / 1e9
            best = (gbs, block_r, warps, block_k) if best is None or gbs > best[0] else best
            print(f"    R={block_r:3d} w={warps} K={block_k:2d}: {t * 1e3:7.3f} ms  {gbs:6.0f} GB/s")
    print(f"  best: {best}")

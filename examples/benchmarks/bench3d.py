import argparse
import gc
import json
import math
import os
import sys
import time
import traceback

import numpy as np

REPO = os.environ.get("FASTQML_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
MEM_FRACTION = 0.9
T_BUDGET = 15.0
A = 4


def edges(nq):
    even = [(i, i + 1) for i in range(0, nq - 1, 2)]
    odd = [(i, i + 1) for i in range(1, nq - 1, 2)]
    t1 = [(i, i + 1, i + 2) for i in range(0, nq - 2, 3)]
    t2 = [(i, i + 1, i + 2) for i in range(1, nq - 2, 3)]
    return even, odd, t1, t2


def lattice(nq):
    m = nq - 2
    return 1 << ((m + 1) // 2), 1 << (m // 2)


def gates_per_layer(family, nq):
    if family == "vqc":
        return 3 * nq
    if family == "multi":
        even, odd, t1, t2 = edges(nq)
        return 2 * len(even) + 2 * len(odd) + len(t1) + len(t2)
    return 1


def layout(family, nq, G_target):
    gpl = gates_per_layer(family, nq)
    L = max(1, round(G_target / gpl))
    return L, L * gpl


def make_params(family, nq, L):
    rng = np.random.default_rng(0)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)
    if family == "vqc":
        return {"ry": f(L, nq), "rz": f(L, nq), "cry": f(L, nq - 1)}
    if family == "multi":
        even, odd, _, _ = edges(nq)
        return {"xx": f(L, len(even)), "yy": f(L, len(odd)), "zz": f(L, len(even)), "crx": f(L, len(odd))}
    L1, L2 = lattice(nq)
    return {"ang": f(L, L1, L2, 4)}


def make_data(nq, B):
    rng = np.random.default_rng(1)
    amp = rng.random((B, 1 << nq), dtype=np.float32)
    amp /= np.linalg.norm(amp, axis=1, keepdims=True)
    tgt = rng.random((B, 1 << nq), dtype=np.float32)
    tgt /= tgt.sum(axis=1, keepdims=True)
    return amp, tgt


def ustack(xp, stack, a):
    def rot(t, p):
        c, s, cp, sp = xp.cos(t / 2), xp.sin(t / 2), xp.cos(p / 2), xp.sin(p / 2)
        re = stack([stack([c * cp, -s * cp], -1), stack([s * cp, c * cp], -1)], -2)
        im = stack([stack([-c * sp, s * sp], -1), stack([s * sp, c * sp], -1)], -2)
        return re, im
    r0, i0 = rot(a[..., 0], a[..., 1])
    r1, i1 = rot(a[..., 2], a[..., 3])
    kr = lambda x, y: xp.einsum("...ik,...jl->...ijkl", x, y).reshape(*x.shape[:-2], 4, 4)
    kre, kim = kr(r0, r1) - kr(i0, i1), kr(r0, i1) + kr(i0, r1)
    cnot = lambda k: stack([k[..., 0, :], k[..., 1, :], k[..., 3, :], k[..., 2, :]], -2)
    return cnot(kre), cnot(kim)


def bench(fn, sync):
    t_start = time.perf_counter()
    t0 = t_start
    sync(fn())
    t1 = time.perf_counter() - t0
    while time.perf_counter() - t_start < 0.3:
        t0 = time.perf_counter()
        sync(fn())
        t1 = min(t1, time.perf_counter() - t0)
    if t1 > 1.0:
        ts = [t1]
        for _ in range(2):
            t0 = time.perf_counter()
            sync(fn())
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    n = max(1, math.ceil(0.1 / max(t1, 1e-7)))
    blocks = []
    for _ in range(5):
        out = None
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn()
        sync(out)
        blocks.append((time.perf_counter() - t0) / n)
    return float(np.median(blocks))


def is_oom(e):
    s = f"{type(e).__name__} {e}"
    return "OutOfMemory" in s or "RESOURCE_EXHAUSTED" in s or "out of memory" in s.lower()


def short(e):
    return f"{type(e).__name__}: {str(e).strip().splitlines()[0][:160] if str(e).strip() else ''}"


class TorchSide:
    def __init__(self, family, nq):
        import torch
        tf32 = os.environ.get("FQ_TF32", "1") == "1"
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
        sys.path.insert(0, REPO)
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 4096
        from fastqml import circulant as cc
        from fastqml import gate_fusion as gf
        gf.USE_TRITON_WINDOW = os.environ.get("FQ_TRITON", "1") == "1"
        if os.environ.get("FQ_MAX_WINDOW"):
            gf.MAX_WINDOW = int(os.environ["FQ_MAX_WINDOW"])
        self.torch, self.family, self.nq = torch, family, nq
        dev, dt = torch.device("cuda"), torch.float32
        self.dev = dev
        if family == "vqc":
            cry_plans = gf.compute_two_qubit_plans([(q, q + 1) for q in range(nq - 1)], nq, dev, dt,
                                                   commute_fn=gf.cry_commute)
            toff_plans = gf.compute_three_qubit_plans([(0, 1, 2)], nq, dev, dt, commute_fn=gf.toffoli_commute)
            T_re, T_im = gf.build_toffoli_8x8(1, dev, dt)

            def layer(re, im, ry, rz, cry):
                G_re, G_im = gf.build_ry_rz_gates(ry, rz)
                re, im = gf.apply_su2_layer(re, im, G_re, G_im)
                for plan in cry_plans:
                    re, im = gf.apply_disjoint_cry_layer(re, im, plan, cry.index_select(0, plan["k_idx"]))
                for plan in toff_plans:
                    re, im = gf.apply_disjoint_three_qubit_layer(re, im, plan, T_re, T_im)
                return re, im
            seq = ([(("su2", q), (q,)) for q in range(nq)] + [(("cry", q), (q, q + 1)) for q in range(nq - 1)]
                   + [("toff", (0, 1, 2))])
            splan = gf.compile_sequence(seq, nq, dev, dt)

            def layer_fused(re, im, ry, rz, cry):
                G_re, G_im = gf.build_ry_rz_gates(ry, rz)
                C_re, C_im = gf.build_cry_4x4(cry)
                mats = {("su2", q): (G_re[q], G_im[q]) for q in range(nq)}
                mats.update({("cry", q): (C_re[q], C_im[q]) for q in range(nq - 1)})
                mats["toff"] = (T_re[0], T_im[0])
                return gf.apply_sequence(re, im, splan, mats)
            self.keys, self.layers = ("ry", "rz", "cry"), {"graph": layer, "fused+graph": layer_fused}
            self.flat = False
        elif family == "multi":
            even, odd, t1, t2 = edges(nq)
            p_even = gf.compute_two_qubit_plans(even, nq, dev, dt, commute_fn=gf.symmetric_commute)
            p_odd = gf.compute_two_qubit_plans(odd, nq, dev, dt, commute_fn=gf.cry_commute)
            p_odd_s = gf.compute_two_qubit_plans(odd, nq, dev, dt, commute_fn=gf.symmetric_commute)
            p_zz = gf.compute_two_qubit_plans(even, nq, dev, dt, commute_fn=gf.diagonal_commute)
            p_t1 = gf.compute_three_qubit_plans(t1, nq, dev, dt, commute_fn=gf.toffoli_commute)
            p_t2 = gf.compute_three_qubit_plans(t2, nq, dev, dt, commute_fn=gf.cswap_commute) if t2 else []
            T_re, T_im = gf.build_toffoli_8x8(len(t1), dev, dt)
            C_re, C_im = gf.build_cswap_8x8(max(len(t2), 1), dev, dt)

            def two(re, im, plans, builder, ang):
                for plan in plans:
                    G_re, G_im = builder(ang.index_select(0, plan["k_idx"]))
                    re, im = gf.apply_disjoint_two_qubit_layer(re, im, plan, G_re, G_im)
                return re, im

            def three(re, im, plans, G_re, G_im):
                for plan in plans:
                    idx = plan["k_idx"]
                    re, im = gf.apply_disjoint_three_qubit_layer(re, im, plan, G_re.index_select(0, idx),
                                                                 G_im.index_select(0, idx))
                return re, im

            def layer(re, im, xx, yy, zz, crx, zz_diag=False):
                re, im = two(re, im, p_even, gf.build_ising_xx_4x4, xx)
                re, im = two(re, im, p_odd_s, gf.build_ising_yy_4x4, yy)
                if zz_diag:
                    re, im = gf.apply_diagonal_layer(re, im, *gf.build_ising_zz_diag(zz, even, nq))
                else:
                    re, im = two(re, im, p_zz, gf.build_ising_zz_4x4, zz)
                re, im = two(re, im, p_odd, gf.build_crx_4x4, crx)
                re, im = three(re, im, p_t1, T_re, T_im)
                re, im = three(re, im, p_t2, C_re, C_im)
                return re, im
            seq = ([(("xx", k), e) for k, e in enumerate(even)] + [(("yy", k), e) for k, e in enumerate(odd)]
                   + [(("zz", k), e) for k, e in enumerate(even)] + [(("crx", k), e) for k, e in enumerate(odd)]
                   + [(("tof", k), t) for k, t in enumerate(t1)] + [(("csw", k), t) for k, t in enumerate(t2)])
            splan = gf.compile_sequence(seq, nq, dev, dt)

            def layer_fused(re, im, xx, yy, zz, crx):
                mats = {}
                for name, G, n in (("xx", gf.build_ising_xx_4x4(xx), len(even)), ("yy", gf.build_ising_yy_4x4(yy), len(odd)),
                                   ("zz", gf.build_ising_zz_4x4(zz), len(even)), ("crx", gf.build_crx_4x4(crx), len(odd)),
                                   ("tof", (T_re, T_im), len(t1)), ("csw", (C_re, C_im), len(t2))):
                    for k in range(n):
                        mats[(name, k)] = (G[0][k], G[1][k])
                return gf.apply_sequence(re, im, splan, mats)
            self.keys = ("xx", "yy", "zz", "crx")
            self.layers = {"graph+diag": lambda re, im, xx, yy, zz, crx: layer(re, im, xx, yy, zz, crx, True),
                           "fused+graph": layer_fused}
            self.flat = False
        else:
            L1, L2 = lattice(nq)
            plan = cc.circulant_plan(L1, L2, dev, dt)

            def layer_dense(re, im, ang):
                U_re, U_im = ustack(torch, torch.stack, ang)
                M_re, M_im = cc.circulant_dense(U_re, U_im, plan)
                return re @ M_re.T - im @ M_im.T, re @ M_im.T + im @ M_re.T

            def layer_k(re, im, ang):
                U_re, U_im = ustack(torch, torch.stack, ang)
                return cc.circulant_apply_k(re, im, U_re, U_im, plan)
            self.keys, self.layers = ("ang",), {"dense+graph": layer_dense, "kspace+graph": layer_k}
            self.flat = True
        for name in [n for n in self.layers if n in ("graph", "graph+diag", "fused+graph", "kspace+graph")]:
            self.layers[name.replace("graph", "compile+graph")] = torch.compile(self.layers[name], dynamic=False)

    def model(self, variant, P):
        torch, nq, flat = self.torch, self.nq, self.flat
        lf = self.layers[variant]
        slices = [P[k] for k in self.keys]
        L = slices[0].shape[0]

        def fwd(amp):
            B = amp.shape[0]
            re = amp if flat else amp.reshape(B, *([2] * nq))
            im = torch.zeros_like(re)
            for l in range(L):
                re, im = lf(re, im, *[s[l] for s in slices])
            return (re * re + im * im).reshape(B, -1)
        return fwd

    def free(self):
        gc.collect()
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()

    def capture(self, body, warm=3):
        torch = self.torch
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warm):
                body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = body()
        return g, out

    def run(self, family, nq, B, L, P_np, amp_t, tgt_t, skip):
        torch = self.torch
        sync = lambda _: torch.cuda.synchronize()
        res = {"variants": {}}
        for v in self.layers:
            if v in ("graph", "graph+diag") and L * gates_per_layer(family, nq) > 512:
                continue
            if v == "fused+graph" and L * gates_per_layer(family, nq) > 4096:
                continue
            vr = res["variants"][v] = {}
            P = {k: torch.tensor(x, device=self.dev, requires_grad=True) for k, x in P_np.items()}
            model = self.model(v, P)
            if "fwd" not in skip:
                try:
                    t0 = time.perf_counter()

                    def fbody():
                        with torch.no_grad():
                            return model(amp_t)
                    g, out = self.capture(fbody, warm=2)
                    torch.cuda.synchronize()
                    vr["fwd_setup_s"] = time.perf_counter() - t0
                    g.replay()
                    res.setdefault("probs_head", out[:2].float().cpu().numpy().tolist())
                    vr["fwd"] = bench(g.replay, sync)
                except Exception as e:
                    vr["fwd_err"] = "OOM" if is_oom(e) else short(e)
                g = out = None
                self.free()
            if "step" not in skip:
                try:
                    def sbody():
                        for p in P.values():
                            p.grad = None
                        loss = ((model(amp_t) - tgt_t) ** 2).mean()
                        loss.backward()
                        return loss
                    t0 = time.perf_counter()
                    sbody()
                    for p in P.values():
                        p.grad = None
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    sbody()
                    torch.cuda.synchronize()
                    vr["mem"] = torch.cuda.max_memory_allocated() - base
                    for p in P.values():
                        p.grad = None
                    g, loss = self.capture(sbody, warm=1)
                    torch.cuda.synchronize()
                    vr["step_setup_s"] = time.perf_counter() - t0
                    vr["step"] = bench(g.replay, sync)
                except Exception as e:
                    vr["step_err"] = "OOM" if is_oom(e) else short(e)
                g = loss = None
                for p in P.values():
                    p.grad = None
            P = model = None
            self.free()
        return res


class JaxSide:
    def __init__(self, family, nq):
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(MEM_FRACTION))
        import jax
        import jax.numpy as jnp
        import pennylane as qml
        from pennylane.devices.qubit import apply_operation
        self.jax, self.jnp, self.family, self.nq = jax, jnp, family, nq
        ap = apply_operation
        if family == "vqc":
            def body(state, lp):
                ry, rz, cry = lp
                for q in range(nq):
                    state = ap(qml.RY(ry[q], wires=q), state)
                    state = ap(qml.RZ(rz[q], wires=q), state)
                for q in range(nq - 1):
                    state = ap(qml.CRY(cry[q], wires=[q, q + 1]), state)
                return ap(qml.Toffoli(wires=[0, 1, 2]), state), None
            self.keys = ("ry", "rz", "cry")
        elif family == "multi":
            even, odd, t1, t2 = edges(nq)

            def body(state, lp):
                xx, yy, zz, crx = lp
                for k, w in enumerate(even):
                    state = ap(qml.IsingXX(xx[k], wires=w), state)
                for k, w in enumerate(odd):
                    state = ap(qml.IsingYY(yy[k], wires=w), state)
                for k, w in enumerate(even):
                    state = ap(qml.IsingZZ(zz[k], wires=w), state)
                for k, w in enumerate(odd):
                    state = ap(qml.CRX(crx[k], wires=w), state)
                for w in t1:
                    state = ap(qml.Toffoli(wires=w), state)
                for w in t2:
                    state = ap(qml.CSWAP(wires=w), state)
                return state, None
            self.keys = ("xx", "yy", "zz", "crx")
        else:
            L1, L2 = lattice(nq)
            N = L1 * L2
            r, c = np.arange(N) // L2, np.arange(N) % L2
            didx = ((r[:, None] - r[None, :]) % L1) * L2 + (c[:, None] - c[None, :]) % L2

            def body(state, ang):
                U_re, U_im = ustack(jnp, jnp.stack, ang[0])
                Adisp = jnp.fft.ifft2(U_re + 1j * U_im, axes=(0, 1)).reshape(N, A, A)
                M = Adisp[didx].transpose(2, 0, 3, 1).reshape(A * N, A * N).astype(jnp.complex64)
                return ap(qml.QubitUnitary(M, wires=range(nq)), state), None
            self.keys = ("ang",)
        self.body = body

    def fns(self, variant):
        jax, jnp, nq, body = self.jax, self.jnp, self.nq, self.body

        def circuit(params, amp):
            state = amp.astype(jnp.complex64).reshape((2,) * nq)
            if variant.startswith("scan"):
                K = int(variant.split(":")[1]) if ":" in variant else 1
                state, _ = jax.lax.scan(body, state, params, unroll=K)
            else:
                for l in range(params[0].shape[0]):
                    state, _ = body(state, tuple(p[l] for p in params))
            flat = state.reshape(-1)
            return flat.real ** 2 + flat.imag ** 2
        batched = jax.vmap(circuit, in_axes=(None, 0))

        def loss(params, amps, tgts):
            return jnp.mean((batched(params, amps) - tgts) ** 2)
        return jax.jit(batched), jax.jit(jax.value_and_grad(loss))

    def variants(self, family, nq, L):
        gpl = gates_per_layer(family, nq)
        small = L * gpl <= 512 if family != "circ" else L <= 8
        K = min(L, max(1, round(100 / gpl)) if family != "circ" else 8)
        v = ["scan"] + (["unrolled"] if small else ([f"scan:{K}"] if K > 1 else []))
        return v, v

    def precompile(self, family, nq, cfgs, threads=8):
        from concurrent.futures import ThreadPoolExecutor
        jax, jnp = self.jax, self.jnp
        self.cache = getattr(self, "cache", {})
        jobs = []
        for B, L in cfgs:
            P = make_params(family, nq, L)
            pspec = tuple(jax.ShapeDtypeStruct(P[k].shape, jnp.float32) for k in self.keys)
            aspec = jax.ShapeDtypeStruct((B, 1 << nq), jnp.float32)
            fwd_v, step_v = self.variants(family, nq, L)
            for v in dict.fromkeys(fwd_v + step_v):
                fwd, vg = self.fns(v)
                for kind, f, args in (("fwd", fwd, (pspec, aspec)), ("step", vg, (pspec, aspec, aspec))):
                    if kind == "fwd" and v not in fwd_v or kind == "step" and v not in step_v:
                        continue
                    t0 = time.perf_counter()
                    try:
                        jobs.append(((B, L, v, kind), f.lower(*args), time.perf_counter() - t0))
                    except Exception as e:
                        self.cache[(B, L, v, kind)] = (e, 0.0)

        def comp(job):
            key, low, t_low = job
            t0 = time.perf_counter()
            try:
                return key, low.compile(), t_low + time.perf_counter() - t0
            except Exception as e:
                return key, e, t_low
        with ThreadPoolExecutor(threads) as ex:
            for key, c, dt in ex.map(comp, jobs):
                self.cache[key] = (c, dt)

    def compiled(self, key, f, args):
        if key in getattr(self, "cache", {}):
            c, dt = self.cache.pop(key)
            if isinstance(c, Exception):
                raise c
            return c, dt
        t0 = time.perf_counter()
        return f.lower(*args).compile(), time.perf_counter() - t0

    def run(self, family, nq, B, L, P_np, amps, tgts, skip):
        jax = self.jax
        sync = jax.block_until_ready
        params = tuple(self.jnp.asarray(P_np[k]) for k in self.keys)
        fwd_v, step_v = self.variants(family, nq, L)
        limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", float("inf"))
        res = {"variants": {}}
        for v in dict.fromkeys(fwd_v + step_v):
            vr = res["variants"][v] = {}
            fwd, vg = self.fns(v)
            if "fwd" not in skip and v in fwd_v:
                try:
                    cf, vr["fwd_compile_s"] = self.compiled((B, L, v, "fwd"), fwd, (params, amps))
                    t0 = time.perf_counter()
                    out = sync(cf(params, amps))
                    vr["fwd_setup_s"] = time.perf_counter() - t0
                    res.setdefault("probs_head", np.asarray(out[:2]).tolist())
                    out = None
                    vr["fwd"] = bench(lambda: cf(params, amps), sync)
                except Exception as e:
                    vr["fwd_err"] = "OOM" if is_oom(e) else short(e)
                cf = out = None
            if "step" not in skip and v in step_v:
                try:
                    cg, vr["step_compile_s"] = self.compiled((B, L, v, "step"), vg, (params, amps, tgts))
                    t0 = time.perf_counter()
                    ma = cg.memory_analysis()
                    vr["mem_needed"] = ma.temp_size_in_bytes + ma.output_size_in_bytes
                    if vr["mem_needed"] + ma.argument_size_in_bytes > limit:
                        if v == "scan" and vr["mem_needed"] > 1.5 * limit:
                            skip = dict(skip, step="OOM")
                        raise MemoryError("RESOURCE_EXHAUSTED: needs "
                                          f"{(vr['mem_needed'] + ma.argument_size_in_bytes) / 2**30:.1f} GiB")
                    sync(cg(params, amps, tgts))
                    vr["step_setup_s"] = time.perf_counter() - t0
                    vr["step"] = bench(lambda: cg(params, amps, tgts), sync)
                    vr["mem"] = vr["mem_needed"]
                except Exception as e:
                    vr["step_err"] = "OOM" if is_oom(e) else short(e)
                cg = None
            gc.collect()
        return res


def summarize(res):
    V = res.get("variants", {})
    out = {}
    for m in ("fwd", "step"):
        ok = {v: r[m] for v, r in V.items() if m in r}
        if ok:
            best = min(ok, key=ok.get)
            out[m] = {"status": "ok", "t": ok[best], "variant": best}
            if m == "step":
                out["mem"] = {"status": "ok", "bytes": V[best]["mem"], "variant": best}
        else:
            errs = [r.get(m + "_err") for r in V.values() if r.get(m + "_err")]
            st = res.get("skip", {}).get(m) or ("OOM" if errs and all(e == "OOM" for e in errs) else
                                                ("error: " + errs[0] if errs else
                                                 ("error: " + res["fatal"] if "fatal" in res else "not run")))
            out[m] = {"status": st}
            if m == "step":
                out["mem"] = {"status": st}
    if out["fwd"]["status"] == "ok" and out["step"]["status"] == "ok":
        out["bwd"] = {"status": "ok", "t": max(out["step"]["t"] - out["fwd"]["t"], 1e-7)}
    else:
        bad = out["step"] if out["step"]["status"] != "ok" else out["fwd"]
        out["bwd"] = {"status": bad["status"]}
    res["best"] = out
    return res


def worker(args):
    fam, fw, nq = args.family, args.fw, args.nq
    Bs = [int(x) for x in args.Bs.split(",")]
    Gs = [int(x) for x in args.Gs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    side = None
    hist = {"fwd": [], "step": []}
    if fw == "jax" and not args.no_precompile:
        todo = [(B, layout(fam, nq, Gt)[0]) for B in Bs for Gt in Gs
                if not os.path.exists(os.path.join(args.out_dir, f"{fam}_{fw}_nq{nq}_B{B}_G{Gt}.json"))]
        if todo:
            side = JaxSide(fam, nq)
            t0 = time.perf_counter()
            side.precompile(fam, nq, todo)
            print(f"{fam} jax nq={nq}: precompiled {len(todo)} configs in {time.perf_counter() - t0:.0f}s", flush=True)
    for B in Bs:
        data = None
        for Gt in Gs:
            L, G = layout(fam, nq, Gt)
            stem = os.path.join(args.out_dir, f"{fam}_{fw}_nq{nq}_B{B}_G{Gt}")
            if os.path.exists(stem + ".json"):
                r = json.load(open(stem + ".json"))
                for m in hist:
                    b = r.get("best", {}).get(m, {})
                    hist[m].append((B, G, b.get("t"), b.get("status")))
                continue
            res = {"family": fam, "fw": fw, "nq": nq, "B": B, "G_target": Gt, "L": L, "G": G}
            if os.path.exists(stem + ".started"):
                res["skip"] = {"fwd": "crashed", "step": "crashed"}
                res["variants"] = {}
                json.dump(summarize(res), open(stem + ".json", "w"))
                os.remove(stem + ".started")
                for m in hist:
                    hist[m].append((B, G, None, "crashed"))
                continue
            skip = {}
            for m in hist:
                for b, g, t, st in hist[m]:
                    if b <= B and g <= G and st == "OOM":
                        skip[m] = "OOM"
                    elif b <= B and g < G and t is not None and t * G / g > T_BUDGET:
                        skip.setdefault(m, "budget")
            res["skip"] = skip
            open(stem + ".started", "w").close()
            t0 = time.perf_counter()
            if len(skip) < 2:
                try:
                    if side is None:
                        side = TorchSide(fam, nq) if fw == "fastqml" else JaxSide(fam, nq)
                    if data is None:
                        amp, tgt = make_data(nq, B)
                        if fw == "fastqml":
                            data = (side.torch.from_numpy(amp).to(side.dev), side.torch.from_numpy(tgt).to(side.dev))
                        else:
                            data = (side.jnp.asarray(amp), side.jnp.asarray(tgt))
                        amp = tgt = None
                    res.update(side.run(fam, nq, B, L, make_params(fam, nq, L), *data, skip))
                except Exception as e:
                    res["fatal"] = short(e)
                    if is_oom(e):
                        res["skip"] = {"fwd": "OOM", "step": "OOM"}
                    traceback.print_exc()
            res["wall_s"] = time.perf_counter() - t0
            summarize(res)
            json.dump(res, open(stem + ".json", "w"))
            os.remove(stem + ".started")
            for m in hist:
                b = res["best"][m]
                hist[m].append((B, G, b.get("t"), b["status"]))
            print(f"{fam} {fw} nq={nq} B={B} G={G} (L={L}): "
                  + "  ".join(f"{m}={res['best'][m].get('t', res['best'][m]['status'])}" for m in ("fwd", "bwd"))
                  + f"  mem={res['best']['mem'].get('bytes', res['best']['mem']['status'])}  [{res['wall_s']:.0f}s]",
                  flush=True)
        data = None
        if side is not None and fw == "fastqml":
            side.free()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("family", choices=["vqc", "multi", "circ"])
    w.add_argument("fw", choices=["fastqml", "jax"])
    w.add_argument("--nq", type=int, required=True)
    w.add_argument("--Bs", required=True)
    w.add_argument("--Gs", required=True)
    w.add_argument("--out-dir", required=True)
    w.add_argument("--no-precompile", action="store_true")
    a = ap.parse_args()
    worker(a)

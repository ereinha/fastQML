import csv
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D

ORANGE, BLACK, VIOLET = "#eb6834", "#0b0b0b", "#6250d6"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
FAMILIES = {
    "vqc": "VQC: RY+RZ layers, CRY chain, Toffoli\n(single-qubit Kronecker fusion)",
    "multi": "Multi-qubit: IsingXX/YY/ZZ, CRX, Toffoli, CSWAP\n(edge-colouring fusion)",
    "circ": "Circulant: translation-equivariant layers\n(2 ancilla qubits x L1xL2 torus)",
}
METRICS = {"mem": "Peak memory (fwd+bwd)", "fwd": "Forward time", "bwd": "Backward time"}
VMAX = 1.0
FONT_SCALE = 1.0


def _lin(h):
    s = np.array([int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _to_oklab(h):
    return _M2 @ np.cbrt(_M1 @ _lin(h))


def _from_oklab(lab):
    lin = np.linalg.inv(_M1) @ (np.linalg.inv(_M2) @ lab) ** 3
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def diverging_cmap(n=256):
    a, m, b = _to_oklab(VIOLET), _to_oklab(BLACK), _to_oklab(ORANGE)
    half = n // 2
    cols = [_from_oklab(a + (m - a) * t) for t in np.linspace(0, 1, half, endpoint=False)]
    cols += [_from_oklab(m + (b - m) * t) for t in np.linspace(0, 1, n - half)]
    cmap = ListedColormap(np.clip(cols, 0, 1), name="violet_black_orange")
    cmap.set_under(VIOLET)
    cmap.set_over(ORANGE)
    return cmap


def load(d):
    R = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        R[(r["family"], r["fw"], r["nq"], r["B"], r["G_target"])] = r
    return R


def value(r, metric):
    if r is None:
        return None, "missing"
    b = r["best"][metric]
    if b["status"] != "ok":
        return None, b["status"]
    return (b["bytes"] if metric == "mem" else b["t"]), "ok"


def points(R, family, metric):
    keys = sorted({k[2:] for k in R if k[0] == family})
    out = []
    for nq, B, G in keys:
        fq, fq_st = value(R.get((family, "fastqml", nq, B, G)), metric)
        jx, jx_st = value(R.get((family, "jax", nq, B, G)), metric)
        if fq_st == "missing" or jx_st == "missing":
            continue
        out.append(dict(nq=nq, B=B, G=G, fq=fq, jx=jx, fq_st=fq_st, jx_st=jx_st,
                        r=None if fq is None or jx is None else math.log10(jx / fq)))
    return out


def sort_depth(ax):
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    xs = np.array([m[1] for m in ax._fq_markers], dtype=float)
    ys = np.array([m[2] for m in ax._fq_markers], dtype=float)
    zs = np.array([m[3] for m in ax._fq_markers], dtype=float)
    depth = proj3d.proj_transform(xs, ys, zs, M)[2]
    for rank, i in enumerate(np.argsort(-depth)):
        ax._fq_markers[i][0].set_zorder(10 + rank)


def draw(ax, pts, family, metric, cmap, norm):
    ax.set_facecolor(SURFACE)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(SURFACE)
        pane.set_edgecolor(GRID)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6)
        axis.label.set_color(INK2)
    X = lambda p: p["nq"]
    Y = lambda p: math.log2(p["B"])
    Z = lambda p: math.log2(p["G"])
    nqs_all = sorted({p["nq"] for p in pts})
    Bs_all = sorted({math.log2(p["B"]) for p in pts})
    zlo, zhi = min(Z(p) for p in pts), max(Z(p) for p in pts)
    for x in nqs_all:
        ax.plot([x] * len(Bs_all), Bs_all, [zlo] * len(Bs_all), color=GRID, lw=0.8, zorder=1)
        for y in Bs_all:
            ax.plot([x, x], [y, y], [zlo, zhi], color="#c4c3be", lw=0.9, zorder=1)
    for y in Bs_all:
        ax.plot(nqs_all, [y] * len(nqs_all), [zlo] * len(nqs_all), color=GRID, lw=0.8, zorder=1)
    ax.computed_zorder = False
    ax._fq_markers = []
    ms = 1.0 + 0.5 * (FONT_SCALE - 1.0)
    for p in pts:
        if p["r"] is not None:
            coll = ax.scatter([X(p)], [Y(p)], [Z(p)], c=[p["r"]], cmap=cmap, norm=norm, s=175 * ms, marker="o",
                              edgecolors=SURFACE, linewidths=1.2, depthshade=False, alpha=1.0)
        else:
            colour = ORANGE if p["fq"] is not None else (VIOLET if p["jx"] is not None else BLACK)
            coll = ax.scatter([X(p)], [Y(p)], [Z(p)], c=colour, s=230 * ms, marker="X", linewidths=0,
                              depthshade=False)
        ax._fq_markers.append((coll, X(p), Y(p), Z(p)))
    sort_depth(ax)
    nqs = sorted({p["nq"] for p in pts})
    Bs = sorted({p["B"] for p in pts})
    Gs = sorted({p["G"] for p in pts})
    FS = FONT_SCALE
    ax.set_xticks(nqs)
    ax.set_yticks([math.log2(b) for b in Bs])
    ax.set_yticklabels([f"{b:,}" for b in Bs])
    ax.set_zticks([math.log2(g) for g in Gs])
    ax.set_zticklabels([f"{g:,}" for g in Gs])
    ax.tick_params(colors=INK2, labelsize=7.5 * FS, pad=1 * FS)
    ax.set_xlabel("qubits", fontsize=9 * FS, labelpad=5 * FS)
    ax.set_ylabel("batch size", fontsize=9 * FS, labelpad=10 * FS)
    ax.set_zlabel("circulant layers" if family == "circ" else "gates", fontsize=9 * FS, labelpad=10 * FS)
    ax.view_init(elev=24, azim=-62)
    ax.set_box_aspect((1, 1, 1.15), zoom=1.0 + 0.2 * (FS - 1.0))
    sort_depth(ax)


def legend_handles():
    mk = lambda c, m, lab: Line2D([], [], linestyle="none", marker=m, markersize=11 * FONT_SCALE, markerfacecolor=c,
                                  markeredgecolor=SURFACE, label=lab)
    return [mk(INK2, "o", "both completed (colour = ratio)"),
            mk(ORANGE, "X", "only fastQML completed"),
            mk(VIOLET, "X", "only PennyLane+JAX completed"),
            mk(BLACK, "X", "neither completed")]


def colorbar(fig, cax, cmap, norm, vmax):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend="both", extendfrac=0.04)
    l3 = math.log10(3)
    cb.set_ticks([-vmax, -l3, 0, l3, vmax])
    big = f"≥{10 ** vmax:,.0f}x"
    cb.set_ticklabels([big, "3x", "matched", "3x", big])
    cb.ax.tick_params(labelsize=8 * FONT_SCALE, colors=INK2)
    cb.outline.set_edgecolor(GRID)
    cb.set_label("← PennyLane+JAX better        speed-up / memory ratio, log scale        "
                 "fastQML better →", fontsize=8 * FONT_SCALE, color=INK2)


def main(d, out):
    os.makedirs(out, exist_ok=True)
    R = load(d)
    cmap = diverging_cmap()
    vmax = VMAX
    norm = Normalize(-vmax, vmax)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})

    for f, ftitle in FAMILIES.items():
        for m, mtitle in METRICS.items():
            pts = points(R, f, m)
            if not pts:
                continue
            fig = plt.figure(figsize=(7.2, 7.4), facecolor=SURFACE)
            ax = fig.add_axes([0.0, 0.17, 1.0, 0.72], projection="3d")
            draw(ax, pts, f, m, cmap, norm)
            fig.text(0.04, 0.955, f"{mtitle}", fontsize=14, weight="bold", color=INK)
            fig.text(0.04, 0.905, ftitle.replace("\n", "  "), fontsize=9, color=INK2)
            fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=2,
                       frameon=False, fontsize=8, labelcolor=INK2)
            colorbar(fig, fig.add_axes([0.1, 0.055, 0.8, 0.022]), cmap, norm, vmax)
            fig.savefig(os.path.join(out, f"{f}_{m}.png"), dpi=150, facecolor=SURFACE)
            plt.close(fig)

    fig = plt.figure(figsize=(18, 20.5), facecolor=SURFACE)
    for i, (f, ftitle) in enumerate(FAMILIES.items()):
        for j, (m, mtitle) in enumerate(METRICS.items()):
            ax = fig.add_axes([0.005 + j * 0.333, 0.692 - i * 0.295, 0.33, 0.23], projection="3d")
            pts = points(R, f, m)
            if pts:
                draw(ax, pts, f, m, cmap, norm)
            ax.set_title(f"{mtitle}", fontsize=12, color=INK, weight="bold", pad=0)
        fig.text(0.012, 0.945 - i * 0.295, ftitle.replace("\n", "  "), fontsize=11.5, color=INK2, weight="bold")
    fig.text(0.012, 0.992, "fastQML vs PennyLane+JAX on an RTX 5090 (TF32 on for both)", fontsize=15,
             weight="bold", color=INK, va="top")
    fig.text(0.012, 0.973, "fastQML: windowed + time-fused gates, Triton window kernel, best of CUDA graph / "
             "torch.compile.   PennyLane+JAX: default.qubit kernels under jit + vmap, best of scan / unrolled.",
             fontsize=11, color=INK2, va="top")
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.047), ncol=4,
               frameon=False, fontsize=10, labelcolor=INK2)
    colorbar(fig, fig.add_axes([0.2, 0.029, 0.6, 0.01]), cmap, norm, vmax)
    fig.savefig(os.path.join(out, "all_3x3.png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(out, "points.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "metric", "nq", "batch", "gate_level", "fastqml", "fastqml_status",
                    "pennylane_jax", "pennylane_jax_status", "jax_over_fastqml"])
        for f in FAMILIES:
            for m in METRICS:
                for p in points(R, f, m):
                    w.writerow([f, m, p["nq"], p["B"], p["G"], p["fq"], p["fq_st"], p["jx"], p["jx_st"],
                                None if p["r"] is None else round(10 ** p["r"], 4)])
    for f in FAMILIES:
        for m in METRICS:
            pts = points(R, f, m)
            if not pts:
                continue
            ok = [p for p in pts if p["r"] is not None]
            fqw = [p for p in ok if p["r"] > 0.0414]
            jxw = [p for p in ok if p["r"] < -0.0414]
            best_fq = max(ok, key=lambda p: p["r"], default=None)
            best_jx = min(ok, key=lambda p: p["r"], default=None)
            fl = lambda c: sum(1 for p in pts if c(p))
            print(f"{f:5s} {m:3s}: {len(ok):3d} both-ok | fastQML >1.1x: {len(fqw):3d}  PL+JAX >1.1x: {len(jxw):3d}"
                  f" | only-fastQML {fl(lambda p: p['fq'] is not None and p['jx'] is None):2d}"
                  f"  only-PL+JAX {fl(lambda p: p['fq'] is None and p['jx'] is not None):2d}"
                  f"  neither {fl(lambda p: p['fq'] is None and p['jx'] is None):2d}"
                  + (f" | best fastQML {10 ** best_fq['r']:.1f}x @nq={best_fq['nq']},B={best_fq['B']},G={best_fq['G']}"
                     if best_fq and best_fq["r"] > 0 else "")
                  + (f" | best PL+JAX {10 ** -best_jx['r']:.1f}x @nq={best_jx['nq']},B={best_jx['B']},G={best_jx['G']}"
                     if best_jx and best_jx["r"] < 0 else ""))
    print(f"colour scale: +/-{vmax:g} decades (saturating)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

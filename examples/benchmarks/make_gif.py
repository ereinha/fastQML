import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot3d
from plot3d import (FAMILIES, INK, INK2, METRICS, SURFACE, VMAX, colorbar, diverging_cmap, draw,
                    legend_handles, load, points, sort_depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=44)
    ap.add_argument("--rows", default="vqc,multi,circ")
    ap.add_argument("--font-scale", type=float, default=2.2)
    ap.add_argument("--check-frame", action="store_true", help="also save the clockwise-extreme frame as PNG")
    a = ap.parse_args()
    plot3d.FONT_SCALE = FS = a.font_scale
    R = load(a.results)
    cmap, norm = diverging_cmap(), Normalize(-VMAX, VMAX)
    fams = [f for f in FAMILIES if f in a.rows.split(",")]
    n = len(fams)
    row_in, head_in, foot_in = 8.0, 1.5, 2.9
    H = head_in + row_in * n + foot_in
    fig = plt.figure(figsize=(22, H), facecolor=SURFACE)
    axes = []
    for i, f in enumerate(fams):
        row_top = H - head_in - i * row_in
        ax_bottom, ax_h = row_top - row_in + 1.7, row_in - 1.7 - 1.05
        for j, (m, mtitle) in enumerate(METRICS.items()):
            ax = fig.add_axes([0.0 + j * 0.32, ax_bottom / H, 0.28, ax_h / H], projection="3d")
            pts = points(R, f, m)
            if pts:
                draw(ax, pts, f, m, cmap, norm)
            ax.set_title(mtitle, fontsize=11 * FS, color=INK, weight="bold", pad=0)
            axes.append(ax)
        fig.text(0.012, (row_top - 0.05) / H, FAMILIES[f].replace("\n", "  "), fontsize=10 * FS, color=INK2,
                 weight="bold", va="top")
    fig.text(0.012, 1 - 0.1 / H, "fastQML vs PennyLane+JAX on an RTX 5090 (TF32 on for both)", fontsize=13 * FS,
             weight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.7 / H, "orange = fastQML better, violet = PennyLane+JAX better, black = matched;  "
             "X = only one side (or neither) completed", fontsize=9 * FS, color=INK2, va="top")
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 1.75 / H), ncol=2,
               frameon=False, fontsize=9 * FS, labelcolor=INK2, columnspacing=2.5, handletextpad=0.6)
    colorbar(fig, fig.add_axes([0.15, 1.0 / H, 0.7, 0.22 / H]), cmap, norm, VMAX)

    fig.set_dpi(a.dpi)
    frames = []
    for k in range(a.frames):
        az = -52 - 28 * math.cos(2 * math.pi * k / a.frames)
        for ax in axes:
            ax.view_init(elev=24, azim=az)
            sort_depth(ax)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    palette = frames[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(a.out, save_all=True, append_images=quantized[1:], duration=int(1000 / a.fps), loop=0,
                      optimize=False)
    if a.check_frame:
        frames[a.frames // 2].save(os.path.splitext(a.out)[0] + "_frame_clockwise.png")
    print(a.out, f"{os.path.getsize(a.out) / 2**20:.1f} MiB, {a.frames} frames at {a.fps} fps "
          f"({a.frames / a.fps:.1f} s loop), {frames[0].size[0]}x{frames[0].size[1]} px")


if __name__ == "__main__":
    main()

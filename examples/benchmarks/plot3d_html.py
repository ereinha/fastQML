import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot3d import BLACK, FAMILIES, METRICS, ORANGE, VIOLET, VMAX, diverging_cmap, load, points


def fmt(metric, v):
    if v is None:
        return None
    if metric == "mem":
        return f"{v / 2**30:.2f} GiB" if v >= 2**28 else f"{v / 2**20:.2f} MiB"
    return f"{v * 1e3:.3f} ms" if v >= 1e-3 else f"{v * 1e6:.1f} us"


def main(d, out):
    R = load(d)
    cmap = diverging_cmap(11)
    scale = [[i / 10, "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in cmap(i / 10)[:3])] for i in range(11)]
    vmax = VMAX
    plots = []
    for f, ftitle in FAMILIES.items():
        for m, mtitle in METRICS.items():
            pts = points(R, f, m)
            for p in pts:
                p["hover"] = (f"qubits {p['nq']}, batch {p['B']:,}, {'layers' if f == 'circ' else 'gates'} "
                              f"{p['G']:,}<br>fastQML: {fmt(m, p['fq']) or p['fq_st']}<br>"
                              f"PennyLane+JAX: {fmt(m, p['jx']) or p['jx_st']}"
                              + ("" if p["r"] is None else
                                 f"<br><b>{10 ** abs(p['r']):.2f}x {'fastQML' if p['r'] > 0 else 'PennyLane+JAX'}"
                                 f" better</b>"))
            plots.append({"id": f"{f}_{m}", "title": mtitle, "family": ftitle.replace("\n", " "),
                          "zlabel": "circulant layers" if f == "circ" else "gates",
                          "pts": [{k: p[k] for k in ("nq", "B", "G", "r", "hover")} |
                                  {"fq_ok": p["fq"] is not None, "jx_ok": p["jx"] is not None} for p in pts]})
    html = TEMPLATE.replace("__DATA__", json.dumps({"plots": plots, "scale": scale, "vmax": vmax,
                                                     "orange": ORANGE, "violet": VIOLET, "black": BLACK}))
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>fastQML vs PennyLane+JAX</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root { color-scheme: light; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --rule:#d9d8d4; }
body { margin:0; background:var(--surface); color:var(--ink); font:14px/1.45 system-ui, sans-serif; }
main { max-width:1500px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:20px; margin:0 0 4px; } p { margin:0 0 10px; color:var(--ink2); max-width:110ch; }
.key { display:flex; flex-wrap:wrap; gap:6px 22px; align-items:center; margin:10px 0 18px; color:var(--ink2); }
.bar { width:260px; height:12px; border-radius:2px; }
.row h2 { font-size:15px; margin:18px 0 6px; color:var(--ink2); font-weight:600; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:10px; }
.cell { border:1px solid var(--rule); border-radius:6px; background:var(--surface); min-width:0; }
.cell h3 { font-size:14px; margin:8px 12px 0; } .plot { height:440px; }
.x { font-weight:700; font-size:16px; }
</style></head>
<body><main>
<h1>fastQML vs PennyLane+JAX: memory, forward and backward time</h1>
<p>RTX 5090 (WSL2), TF32 on in both. fastQML = windowed + time-fused gates with the Triton window kernel, best of
CUDA-graph / torch.compile variants; PennyLane+JAX = default.qubit kernels under jit + vmap, best of scan / unrolled loop.
Drag to rotate, hover a point for raw values.</p>
<div class="key"><span>PennyLane+JAX better</span><span class="bar" id="bar"></span><span>fastQML better</span>
<span id="ticks"></span>
<span><span class="x" style="color:__O__">&#x2716;</span> only fastQML completed</span>
<span><span class="x" style="color:__V__">&#x2716;</span> only PennyLane+JAX completed</span>
<span><span class="x" style="color:#0b0b0b">&#x2716;</span> neither completed</span></div>
<div id="rows"></div>
</main>
<script>
const D = __DATA__;
document.getElementById("bar").style.background =
  "linear-gradient(90deg," + D.scale.map(s => s[1]).join(",") + ")";
document.getElementById("ticks").textContent =
  "(log scale, black = matched, saturates at ≥" + Math.pow(10, D.vmax).toLocaleString() + "x; hover shows the exact ratio)";
const rows = document.getElementById("rows");
const fams = [...new Set(D.plots.map(p => p.family))];
for (const fam of fams) {
  const row = document.createElement("section"); row.className = "row";
  row.innerHTML = "<h2>" + fam + "</h2><div class='grid'></div>"; rows.appendChild(row);
  for (const P of D.plots.filter(p => p.family === fam)) {
    const cell = document.createElement("div"); cell.className = "cell";
    cell.innerHTML = "<h3>" + P.title + "</h3><div class='plot' id='" + P.id + "'></div>";
    row.querySelector(".grid").appendChild(cell);
    const ok = P.pts.filter(p => p.r !== null);
    const sets = [[D.orange, p => p.fq_ok && !p.jx_ok], [D.violet, p => !p.fq_ok && p.jx_ok],
                  [D.black, p => !p.fq_ok && !p.jx_ok]];
    const lb = v => Math.log2(v);
    // stems through every (qubits, batch) column + base lattice, so markers read as column + level
    const nqs = [...new Set(P.pts.map(p => p.nq))].sort((a, b) => a - b);
    const bs = [...new Set(P.pts.map(p => lb(p.B)))].sort((a, b) => a - b);
    const zs = P.pts.map(p => lb(p.G)), zlo = Math.min(...zs), zhi = Math.max(...zs);
    const sx = [], sy = [], sz = [];
    for (const x of nqs) for (const y of bs) { sx.push(x, x, null); sy.push(y, y, null); sz.push(zlo, zhi, null); }
    for (const x of nqs) { for (const y of bs) { sx.push(x); sy.push(y); sz.push(zlo); } sx.push(null); sy.push(null); sz.push(null); }
    for (const y of bs) { for (const x of nqs) { sx.push(x); sy.push(y); sz.push(zlo); } sx.push(null); sy.push(null); sz.push(null); }
    const traces = [{type: "scatter3d", mode: "lines", x: sx, y: sy, z: sz, hoverinfo: "skip",
      line: {color: "#c4c3be", width: 1.5}},
      {type: "scatter3d", mode: "markers", x: ok.map(p => p.nq), y: ok.map(p => lb(p.B)),
      z: ok.map(p => lb(p.G)), text: ok.map(p => p.hover), hoverinfo: "text",
      marker: {size: 9, color: ok.map(p => p.r), colorscale: D.scale, cmin: -D.vmax, cmax: D.vmax,
               line: {color: "#fcfcfb", width: 1}, opacity: 1}}];
    for (const [c, sel] of sets) {
      const f = P.pts.filter(sel);
      if (f.length) traces.push({type: "scatter3d", mode: "markers", x: f.map(p => p.nq), y: f.map(p => lb(p.B)),
        z: f.map(p => lb(p.G)), text: f.map(p => p.hover), hoverinfo: "text",
        marker: {size: 8, symbol: "x", color: c, line: {color: c, width: 1}}});
    }
    const Bs = [...new Set(P.pts.map(p => p.B))].sort((a, b) => a - b);
    const Gs = [...new Set(P.pts.map(p => p.G))].sort((a, b) => a - b);
    const ax = (title, vals, labels) => ({title: {text: title, font: {color: "#52514e"}}, tickvals: vals,
      ticktext: labels, gridcolor: "#d9d8d4", zerolinecolor: "#d9d8d4", backgroundcolor: "#fcfcfb",
      showbackground: true, tickfont: {color: "#52514e", size: 10}});
    Plotly.newPlot(P.id, traces, {
      margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: "#fcfcfb", showlegend: false,
      scene: {xaxis: ax("qubits", [...new Set(P.pts.map(p => p.nq))], null),
              yaxis: ax("batch", Bs.map(lb), Bs.map(b => b.toLocaleString())),
              zaxis: ax(P.zlabel, Gs.map(lb), Gs.map(g => g.toLocaleString())),
              camera: {eye: {x: 1.6, y: -1.6, z: 0.9}}, aspectmode: "cube"},
      hoverlabel: {bgcolor: "#fcfcfb", bordercolor: "#d9d8d4", font: {color: "#0b0b0b"}}},
      {displaylogo: false, responsive: true});
  }
}
</script></body></html>
"""
TEMPLATE = TEMPLATE.replace("__O__", ORANGE).replace("__V__", VIOLET)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

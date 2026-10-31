# 3D benchmark harness: fastQML vs PennyLane + JAX

Measures forward time, backward time and peak fwd+bwd memory over a qubits x batch x gate-count
grid for three model families, on the same GPU, each framework in its fastest configuration:

| family | one layer | fastQML path |
|---|---|---|
| `vqc` | RY+RZ on every qubit, CRY chain `(q, q+1)`, Toffoli(0,1,2) | per-layer API and `compile_sequence`; best of CUDA graph / per-layer `torch.compile` |
| `multi` | IsingXX (even bonds), IsingYY (odd), IsingZZ (even, diagonal), CRX (odd), Toffoli triples, CSWAP triples | same, plus the diagonal-layer path for ZZ |
| `circ` | one translation-equivariant circulant operator on a 2-ancilla x L1xL2 torus register | `circulant_dense` / `circulant_apply_k`, best of eager graph / compiled |

PennyLane side: `default.qubit`'s own `apply_operation` kernels under `jax.jit` + `jax.vmap`,
best of `lax.scan` and an unrolled loop (`circ` applies the operator as a dense `QubitUnitary`).
Both frameworks get 90% of GPU memory; out-of-memory and over-budget (> 15 s per call) points are
recorded as such, not retried.

## Files

- `bench3d.py` — one worker per (family, framework, qubit count); writes one JSON per point.
- `plot3d.py` — the nine 3D scatter plots, the 3x3 panel and `points.csv`.
- `plot3d_html.py` — the same plots as one interactive (rotatable, hoverable) HTML page.
- `make_gif.py` — the rotating-view GIF of the panel shown in the top-level README (GitHub strips
  scripts from READMEs, so the interactive page cannot be embedded there; a GIF can).
- `tune_triton.py` — tile-config sweep for `fastqml.window_triton`'s forward kernel.
- `results/grid3d_final/` — the RTX 5090 results shown in `docs/benchmarks/` (576 points;
  per-point status, best variant, every variant's timings, and `max_abs_dP_vs_pennylane_jax`).

## Running

Two Python environments keep torch's CUDA wheels and `jax[cuda12]` from fighting:

```bash
uv venv --python 3.12 ~/.cache/fqbench/torch
uv pip install --python ~/.cache/fqbench/torch/bin/python torch --index-url https://download.pytorch.org/whl/cu128 numpy
uv venv --python 3.12 ~/.cache/fqbench/pl
uv pip install --python ~/.cache/fqbench/pl/bin/python "jax[cuda12]==0.7.1" pennylane optax   # PennyLane 0.45 pins JAX 0.7.1
```

Then, for each family / qubit count (the worker skips points that already have a JSON):

```bash
export FASTQML_REPO=/path/to/fastQML          # defaults to this checkout
T=~/.cache/fqbench/torch/bin/python; P=~/.cache/fqbench/pl/bin/python
for fam in vqc multi; do for nq in 4 6 8 10 12 14; do
  $T bench3d.py worker $fam fastqml --nq $nq --Bs 1,32,1024,32768 --Gs 64,512,4096,32768 --out-dir results/mygrid
  $P bench3d.py worker $fam jax     --nq $nq --Bs 1,32,1024,32768 --Gs 64,512,4096,32768 --out-dir results/mygrid
done; done
for nq in 4 6 8 10 12 14; do
  $T bench3d.py worker circ fastqml --nq $nq --Bs 1,32,1024,32768 --Gs 1,8,64,512 --out-dir results/mygrid
  $P bench3d.py worker circ jax     --nq $nq --Bs 1,32,1024,32768 --Gs 1,8,64,512 --out-dir results/mygrid
done
python plot3d.py results/mygrid plots && python plot3d_html.py results/mygrid plots/interactive_3d.html
```

Environment switches read by the fastQML worker: `FQ_TRITON=0` forces the cuBLAS window path,
`FQ_TF32=0` runs true fp32 (default is TF32 on), `FQ_MAX_WINDOW=n` overrides the window size.
`TORCHINDUCTOR_COMPILE_THREADS` caps Inductor's compile pool (it defaults to one process per CPU).

Run the two frameworks one after the other, never concurrently: both timing loops assume an idle GPU.

## Reading the plots

Colour is `log10(PennyLane+JAX / fastQML)` of the metric, centred at black (matched) and
saturating at 10x: orange = fastQML faster or leaner, violet = PennyLane+JAX. Spheres are points
where both completed; an X marks a point where only fastQML (orange), only PennyLane+JAX
(violet) or neither (black) completed. Exact per-point values are in `points.csv` and in the hover
text of the interactive page.

Precision: with TF32 on, fastQML's window matmuls round every pass to 10-bit mantissas and the
error compounds with depth (~1e-3 relative at a few hundred gates, unusable beyond a few
thousand), whereas PennyLane's 1-2-qubit kernels are small einsums that XLA runs in full fp32
regardless of the flag. For circuits deeper than a few hundred gates, or whenever probabilities
must be exact to better than ~1e-3, run fastQML with `torch.backends.cuda.matmul.allow_tf32 =
False` (the window kernel then uses three-pass TF32, `tf32x3`, which is fp32-class: 1e-6 relative);
the fp32-mode timings are reported in the top-level README.

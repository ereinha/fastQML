# fastQML

A small PyTorch quantum-circuit simulation toolkit built around two ideas:

1. **Layer-fusion via Kronecker product + matmul.** When a circuit applies a single-qubit gate to many qubits in parallel, the combined operator is the Kronecker product of those gates. Build the product once, apply via a single matmul on the flat state — replaces N strided per-qubit kernels with one Tensor-Core matmul.

2. **Disjoint multi-qubit-layer fusion via edge coloring.** A layer that applies 2- or 3-qubit gates to many tuples is partitioned into "colors" — each color is a set of tuples that act on pairwise-disjoint qubits AND commute with the gates in later colors. Each color becomes one matmul.

Both shapes collapse to the same kernel: `(B, 2^nq) @ (2^nq, 2^nq)`. On a modern GPU this saturates Tensor Cores, and the whole forward+backward fits cleanly into a CUDA graph.

The result: a per-sample throughput **~36× faster than PennyLane lightning.qubit** (C++ backend) for the same circuit at `n_qubits=8, n_layers=3` — see benchmark below.

---

## What's in here

| File | Purpose |
|---|---|
| `fastqml/gate_fusion.py` | Gate matrix builders (RX/RY/RZ/Rot/PhaseShift, X/Y/Z/H/S/T, CRX/CRY/CRZ/CPhase/IsingXX/YY/ZZ, CNOT/CY/CZ/SWAP/ISWAP, Toffoli, CSWAP) + fused layer-apply functions + edge coloring with commutation predicates |
| `fastqml/torch_sim_real.py` | Raw per-gate kernels in `(B, 2, 2, ..., 2)` state layout, plus state-utility helpers, block-encoding subroutines, and the Hadamard-test overlap primitive |
| `examples/minimal_cuda_graph_model.py` | A small variational-quantum-circuit-style `nn.Module`, plus benchmarks for CPU eager / CPU `torch.compile` / CUDA eager / CUDA graph / PennyLane `lightning.qubit` |
| `fastqml/register_ops.py` | Multiplexed (address-controlled) rotations for single large registers — angle indexed by computational-basis address bits — with optional Triton kernels (autograd included), pure-torch fallback, and dense composition helpers (`compose_dense` / `apply_dense`) |
| `fastqml/circulant.py` | Translation-equivariant circulant layers for lattice registers: momentum-block operators `M = F' (sum_k U(k)) F` as dense matrices or FFT-style applies |
| `examples/register_ops_bench.py` | Benchmarks for the multiplexer (Triton / eager / per-batch chain), circulant applies, and dense composition |
| `test_register_ops.py` | 15 numerical tests: multiplexers vs explicit dense references at fp64, Triton forward/backward vs eager autograd, circulant unitarity / dense-vs-k-space equality / translation equivariance |
| `test_pennylane_equivalence.py` | 32 numerical equivalence tests against PennyLane `default.qubit` covering every gate, plus multi-pair colors, chain ordering, and per-batch chains. Every test must pass at fp64 noise floor (max diff = 0 or ~1e-17) |

### Which module should I use?

In almost all training / inference workloads, **use `fastqml.gate_fusion` first**. It's the fast path — every common gate is exposed as a layer-fusing builder + apply pair, and the kron-product + matmul pattern is what makes the CUDA-graph + per-sample throughput numbers in the benchmark below work. If you're building an `nn.Module`, building a VQC, or anything that runs many gates on many qubits per forward, this is the right tool.

**Reach for `fastqml.torch_sim_real`** when one of these applies:

- **One-off single-gate ops.** Building a full `(nq, 2, 2)` gate stack to use `apply_su2_layer` for a single non-identity gate is wasteful. `apply_ry(re, im, theta, qubit, nq)` etc. apply one gate directly via `movedim` + stack ops — fewer launches when the work isn't layer-shaped.
- **State / amplitude / probability conversion.** `_amp_to_state_real(amp, nq)` initializes the `(B, 2, 2, ..., 2)` state from an amplitude vector. `_state_to_probs_real(re, im)` flattens `|ψ|²` to a probability vector. `_renormalize_real(re, im, nq)` re-normalizes after non-unitary operations. `gate_fusion` deliberately doesn't ship these — they aren't gates.
- **Block-encoded operators.** `apply_block_encoding_real(amp, ry_layers, cry_anchor, cry_data, ry_final, nq)` and `apply_block_encoding_complex(...)` post-select an ancilla qubit to encode a learnable sub-norm-1 linear operator. Useful for quantum LCU-style circuits or learned non-unitary maps.
- **Hadamard test for overlaps.** `hadamard_test_overlap(re_Q, im_Q, re_K, im_K, nq)` returns `Re<ψ_Q | ψ_K>` via the standard 1-ancilla Hadamard-test circuit — useful whenever you need a differentiable estimate of an inner product between two state vectors.

If you're just doing standard parametric circuits with rotations and entanglers, you only need `fastqml.gate_fusion`. For more exotic QML primitives (LCU, overlap-based losses, hybrid amplitude/probability workflows), you'll want both.

**Reach for `fastqml.register_ops` / `fastqml.circulant`** when your model is a *single large register* rather than many small ones — e.g. a lattice register where sites are basis states. The regime rule: many small registers are launch-bound (gate_fusion + CUDA graphs is the fast path); one big register is bandwidth-bound, and the fast path is minimizing passes over the state. `register_ops` provides the multiplexed rotation (angle indexed by basis-address bits — the uniformly controlled rotation of circuit synthesis), which per-batch gate primitives cannot express without a batch-fold that costs ~150x (see benchmark). `circulant` provides translation-equivariant lattice layers; combined with `compose_dense` for parameters that are static across many evaluations, whole gate cascades collapse to one matmul per block.

---

## Installation

### Python packages

For NVIDIA GPUs (Windows/Linux), pin a CUDA-matched PyTorch wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(Use `cu124` / `cu121` / etc. for older CUDA toolkits, or skip this and let `pip install fastqml` pull the default wheel for CPU / Apple silicon.)

Then install fastqml:

```bash
pip install fastqml
# or directly from GitHub for the latest:
pip install git+https://github.com/ereinha/fastQML.git
```

For editable / contributor install:

```bash
git clone https://github.com/ereinha/fastQML.git && cd fastQML
pip install -e .          # core
pip install -e .[test]    # also install pennylane + pennylane-lightning for the equivalence tests
```

The core toolkit (`fastqml.gate_fusion` + `fastqml.torch_sim_real`) needs only torch + numpy. PennyLane is an optional dep used for `test_pennylane_equivalence.py` and the PennyLane row in the example benchmark.

> **Note:** Triton is *not* a dependency. The gate_fusion fast path uses CUDA graphs (`torch.cuda.make_graphed_callables`), which capture raw CUDA kernel launches and do no codegen. `fastqml.register_ops` will use Triton kernels for its multiplexed rotations when Triton is importable (~4x over its pure-torch fallback) and falls back to pure torch otherwise. On Linux/macOS Triton ships transitively with `torch`; on Windows `pip install triton-windows`.

### C++ compiler (only needed for `torch.compile` on CPU)

CUDA users don't need any compiler — `torch.cuda.make_graphed_callables` works out of the box.

For CPU `torch.compile`, Inductor calls a C++ compiler to emit fused kernels.

**Linux** — install GCC if you don't have it:

```bash
sudo apt install build-essential          # Debian/Ubuntu
sudo dnf groupinstall "Development Tools" # Fedora/RHEL
```

**macOS** — install the Xcode Command Line Tools:

```bash
xcode-select --install
```

This puts `clang`/`clang++` on PATH; PyTorch picks them up automatically.

**Windows** — install Visual Studio 2022 Build Tools' C++ workload:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override `
  "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"
```

(~3 GB download.) After install, launch your shell from a **"Developer Command Prompt for VS 2022"** so `cl.exe` is on PATH, OR call `vcvars64.bat` first:

```powershell
cmd /c '"%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && python examples\minimal_cuda_graph_model.py'
```

If `cl.exe` isn't found at run-time the example's CPU compile path gracefully skips with the error and continues with the eager + CUDA paths.

---

## Quick example

### Apply a single layer of mixed single-qubit gates

```python
import torch
from fastqml import gate_fusion as gf

device, dtype = torch.device("cuda"), torch.float32
nq, B = 6, 32

state_re = torch.rand(B, *([2] * nq), device=device, dtype=dtype)
state_im = torch.zeros_like(state_re)

ry_angles = torch.randn(nq, device=device, dtype=dtype)
rz_angles = torch.randn(nq, device=device, dtype=dtype)

G_re, G_im = gf.build_ry_rz_gates(ry_angles, rz_angles)
state_re, state_im = gf.apply_su2_layer(state_re, state_im, G_re, G_im)
```

That's one (B, 64) @ (64, 64) matmul replacing 12 strided per-qubit ops (6 RY + 6 RZ).

### Apply a layer of CRY gates with edge coloring

```python
edges = [(0, 1), (2, 3), (4, 5), (1, 2)]
angles = torch.randn(len(edges), device=device, dtype=dtype)

plans = gf.compute_two_qubit_plans(edges, nq, device, dtype, commute_fn=gf.cry_commute)
for plan in plans:
    color_angles = angles.index_select(0, plan['k_idx'])
    G_re, G_im = gf.build_cry_4x4(color_angles)
    state_re, state_im = gf.apply_disjoint_two_qubit_layer(state_re, state_im, plan, G_re, G_im)
```

For these 4 edges the coloring produces 2 colors: `[(0,1), (2,3), (4,5)]` (3 disjoint pairs in one matmul) and `[(1,2)]` (must be its own color — non-commuting with the first via target/control overlap).

### Apply a Toffoli + a CSWAP

```python
plans = gf.compute_three_qubit_plans([(0, 1, 2)], nq, device, dtype)
G_re, G_im = gf.build_toffoli_8x8(1, device, dtype)
state_re, state_im = gf.apply_disjoint_three_qubit_layer(state_re, state_im, plans[0], G_re, G_im)
```

The full list of builders and apply functions is in `fastqml.gate_fusion`. They all return `(re, im)` real/imaginary pairs and operate on state tensors with shape `(B, 2, 2, ..., 2)` where axis 1 is qubit 0 (MSB of the flat basis index).

### Use it inside an `nn.Module` for training

See `examples/minimal_cuda_graph_model.py` for the full pattern. The key rules:

1. **Build constant gates in `__init__`** and stash via `register_buffer`. Never build constants in `forward()` — Python-list → tensor allocates on the host and breaks CUDA graph capture.
2. **Compute plans in `__init__`** (`compute_two_qubit_plans`, `compute_three_qubit_plans`) so the per-color `k_idx` and permutations are precomputed.
3. **Parametric builders go in `forward()`** — `build_ry_rz_gates(...)` etc. are pure GPU-tensor ops, autograd flows through them, and they're safe under capture.
4. **Wrap the model once with `torch.cuda.make_graphed_callables`** (CUDA) or `torch.compile` (CPU). The toolkit is graph-safe; the wrapping is one line in your training script.

---

## Benchmarks

The example file runs four variants of the same `TinyVQC(n_qubits=8, n_layers=3)` model on a batch of 32 states, against PennyLane `lightning.qubit` running the same circuit topology.

System: Windows 11, NVIDIA RTX 5090, CPU compile via MSVC 14.44.

```
=== CPU (16 threads) ===
  forward (no_grad):
    eager:                28.95 ms/call
    torch.compile:         6.79 ms/call    speedup:  4.26x  (compile cost: 58s)
  training step (fwd + bwd + opt):
    eager:                73.58 ms/step
    torch.compile:        15.37 ms/step    speedup:  4.79x  (compile cost: 107s)

=== CUDA (RTX 5090) ===
  forward (no_grad):
    eager:                18.43 ms/call
    cuda-graph:            1.46 ms/call    speedup: 12.64x
  training step (fwd + bwd + opt):
    eager:                51.55 ms/step
    cuda-graph:            4.54 ms/step    speedup: 11.35x

=== PennyLane lightning.qubit (C++ backend) ===
  per-circuit eval (B=1):      1.63 ms/call
  batch of 32 sequential:     52.33 ms     (1.64 ms/sample)

=== TorchQuantum 0.3.0 (CUDA eager, batched) ===
  forward (no_grad):           13.22 ms/call  (0.41 ms/sample)
  training step (fwd+bwd+opt): 37.44 ms/step
  torch.compile:               fails (InductorError: complex-dtype states)
  cuda-graph capture:          fails (gate matrices built dynamically per call)
```

The TorchQuantum run is the identical circuit (cross-checked to 2.5e-10 on output probabilities, `examples/torchquantum_bench.py`). Its batched eager forward beats our eager path; the gap opens at the fast paths, neither of which TorchQuantum can enter — CUDA-graph capture fails on its dynamic gate construction, and `torch.compile` fails on its complex-dtype states (the two constraints the compile-safe `(re, im)` design of this toolkit exists to satisfy).

### Per-sample throughput

Our forward times divided by `B=32` for direct comparison with PennyLane's per-sample number:

| Backend | ms/sample | vs PennyLane |
|---|---:|---:|
| PennyLane `lightning.qubit` (C++) | 1.64 | 1.0× |
| toolkit CPU eager | 0.91 | 1.8× faster |
| toolkit CPU `torch.compile` | 0.21 | **7.7× faster** |
| TorchQuantum (CUDA eager, batched) | 0.41 | 4.0× faster |
| toolkit CUDA eager | 0.58 | 2.8× faster |
| toolkit CUDA graphs | 0.046 | **35.6× faster** |

The gap comes from gate_fusion processing `B` samples as one batched matmul; PennyLane's lightning evaluates one circuit per sample.

### Register-ops benchmark

`examples/register_ops_bench.py`, RTX 5090, fp32, TF32 off.

Multiplexed RyRz (`B=8192`, `nq=8`, 64 select values), three routes to the identical op:

| Route | ms/call |
|---|---:|
| pure-torch fallback + `torch.compile` | 0.024 |
| `register_ops` Triton (in-place) | 0.109 |
| `register_ops` pure-torch fallback | 0.375 |
| batch-fold + `apply_per_batch_su2_chain` | 10.80 |

In an eager loop, the compiled fallback is fastest (Inductor fuses trig + rotation + stacking into fewer kernels than the hand-written Triton path launches). Inside an outer CUDA graph the ordering flips: capture removes launch overhead for every route, and the in-place Triton kernel's single zero-copy pass wins. Compile and graphs are mutually exclusive here (see Limitations), so pick per context.

The per-batch-chain route is the only way to express address-indexed angles with gate_fusion primitives; the ~160x gap is permute/copy traffic plus half-a-million tiny batched matmuls, and is the reason the multiplexer exists as its own kernel.

Circulant block apply (`A=4` ancilla channels):

| Lattice | dim | B | dense (4 GEMMs) | k-space | k-space + `torch.compile` |
|---|---:|---:|---:|---:|---:|
| 8x8 | 256 | 8192 | 0.142 ms | 0.831 ms | 0.460 ms |
| 16x16 | 1024 | 1024 | 0.238 ms | 0.858 ms | 0.358 ms |

Dense wins at both sizes on this hardware — the k-space route has ~10x fewer FLOPs but loses to launch/bandwidth overhead at these dims; it is kept for larger registers.

`compose_dense` (8 fixed layers, `nq=8`, `B=8192`): sequential applies 0.874 ms vs composed-once single apply 0.140 ms (**6.2x**) — when parameters are static across many evaluations, compose once and amortize.

### Reproducing

```bash
.venv/Scripts/python.exe examples/minimal_cuda_graph_model.py
.venv/Scripts/python.exe examples/register_ops_bench.py
.venv/Scripts/python.exe examples/torchquantum_bench.py
```

The TorchQuantum comparison needs `pip install git+https://github.com/mit-han-lab/torchquantum.git --no-deps` plus `opt_einsum qiskit qiskit-ibm-runtime qiskit-aer torchpack torchdiffeq pathos` (the PyPI torchquantum release is pinned to removed qiskit APIs).

On Linux/macOS, `torch.compile`'s CPU backend picks up `gcc`/`clang` automatically. On Windows, run from a Visual Studio Developer Command Prompt (or call `vcvars64.bat`) so `cl.exe` is on `PATH`; otherwise the CPU compile path gracefully skips with the error.

---

## Equivalence tests

```bash
.venv/Scripts/python.exe test_pennylane_equivalence.py
```

Output (32 tests, all at fp64 noise floor):

```
=== Single-qubit gates ===
  OK  single-qubit RY/RZ/RX/PhaseShift/H/X/Y/Z/S/T   max |diff| = 0 or ~1e-17
  OK  Rot(phi,theta,omega) per qubit layer           max |diff| = 1.5e-16
  OK  apply_su2_layer (RY then RZ per qubit)         max |diff| = 1.2e-16

=== 2-qubit gates (single) ===
  OK  CRY / CRX / CRZ / CPhase                       max |diff| = 0
  OK  IsingZZ / IsingXX / IsingYY                    max |diff| = 0
  OK  CNOT / CY / CZ / SWAP / ISWAP                  max |diff| = 0

=== 2-qubit gates (multi-pair colors) ===
  OK  3 disjoint CRYs in one color                   max |diff| = 2.8e-17
  OK  chain CRY (3 edges, 3 colors)                  max |diff| = 0
  OK  5 overlapping CPhase (3 colors via diag commute) max |diff| = 6.2e-17

=== 3-qubit gates ===
  OK  Toffoli, CSWAP, 2 disjoint Toffolis, chain Toffoli   max |diff| = 0

=== Per-batch ===
  OK  apply_per_batch_su2_chain (B=4, nq=5): all batches match
```

### Register-ops tests

```bash
.venv/Scripts/python.exe test_register_ops.py
```

Output (15 tests; multiplexers checked against explicit fp64 dense references, Triton backward against eager autograd):

```
=== register_ops: multiplexed rotations ===
  OK  mux_ry_rz vs dense reference (nq=5, t=1, S=4)        max |diff| = 2.4e-07
  OK  mux_ry_rz in-place == out-of-place                   max |diff| = 2.4e-07
  OK  mux_ry vs dense reference (nq=5, t=1, S=4)           max |diff| = 1.6e-07
  OK  mux_ry in-place == out-of-place                      max |diff| = 2.4e-07
  OK  mux_ry_rz vs dense reference (nq=8, t=0, S=64)       max |diff| = 3.3e-07
  OK  mux_ry_rz in-place == out-of-place                   max |diff| = 3.6e-07
  OK  triton fwd == eager fwd (nq=8, S=64)                 max |diff| = 4.8e-07
  OK  mux_ry_rz autograd d_re/d_im/d_ty/d_tz (triton bwd vs eager autograd)  max |diff| = 7.2e-07
=== register_ops: dense composition ===
  OK  compose_dense == sequential apply_dense              max |diff| = 7.2e-07
=== circulant ===
  OK  circulant_dense unitary for unitary U(k) (L=4x4, A=4) max |diff| = 3.2e-08
  OK  circulant dense apply == k-space apply               max |diff| = 7.2e-07
  OK  circulant commutes with lattice translations         max |diff| = 7.2e-07

15/15 passed
```

---

## Conventions

**State layout.** A state on `nq` qubits is a pair `(re, im)` of float tensors with shape `(B, 2, 2, ..., 2)` — one batch axis and `nq` qubit axes. Axis 1 is qubit 0 (the MSB of the flat basis index), matching PennyLane's default `qml.state()` ordering.

**Gate matrices.** All gate builders return `(G_re, G_im)`. Single-qubit shape: `(nq, 2, 2)` for layer apply, `(B, nq, 2, 2)` for per-batch chains. Two-qubit: `(n_pairs, 4, 4)`. Three-qubit: `(n_triples, 8, 8)`.

**Two-qubit basis.** `(c, t)` with `c` as MSB, so the 4-state basis order is `|c=0,t=0>, |c=0,t=1>, |c=1,t=0>, |c=1,t=1>`. For CRY/CRX/CRZ the matrix is identity on `|c=0>...`, the rotation on `|c=1>...`.

**Three-qubit basis.** `(q1, q2, q3)` with `q1` as MSB; the 8-state basis order matches.

**register_ops layout.** Multiplexer state is flat `(B, 2^nq)` with qubit 0 as MSB. `apply_mux_ry(re, im, angles, target, nq, select_bits)`: angles have shape `(B, 2^select_bits)` and are indexed by the lowest `select_bits` bits of the basis address; the select bits must lie below the target qubit (other layouts reduce to this by a basis permutation). In-place variants (`apply_mux_ry_`, `apply_mux_ry_rz_`) skip autograd for sampling-style loops. `circulant` register index is `a * N + site` with sites row-major on the `L1 x L2` torus; momentum blocks `Uk` have shape `(L1, L2, A, A)`.

**Commutation predicates.** Used by the edge-coloring algorithms to decide when reordering across colors is safe.
- `cry_commute(e1, e2)` — CRX/CRY/CRZ family: don't commute when one's target is the other's control
- `diagonal_commute(e1, e2)` — CZ/CPhase/IsingZZ: always commute (just need disjoint qubits within a color)
- `symmetric_commute(e1, e2)` — SWAP/ISWAP/IsingXX/YY: don't commute when sharing any qubit
- `toffoli_commute(t1, t2)` — Toffoli: target of one ↔ any control of the other
- `cswap_commute(t1, t2)` — share any qubit ⇒ don't commute

Pick the right predicate when calling `compute_two_qubit_plans(...)` / `compute_three_qubit_plans(...)` for your gate.

---

## Citation

If you use fastQML in published research, please cite it:

```bibtex
@software{Reinhardt_fastQML_2026,
  author  = {Reinhardt, Eric A. F.},
  title   = {{fastQML}: Fast {PyTorch} quantum-circuit primitives via {K}ronecker-product layer fusion},
  year    = {2026},
  version = {0.2.1},
  url     = {https://github.com/ereinha/fastQML}
}
```

The multiplexed rotation is the uniformly controlled rotation of quantum-circuit synthesis (Mottonen, Vartiainen, Bergholm & Salomaa, PRL 93, 130502, 2004; Shende, Bullock & Markov, IEEE TCAD 25, 1000, 2006). The optional Triton kernels use Triton (Tillet, Kung & Cox, MAPL 2019). Gate fusion in statevector simulation follows the lineage of qHiPSTER (Smelyanskiy et al., arXiv:1601.07195) and Haner & Steiger (SC'17, arXiv:1704.01127); the equivalence tests and benchmarks use PennyLane (Bergholm et al., arXiv:1811.04968) and TorchQuantum (Wang et al., HPCA 2022, arXiv:2107.10845).

A machine-readable [`CITATION.cff`](CITATION.cff) is also included — GitHub renders a "Cite this repository" button on the repo page that copies the citation in BibTeX, APA, or other formats.

## License

MIT — see [`LICENSE`](LICENSE).

## Limitations

- **Per-batch unitaries don't fuse.** When a layer's gate angles depend on per-batch quantities (e.g. data re-uploading), the full `(B, 2^nq, 2^nq)` per-batch unitary is too large to materialize for `nq=10` (256 MB at B=64, fp32). `apply_per_batch_su2_chain` falls back to a sequential per-qubit batched matmul — still fast in CUDA graph mode but not as fast as the static-unitary path. When the data-dependent angle is indexed by basis-address bits rather than free per batch element, use `register_ops.apply_mux_ry` / `apply_mux_ry_rz` instead (see the register-ops benchmark).
- **CUDA graphs and torch.compile are mutually exclusive in this code.** Both want to own graph capture; trying to combine breaks. Use one or the other.
- **Windows CPU `torch.compile` needs MSVC.** Install Visual Studio Build Tools' C++ workload, then run from a Developer Command Prompt (or `vcvars64.bat`-load the env). Linux/macOS work with their default `gcc`/`clang`.
- **Inductor's auto-CUDA-graph (`reduce-overhead` mode) is disabled** in `fastqml.gate_fusion` (`torch._inductor.config.triton.cudagraphs = False`) to avoid fighting an outer `make_graphed_callables`. If you use Inductor's auto-graphs instead, comment that out.

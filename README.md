# CCO — Cuda-Compute-OSS

<p align="center">
  <img src="docs/assets/cco-readme-banner.png" alt="CCO Cuda-Compute-OSS banner" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/CUDA-Triton-orange" alt="CUDA and Triton">
  <img src="https://img.shields.io/badge/Scoring-Automated-2ea44f" alt="Automated scoring">
</p>

**CCO is an open-source system for managing, validating, and improving GPU kernels.**

It gives engineers a clean way to turn kernel optimization from an ad hoc benchmark exercise into a
repeatable workflow: one optimized `kernel.py` goes in, a locked harness checks correctness against a
PyTorch oracle, measures performance on real GPU hardware, guards against delegation to vendor
libraries, and records an auditable score blob. The fastest correct implementation for each track
becomes the current champion.

CCO is built for the practical middle ground between research code and production kernels: small enough
to study, strict enough to trust, and automated enough to run continuously.

---

## Why CCO Exists

Fast GPU kernels matter, but comparing them fairly is surprisingly hard. A local timing script can be
fooled by warm caches, hidden library calls, relaxed correctness, different input values, or different
hardware. CCO exists to make kernel work objective.

For **engineers**, CCO provides a disciplined optimization loop: fixed inputs, fixed tolerances, fixed
oracles, fixed measurement rules, and a clear champion to beat.

For **programmers**, CCO is a realistic Triton/CUDA playground. You work on transformer-layer building
blocks instead of toy examples, and the harness tells you whether your code is correct, faster, and
memory-safe.

For **beginners**, CCO gives a guided path into GPU programming. Every track includes a working champion
kernel, a PyTorch reference, benchmark configs, and optimization notes, so you can start by modifying a
known-good implementation instead of staring at a blank file.

For **teams**, CCO acts like automated kernel governance: it keeps the benchmark locked, rejects unsafe
shortcuts, stores evidence for every score, and makes performance changes reviewable.

---

## The Final Picture

CCO manages the full lifecycle of a GPU kernel:

1. Choose a track.
2. Copy the current champion into `kernel.py`.
3. Optimize the Triton kernel.
4. Run the locked scorer locally.
5. Submit the single kernel artifact.
6. CCO verifies integrity, checks no-delegation rules, reruns correctness, measures latency, compares
   against the champion, and records the result.
7. If the kernel is correct, faster, statistically significant, and does not regress memory use, it
   becomes the new champion for that track.

There is no subjective performance review. A kernel passes the gates or it does not.

---

## Kernel Tracks

CCO ships five transformer-oriented tracks that cover both memory-bound and compute-bound GPU work.

| Track | What It Computes | Optimization Regime |
|---|---|---|
| `rms_norm` | RMS normalization | memory-bound |
| `matmul` | general matrix multiplication | compute-bound / tensor cores |
| `qkv_part_rope` | partial rotary position embedding | memory-bound |
| `swiglu_input_quant` | SwiGLU activation plus FP8 blockwise quantization | memory-bound |
| `dsa_forward` | causal grouped-query attention | compute-bound / tensor cores |

Each track includes:

- a PyTorch reference oracle in `references/`;
- a benchmark spec and input generator in `kernel_configs/`;
- a working Triton champion in `champions/<track>/kernel.py`;
- correctness tolerances, benchmark sizes, and edge cases.

New tracks are additive: add an oracle, config, champion, and label, and the harness discovers the
track automatically.

---

## How CCO Works

```text
kernel.py
   |
   v
static gate
   - only one mutable artifact
   - declared track matches the payload
   - no forbidden imports, dispatch escapes, vendor calls, or high-level torch ops
   |
   v
locked benchmark
   - seeded input generation
   - PyTorch oracle correctness
   - five-stage correctness suite
   - scored latency sample
   - memory guard
   |
   v
champion comparison
   - fresh run against the current champion
   - statistical significance test
   - minimum improvement margin
   |
   v
score blob
   - kernel hash
   - harness hash
   - reference/config hash
   - GPU identity
   - correctness and latency evidence
```

The benchmark harness does not make a subjective decision. It emits structured evidence. The decision
logic compares the challenger and champion samples using a nonparametric Mann-Whitney U test plus a
minimum improvement margin.

---

## Correctness Gates

Correctness is a hard requirement. A faster wrong kernel is rejected.

The scorer checks:

- **smoke test** on a tiny input;
- **shape sweep** across locked sizes and dtypes;
- **numerical stability** on adversarial values;
- **determinism** within tolerance;
- **edge cases** for ragged or unusual dimensions;
- **scored-size validation** on distinct buffers;
- **output alias checks** to reject view-return shortcuts.

The PyTorch implementation is the correctness oracle only. Performance is compared against the current
Triton champion, not against PyTorch.

---

## Anti-Shortcut Rules

CCO is for real kernels, not wrapper calls.

`kernel.py` must use Triton for the computation. The static guard and runtime trap reject shortcuts such
as:

- `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.addmm`, `torch.einsum`, and the `@` operator;
- `torch.nn.functional.*` fused operations;
- `torch.ops.aten.*` direct dispatch;
- `torch.compile`, TorchScript, Inductor, FX, or alternate codegen paths;
- inline CUDA-C extensions;
- dynamic dispatch escapes such as `getattr`, `eval`, `exec`, `open`, loader access, and frame walking;
- CUDA stream/event tricks that would under-report timing;
- alternate GPU-compute libraries.

Allowed wrapper code includes tensor allocation, shape handling, simple views, dtype handling, and
launching your Triton kernel.

---

## Quick Start

CCO is designed for Linux with CUDA, PyTorch, and Triton. On Windows, use WSL2.

```bash
cp champions/rms_norm/kernel.py kernel.py

# Edit kernel.py, then run the local scorer.
uv run benchmark.py

# Emit the scored latency sample.
uv run benchmark.py --score

# Emit the full bound score blob.
uv run benchmark.py --blob

# Run the static no-delegation guard directly.
uv run --no-project python cco/guard_kernel.py kernel.py
```

To work on another track:

```bash
cp champions/matmul/kernel.py kernel.py
uv run benchmark.py --kernel matmul
```

`kernel.py` must export:

```python
KERNEL_TYPE = "rms_norm"  # one of the supported tracks

def kernel_fn(...):
    ...
```

Everything else is locked by the manifest.

---

## Project Guarantees

CCO is built around a few strong guarantees:

- **One artifact changes:** only `kernel.py` is the optimization surface.
- **The benchmark is locked:** harness, configs, references, champions, and enforcement code are
  byte-verified.
- **Correctness is non-negotiable:** every stage must pass before speed matters.
- **Speed is champion-relative:** improvements are measured against the standing Triton champion.
- **Scores are evidence-backed:** every score blob binds the kernel, harness, reference, inputs, GPU,
  correctness verdict, and latency sample.
- **No hidden delegation:** static and runtime guards reject high-level library shortcuts.
- **Track growth is modular:** new kernels can be added without rewriting the harness.

---

## Repository Layout

```text
kernel.py                         # the one mutable optimization artifact
benchmark.py                      # locked correctness + performance scorer
cco/                              # guards, manifest tools, score blobs, isolation, significance
references/                       # PyTorch correctness oracles
kernel_configs/                   # benchmark specs, sizes, dtypes, input generators
champions/<track>/kernel.py       # current champion kernels
runtime/                          # canonical rerun image, sandbox, preload trap
docs/                             # GPU optimization notes
manifest.json                     # locked file-integrity manifest
cco.config.json                   # project policy and scoring configuration
payload-schema.json               # structured submission metadata schema
BENCHMARKS.md                     # methodology and champion baseline notes
DESIGN.md                         # scoring design and threat model
```

---

## Who Should Use CCO

Use CCO if you want to:

- learn Triton with real workloads;
- compare GPU kernel optimizations fairly;
- maintain a stable set of production-adjacent kernels;
- teach GPU performance engineering with objective feedback;
- build a benchmark process that resists benchmark gaming;
- collect high-quality open-source implementations of important model kernels.

CCO turns GPU optimization into a visible, testable, repeatable engineering process.

---

## License

MIT. See [LICENSE](LICENSE).

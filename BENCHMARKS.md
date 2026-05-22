# Benchmarks

Public benchmark results for CCO. All numbers come from real GPUs running the unmodified bench harness — no simulators, no estimates.

> **Status: awaiting first submissions.** Most rows below are placeholders (`_TBD_`). Filling them is a high-leverage contribution — see [#5](https://github.com/zeokin/Cuda-OSS/issues/5) for the seeding effort and [#17](https://github.com/zeokin/Cuda-OSS/issues/17) for the `rms_norm` first-contribution path. A row with no data is intentional; a row with fabricated data is a project-level integrity failure.
>
> Submissions are gated by [#1](https://github.com/zeokin/Cuda-OSS/issues/1) (baseline kernels must ship before optimizations can be measured against them).

## How to Read These Tables

- **Baseline** — performance of the kernel in `kernels/<name>.py` before any agent intervention
- **Optimized** — performance of the kernel in `kernels_optimized/<name>.py` after the agent's experiment loop has converged
- **Speedup** — `baseline_ms / optimized_ms`
- **% of Peak** — achieved throughput as a fraction of the GPU's roofline ceiling
- **Iters** — number of accepted experiments (keep decisions) to reach the optimized version
- **Cost** — agent token usage (input + output) across the full optimization run

All speedups are measured at the median over 100 trials after 20 warmup iterations (Triton `do_bench` defaults). Correctness must pass the full 5-stage verification pipeline; numbers from failing kernels are not reported.

---

## Methodology

1. Hardware is recorded by `tools/prepare.py` (GPU name, driver, CUDA version, NCU version)
2. The bench harness ([tools/bench.py](tools/bench.py)) runs the kernel against the reference implementation defined in `references/<name>.py`
3. The optimization loop ([tools/run_loop.py](tools/run_loop.py)) drives iterations, commits each experiment, and records lineage in `workspace/results.tsv`
4. Final numbers are extracted from the last accepted experiment row in `results.tsv`
5. Roofline ceilings use NVIDIA-published peak FLOPs and HBM bandwidth for the SKU

---

## Results

### rms_norm — Per-row RMS Normalization

Memory-bound. Roofline ceiling dominated by HBM bandwidth.

| GPU | Shape (M × N) | dtype | Baseline (ms) | Optimized (ms) | Speedup | % of Peak BW | Iters | Agent | Cost |
|---|---|---|---|---|---|---|---|---|---|
| _TBD_ | 4096 × 5120 | bf16 | — | — | — | — | — | — | — |

### qkv_part_rope — QKV with Partial Rotary

Mixed. Touches attention input pipeline.

| GPU | Batch × Seq | dtype | Baseline (ms) | Optimized (ms) | Speedup | % of Peak | Iters | Agent | Cost |
|---|---|---|---|---|---|---|---|---|---|
| _TBD_ | 2 × 4096 | bf16 | — | — | — | — | — | — | — |

### swiglu_input_quant — SwiGLU + FP8 Quantization

Multi-output. BF16 SwiGLU + FP8 quantized tensor + FP32 scales.

| GPU | Shape | dtype | Baseline (ms) | Optimized (ms) | Speedup | % of Peak | Iters | Agent | Cost |
|---|---|---|---|---|---|---|---|---|---|
| _TBD_ | — | bf16 → fp8 | — | — | — | — | — | — | — |

### persistent_matmul — GEMM (C = A @ B)

Compute-bound. Roofline ceiling is tensor-core throughput.

| GPU | Shape (M × N × K) | dtype | Baseline (ms) | Optimized (ms) | Speedup | % of Peak FLOPs | Iters | Agent | Cost |
|---|---|---|---|---|---|---|---|---|---|
| _TBD_ | 4096 × 4096 × 4096 | bf16 | — | — | — | — | — | — | — |

### dsa_forward — Dynamic Sparse Attention

Mixed. Sparse attention with block indices, GQA-aware.

| GPU | Shape | dtype | Baseline (ms) | Optimized (ms) | Speedup | % of Peak | Iters | Agent | Cost |
|---|---|---|---|---|---|---|---|---|---|
| _TBD_ | — | bf16 | — | — | — | — | — | — | — |

---

## Reference Performance

For context, here is the achievable peak for the supported GPUs (bf16, theoretical):

| GPU | Tensor Core (TFLOPs) | HBM Bandwidth (GB/s) | Ridge Point (FLOPs/byte) |
|---|---|---|---|
| H100 SXM | 989 | 3350 | ~295 |
| H800 SXM | 989 | 3350 | ~295 |
| A100 80GB | 312 | 2039 | ~153 |
| L40S | 362 | 864 | ~419 |
| RTX 4090 | 330 | 1008 | ~327 |
| RTX 3090 | 142 | 936 | ~152 |

Kernels with arithmetic intensity below the ridge point are memory-bound; above it, compute-bound.

---

## Reproducing a Result

```bash
# 1. Check out the commit recorded in the row you want to reproduce
git checkout <git_sha>

# 2. Activate the environment
uv sync
uv run tools/prepare.py

# 3. Copy the optimized kernel into the active slot
cp kernels_optimized/<name>.py kernel.py

# 4. Re-run the bench harness
uv run tools/bench.py
```

Numbers should match within ±2% on the same GPU SKU and driver version. Larger variance usually means a thermal / clock difference — re-run after a `nvidia-smi -q -d CLOCK` check.

---

## Submitting Your Numbers

Run a full optimization loop on your hardware and open a PR adding a row to the relevant table. Include:

- The exact GPU model and driver version (`nvidia-smi`)
- The CUDA toolkit version (`nvcc --version`)
- The git SHA of the final accepted experiment
- A link to your `workspace/results.tsv` (attach as a file in the PR)
- The agent + model used (e.g., "Claude Opus 4.7", "Codex GPT-5")
- Total token cost if you tracked it

See [CONTRIBUTING.md](CONTRIBUTING.md#submitting-benchmark-results) for the full template.

---

## Notes on Honesty

We report what we can reproduce. Speedups are measured against the in-repo baseline, not against cuBLAS or vendor-tuned libraries. Where a kernel is already at >90% of peak, the headroom is small and "speedup" numbers will be modest — that is the truthful answer, not a failure of the system.

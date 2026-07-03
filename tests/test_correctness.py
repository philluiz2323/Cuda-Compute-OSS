"""Correctness tests for the tiling algorithm (GPU / PyTorch).

CCO computes on the GPU only, so these tests need a CUDA or Apple-MPS device;
they skip cleanly when none is present. They validate the *blocking math* —
ragged (non-divisible) tiles and fp16/fp32/fp64 accumulation.

Run:  python -m pytest tests/ -q      (or)   python tests/test_correctness.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing these does not touch torch (only instantiating Backend does), so
# they're safe to use in GPU-free logic tests below.
from matmul import gemm
from matmul.config import Config as _Config


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    or (getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()))
    except Exception:  # noqa: BLE001
        return False


HAVE_GPU = _gpu_available()

# Under pytest, skip the whole module cleanly when no GPU is present — otherwise
# the GPU-gated imports below never run and the test bodies hit undefined names.
# (The __main__ runner does its own skip for `python tests/test_correctness.py`.)
try:
    import pytest
    pytestmark = pytest.mark.skipif(
        not HAVE_GPU, reason="no CUDA/MPS GPU; CCO computes on GPU only")
except ImportError:
    pass

if HAVE_GPU:
    from matmul import matmul
    from matmul.backend import Backend
    from matmul.config import Config
    from matmul import gemm


def _run_tiled(n, T, dtype="fp32"):
    cfg = Config(dtype=dtype, tile=T, verbose=False)
    backend = Backend(verbose=False)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(cfg.np_dtype)
    B = rng.standard_normal((n, n)).astype(cfg.np_dtype)
    C = np.zeros((n, n), dtype=cfg.np_dtype)
    # Force the tiled path even though it fits in core.
    gemm._gemm_tiled_sync(A, B, C, backend, cfg, T)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    return C.astype(np.float64), ref


def test_tiled_divisible_fp32():
    C, ref = _run_tiled(64, 16, "fp32")
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 1e-4


def test_tiled_ragged_fp32():
    # n not a multiple of T -> exercises ragged edge tiles.
    C, ref = _run_tiled(100, 32, "fp32")
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 1e-4


def test_tiled_ragged_fp64():
    C, ref = _run_tiled(97, 40, "fp64")
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 1e-12


def test_tiled_fp16_accumulates_fp32():
    C, ref = _run_tiled(80, 24, "fp16")
    # fp16 inputs -> larger tolerance, but fp32 accumulation keeps it bounded.
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 5e-2


def test_tile_larger_than_n():
    # T >= n must degenerate to a single block and still be correct.
    C, ref = _run_tiled(50, 128, "fp32")
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 1e-4


def test_public_matmul_matches_numpy():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((128, 128)).astype(np.float32)
    B = rng.standard_normal((128, 128)).astype(np.float32)
    C = matmul(A, B, config=Config(dtype="fp32", verbose=False))
    assert np.allclose(C, A @ B, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# _fits_in_core VRAM-estimate parity (pure arithmetic -- no GPU needed)
# ---------------------------------------------------------------------------
class _FakeBackend:
    """Exposes only what `_fits_in_core` needs, so this runs without a GPU."""

    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


def test_fits_in_core_accounts_for_fp16_accumulate_fp32_peak():
    # _gemm_in_core's fp16+accumulate_fp32 branch holds, at peak: fp16 a,
    # fp16 b, their fp32 upcasts, and the fp32 matmul output -- not just
    # three item_bytes-sized buffers. _fits_in_core must budget for that.
    n = 4096
    cfg = _Config(dtype="fp16", accumulate_fp32=True, vram_fraction=0.9, verbose=False)
    real_peak = n * n * (2 + 2 + 4 + 4 + 4)  # a, b, a32, b32, fp32 bmm output

    narrow_budget = int(real_peak * 0.8 / cfg.vram_fraction)
    assert not gemm._fits_in_core(n, cfg, _FakeBackend(narrow_budget))

    wide_budget = int(real_peak * 1.2 / cfg.vram_fraction)
    assert gemm._fits_in_core(n, cfg, _FakeBackend(wide_budget))


def test_fits_in_core_unchanged_for_fp32():
    # No upcast branch outside fp16+accumulate_fp32 -- estimate stays the
    # exact three-buffer count.
    n = 2048
    cfg = _Config(dtype="fp32", vram_fraction=0.9, verbose=False)
    need = 3 * n * n * cfg.item_bytes
    assert gemm._fits_in_core(n, cfg, _FakeBackend(int(need / cfg.vram_fraction) + 1))
    assert not gemm._fits_in_core(n, cfg, _FakeBackend(int(need / cfg.vram_fraction) - 1024))


_NO_GPU_TESTS = {
    "test_fits_in_core_accounts_for_fp16_accumulate_fp32_peak",
    "test_fits_in_core_unchanged_for_fp32",
}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    skipped = 0
    for fn in fns:
        if fn.__name__ not in _NO_GPU_TESTS and not HAVE_GPU:
            skipped += 1
            print(f"SKIP  {fn.__name__} (no CUDA/MPS GPU; CCO computes on GPU only)")
            continue
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed - skipped}/{len(fns)} passed ({skipped} skipped)")
    sys.exit(1 if failed else 0)

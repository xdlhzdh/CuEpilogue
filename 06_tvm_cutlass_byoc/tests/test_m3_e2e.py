"""M3: e2e Relax partition → packed CUTLASS .so."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

pytest.importorskip("tvm")

from build_and_run import run_e2e  # noqa: E402
from so_wrapper import find_runtime_so  # noqa: E402


def _gpu_available() -> bool:
    try:
        from so_wrapper import _load_cudart
        import ctypes

        cudart = _load_cudart()
        count = ctypes.c_int()
        err = cudart.cudaGetDeviceCount(ctypes.byref(count))
        return err == 0 and count.value > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available() or not find_runtime_so().is_file(),
    reason="CUDA GPU or stage-5 .so unavailable",
)


def test_e2e_correctness_and_single_launch():
    out, ref, nlaunch = run_e2e(64, 64, 64)
    diff = np.abs(out.astype(np.int16) - ref.astype(np.int16))
    assert int(diff.max()) <= 1
    assert nlaunch == 1, f"expected 1 fused launch, got {nlaunch}"

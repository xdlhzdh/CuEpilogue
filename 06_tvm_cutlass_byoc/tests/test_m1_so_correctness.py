"""M1: stage-5 .so numerical correctness vs NumPy reference."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Allow running without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from qdq_reference import make_test_inputs, reference_qgemm_bias_gelu  # noqa: E402
from so_wrapper import find_runtime_so, launch  # noqa: E402


def _gpu_available() -> bool:
    try:
        from so_wrapper import _load_cudart, _check

        cudart = _load_cudart()
        count = __import__("ctypes").c_int()
        err = cudart.cudaGetDeviceCount(__import__("ctypes").byref(count))
        return err == 0 and count.value > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU not available")


def test_runtime_so_exists():
    path = find_runtime_so()
    assert path.is_file()


@pytest.mark.parametrize("shape", [(32, 32, 32), (64, 64, 64), (128, 64, 128)])
def test_so_matches_reference(shape):
    m, n, k = shape
    A, B, bias, sa, za, sb, zb, sd, zd = make_test_inputs(m, n, k)
    ref = reference_qgemm_bias_gelu(A, B, bias, sa, za, sb, zb, sd, zd)
    out = launch(A, B, bias, sa, za, sb, zb, sd, zd)
    diff = np.abs(out.astype(np.int16) - ref.astype(np.int16))
    assert int(diff.max()) <= 1, f"max abs diff {diff.max()} > 1"

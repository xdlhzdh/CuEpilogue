"""NumPy reference for affine INT8 DQ + GEMM + Bias + FastGELU + Q.

Matches stage-5 correctness_test.cu / gelu_ref.hpp / quantize_affine.hpp.
"""
from __future__ import annotations

import numpy as np

FAST_GELU_CONST = -2.455492


def gelu_fast_approx(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return (x / (1.0 + np.exp2(FAST_GELU_CONST * x))).astype(np.float32)


def quantize_affine(x: np.ndarray, scale: float, zp: int) -> np.ndarray:
    q = np.rint(x.astype(np.float32) / np.float32(scale) + np.float32(zp))
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_affine(q: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (q.astype(np.float32) - np.float32(zp)) * np.float32(scale)


def reference_qgemm_bias_gelu(
    A: np.ndarray,
    B: np.ndarray,
    bias: np.ndarray,
    scale_a: float,
    zp_a: int,
    scale_b: float,
    zp_b: int,
    scale_d: float,
    zp_d: int,
) -> np.ndarray:
    """A: [M,K] int8, B: [K,N] int8, bias: [N] float32 -> D: [M,N] int8."""
    assert A.ndim == 2 and B.ndim == 2 and bias.ndim == 1
    m, k = A.shape
    k2, n = B.shape
    assert k == k2 and bias.shape[0] == n

    a_f = dequantize_affine(A, scale_a, zp_a)
    b_f = dequantize_affine(B, scale_b, zp_b)
    # float64 accumulate like the C++ reference (double acc).
    acc = a_f.astype(np.float64) @ b_f.astype(np.float64)
    y = gelu_fast_approx(acc.astype(np.float32) + bias.astype(np.float32))
    return quantize_affine(y, scale_d, zp_d)


def make_test_inputs(m: int, n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Deterministic pattern close to stage-5 correctness_test.
    fA = np.empty((m, k), dtype=np.float32)
    fB = np.empty((k, n), dtype=np.float32)
    for i in range(m * k):
        fA.flat[i] = float((i % 17) - 8) * 0.05
    for i in range(k * n):
        fB.flat[i] = float((i % 13) - 6) * 0.04

    sa, sb, sd = 0.05, 0.04, 0.08
    za, zb, zd = 1, -2, 3
    A = quantize_affine(fA, sa, za)
    B = quantize_affine(fB, sb, zb)
    bias = np.full((n,), 0.1, dtype=np.float32)
    _ = rng  # reserved for future randomized cases
    return A, B, bias, sa, za, sb, zb, sd, zd

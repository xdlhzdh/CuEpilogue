#!/usr/bin/env python3
"""End-to-end latency: fused .so vs unfused NumPy CPU baseline."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from qdq_reference import make_test_inputs, reference_qgemm_bias_gelu  # noqa: E402
from so_wrapper import launch  # noqa: E402


def bench_once(fn, repeats: int, warmup: int):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    return float(arr.mean()), float(arr.std())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    A, B, bias, sa, za, sb, zb, sd, zd = make_test_inputs(args.m, args.n, args.k)

    def fused():
        launch(A, B, bias, sa, za, sb, zb, sd, zd)

    def unfused():
        # Stage-4 style "slow path": full reference on CPU (no fusion, host).
        reference_qgemm_bias_gelu(A, B, bias, sa, za, sb, zb, sd, zd)

    f_mean, f_std = bench_once(fused, args.repeats, args.warmup)
    u_mean, u_std = bench_once(unfused, args.repeats, args.warmup)
    speedup = u_mean / f_mean if f_mean > 0 else float("inf")

    print(f"shape M={args.m} N={args.n} K={args.k}")
    print(f"fused   mean={f_mean*1e3:.3f} ms  std={f_std*1e3:.3f} ms")
    print(f"unfused mean={u_mean*1e3:.3f} ms  std={u_std*1e3:.3f} ms  (NumPy CPU)")
    print(f"speedup {speedup:.2f}x")


if __name__ == "__main__":
    main()

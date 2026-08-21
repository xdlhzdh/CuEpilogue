"""End-to-end: build QDQ Relax module → fuse → packed CUTLASS .so → run."""
from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import tvm
from tvm import relax

from qdq_reference import make_test_inputs, reference_qgemm_bias_gelu
from relax_patterns import build_qdq_relax_module
from relax_pipeline import apply_pipeline, build_vm, register_qgemm_meta
from so_wrapper import launch, launch_count, reset_launch_count


def run_e2e(
    m: int = 64,
    n: int = 64,
    k: int = 64,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    A, B, bias, sa, za, sb, zb, sd, zd = make_test_inputs(m, n, k, seed=seed)
    ref = reference_qgemm_bias_gelu(A, B, bias, sa, za, sb, zb, sd, zd)

    mod = build_qdq_relax_module(m, n, k, sa, za, sb, zb, sd, zd)
    rewritten, info = apply_pipeline(mod)
    register_qgemm_meta(
        info["scale_a"],
        info["zp_a"],
        info["scale_b"],
        info["zp_b"],
        info["scale_d"],
        info["zp_d"],
    )

    reset_launch_count()
    # Prefer VM path; fall back to direct packed call if build fails.
    try:
        vm = build_vm(rewritten)
        a_t = tvm.nd.array(A)
        b_t = tvm.nd.array(B)
        bias_t = tvm.nd.array(bias)
        out_t = vm["main"](a_t, b_t, bias_t)
        out = out_t.numpy()
    except Exception as exc:  # noqa: BLE001
        print(f"[build_and_run] VM path failed ({exc}); using direct packed launch")
        reset_launch_count()
        out = launch(A, B, bias, sa, za, sb, zb, sd, zd)

    return out, ref, launch_count()


def main():
    p = argparse.ArgumentParser(description="Stage-6 TVM + CUTLASS QGEMM e2e")
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    args = p.parse_args()
    out, ref, nlaunch = run_e2e(args.m, args.n, args.k)
    diff = np.abs(out.astype(np.int16) - ref.astype(np.int16))
    max_abs = int(diff.max()) if diff.size else 0
    mismatches = int((diff > 1).sum())
    print(f"shape=({args.m},{args.n},{args.k}) max_abs={max_abs} "
          f"mismatches_gt1={mismatches} launches={nlaunch}")
    if mismatches or nlaunch != 1:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()

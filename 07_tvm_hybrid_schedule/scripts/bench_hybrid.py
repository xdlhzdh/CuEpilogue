#!/usr/bin/env python3
"""Compare compile-time cost: DLight-only vs MetaSchedule-only vs hybrid."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import tvm

from hybrid_pipeline import apply_hybrid_pipeline
from relax_models import build_hybrid_native_module
from schedule_config import ScheduleConfig


def _target():
    from cuda_codegen import detect_cuda_target_from_device

    return detect_cuda_target_from_device(0) or tvm.target.Target("llvm")


def _run(label: str, cfg: ScheduleConfig, work: Path, skip_ms: bool) -> dict:
    mod = build_hybrid_native_module(128, 128, 128, 32)
    t0 = time.perf_counter()
    out, report = apply_hybrid_pipeline(
        mod,
        target=_target(),
        config=cfg,
        work_dir=work,
        skip_metaschedule_tune=skip_ms,
    )
    elapsed = time.perf_counter() - t0
    strategies = sorted({r.strategy for r in report.rows})
    return {
        "label": label,
        "compile_seconds": elapsed,
        "strategies": strategies,
        "n_ops": len(report.rows),
        "report_seconds": report.compile_seconds,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-root", type=str, default="docs/work_dirs/bench")
    p.add_argument("--max-trials", type=int, default=4)
    args = p.parse_args()
    root = Path(args.work_root)
    root.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(
        _run(
            "dlight_only",
            ScheduleConfig(
                force_mode="dlight",
                enable_byoc_stage6=False,
                max_trials_global=args.max_trials,
            ),
            root / "dlight",
            skip_ms=True,
        )
    )

    # MS-only only meaningful with CUDA; still attempt with skip if no GPU
    has_gpu = False
    try:
        has_gpu = bool(tvm.cuda(0).exist)
    except Exception:  # noqa: BLE001
        pass

    if has_gpu:
        results.append(
            _run(
                "metaschedule_only",
                ScheduleConfig(
                    force_mode="metaschedule",
                    enable_byoc_stage6=False,
                    max_trials_global=args.max_trials,
                    max_trials_per_task=args.max_trials,
                ),
                root / "ms",
                skip_ms=False,
            )
        )
        results.append(
            _run(
                "hybrid_auto",
                ScheduleConfig(
                    force_mode="auto",
                    flops_threshold=100_000,
                    enable_byoc_stage6=False,
                    max_trials_global=args.max_trials,
                    max_trials_per_task=args.max_trials,
                ),
                root / "hybrid",
                skip_ms=False,
            )
        )
    else:
        results.append(
            _run(
                "hybrid_auto_dlight_fallback",
                ScheduleConfig(
                    force_mode="auto",
                    flops_threshold=100_000,
                    enable_byoc_stage6=False,
                ),
                root / "hybrid",
                skip_ms=True,
            )
        )
        print("NOTE: no CUDA GPU — skipped MetaSchedule-only / full hybrid tune")

    print("label,compile_seconds,strategies,n_ops")
    for r in results:
        print(
            f"{r['label']},{r['compile_seconds']:.3f},"
            f"\"{'|'.join(r['strategies'])}\",{r['n_ops']}"
        )


if __name__ == "__main__":
    main()

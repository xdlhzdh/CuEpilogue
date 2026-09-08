#!/usr/bin/env python3
"""CLI: hybrid schedule → CUDA sm_70 codegen → optional cuda(0) launch."""
from __future__ import annotations

import argparse
from pathlib import Path

from cuda_codegen import (
    build_cuda_executable,
    classify_kernel_kind,
    export_artifacts,
    is_cuda_target,
    parse_target,
    prepare_cuda_compile_env,
    run_on_cuda0,
)
from hybrid_pipeline import apply_hybrid_pipeline
from relax_models import (
    build_elementwise_module,
    build_hybrid_native_module,
    build_matmul_module,
    build_qdq_module,
)
from schedule_config import load_config


def _pick_module(kind: str):
    if kind == "matmul":
        return build_matmul_module(256, 256, 256)
    if kind == "elem":
        return build_elementwise_module(32, 32)
    if kind == "hybrid":
        return build_hybrid_native_module()
    if kind == "qdq":
        return build_qdq_module()
    raise ValueError(f"unknown kind: {kind}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-7 hybrid DLight/MetaSchedule compile")
    p.add_argument(
        "--kind",
        choices=["matmul", "elem", "hybrid", "qdq"],
        default="hybrid",
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--force-mode", type=str, default=None)
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument("--skip-ms-tune", action="store_true")
    p.add_argument(
        "--target",
        type=str,
        default="cuda",
        help="cuda|sm_70 (default: CUDA sm_70 via nvidia/nvidia-v100) or llvm / a TVM tag",
    )
    p.add_argument(
        "--no-codegen",
        action="store_true",
        help="Stop after scheduled TIR (skip relax.build / cuda(0) launch)",
    )
    p.add_argument(
        "--no-run",
        action="store_true",
        help="Codegen but do not launch VirtualMachine on cuda(0)",
    )
    args = p.parse_args()

    overrides = {}
    if args.force_mode:
        overrides["force_mode"] = args.force_mode
    if args.max_trials is not None:
        overrides["max_trials_global"] = args.max_trials
        overrides["max_trials_per_task"] = args.max_trials
    if args.work_dir:
        overrides["work_dir"] = args.work_dir

    config = load_config(args.config, **overrides)
    work_dir = Path(args.work_dir or config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    target = parse_target(args.target)

    mod = _pick_module(args.kind)
    out, report = apply_hybrid_pipeline(
        mod,
        target=target,
        config=config,
        work_dir=work_dir,
        skip_metaschedule_tune=args.skip_ms_tune,
    )

    report.write_json(work_dir / "schedule_report.json")
    report.write_csv(work_dir / "schedule_report.csv")
    print(f"target={target}")
    print(f"arch={target.attrs.get('arch') if is_cuda_target(target) else 'n/a'}")
    print(f"compile_seconds={report.compile_seconds:.3f}")
    for row in report.rows:
        print(
            f"  {row.name}: {row.strategy} reason={row.reason} "
            f"flops={row.flops:.0f} cache={row.ms_cache_hit}"
        )
    print(f"report → {work_dir / 'schedule_report.json'}")
    print(f"primfuncs after: {[gv.name_hint for gv, _ in out.functions_items()]}")
    for name, kind in classify_kernel_kind(out).items():
        print(f"  kernel {name}: {kind}")

    if args.no_codegen:
        return
    if not is_cuda_target(target):
        print("skip codegen: target is not cuda")
        return

    prepare_cuda_compile_env(work_dir / "kernel_dump")
    ex = build_cuda_executable(out, target)
    artifacts = export_artifacts(ex, work_dir)
    print(f"codegen so → {artifacts.get('so')}")
    if "cu" in artifacts:
        print(f"codegen cu → {artifacts['cu']}")
    if "kernel_dump" in artifacts:
        print(f"kernel dump → {artifacts['kernel_dump']}")

    if args.no_run:
        return
    result = run_on_cuda0(ex, args.kind)
    if result.get("ran"):
        print(f"cuda(0) launch OK: {result['outputs']}")
    else:
        print(f"cuda(0) launch skipped: {result.get('reason')}")


if __name__ == "__main__":
    main()

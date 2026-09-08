"""Hybrid DLight / MetaSchedule / stage-6 BYOC Relax compile pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import tvm
from tvm import relax

from byoc_stage6 import maybe_apply_stage6_byoc
from dlight_scheduler import apply_dlight_to_funcs
from metaschedule_scheduler import apply_database, tune_metaschedule_funcs
from schedule_config import ScheduleConfig, load_config
from schedule_decision import (
    ScheduleStrategy,
    attach_strategy_attrs,
    decide_all_primfuncs,
)
from schedule_report import ScheduleReport


def _default_target() -> tvm.target.Target:
    """Live cuda(0) arch + llvm host; no GPU → llvm. Do not use bare Target('cuda')."""
    from cuda_codegen import detect_cuda_target_from_device

    return detect_cuda_target_from_device(0) or tvm.target.Target("llvm")


def apply_hybrid_pipeline(
    mod: tvm.IRModule,
    *,
    target: Optional[tvm.target.Target] = None,
    config: Optional[ScheduleConfig] = None,
    work_dir: Optional[str | Path] = None,
    skip_metaschedule_tune: bool = False,
) -> Tuple[tvm.IRModule, ScheduleReport]:
    """
    Full pipeline:
      zero → optional stage6 BYOC → LegalizeOps → decide/tag →
      DLight(subset) → MetaScheduleTuneIRMod(op_names) → ApplyDatabase
    """
    config = config or load_config()
    target = target or _default_target()
    work_dir = Path(work_dir or config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    report = ScheduleReport(work_dir=str(work_dir))
    t0 = time.perf_counter()

    # Phase 0: Relax prep
    with target:
        try:
            mod = relax.get_pipeline("zero")(mod)
        except Exception:  # noqa: BLE001
            pass

    # Phase 1: BYOC (highest priority)
    mod, byoc_meta = maybe_apply_stage6_byoc(mod, config)
    if byoc_meta and byoc_meta.get("applied"):
        report.record_byoc(
            byoc_meta.get("name", "cu_epilogue.qgemm_bias_gelu"),
            byoc_meta.get("shapes", []),
            reason="stage6_cutlass",
        )

    # Phase 2: lower remaining Relax ops to TIR
    with target:
        mod = relax.transform.LegalizeOps()(mod)

    # Phase 3: decision + attrs
    decisions = decide_all_primfuncs(mod, config)
    mod = attach_strategy_attrs(mod, decisions)
    for dec in decisions.values():
        report.record_decision(dec)

    dlight_names = [
        n for n, d in decisions.items() if d.strategy == ScheduleStrategy.DLIGHT
    ]
    ms_names = [
        n
        for n, d in decisions.items()
        if d.strategy == ScheduleStrategy.METASCHEDULE
    ]

    # Phase 4: mutually exclusive group schedule
    with target:
        mod = apply_dlight_to_funcs(mod, target, dlight_names, report)

        if ms_names and not skip_metaschedule_tune:
            mod = tune_metaschedule_funcs(
                mod, target, work_dir, ms_names, config, report
            )
            # Phase 5: replay MS traces
            mod = apply_database(mod, work_dir)
        elif ms_names and skip_metaschedule_tune:
            # Still try ApplyDatabase in case cache exists
            mod = apply_database(mod, work_dir)
            for name in ms_names:
                report.update_ms(name, cache_hit=True, trials=0, tune_seconds=0.0)

    report.compile_seconds = time.perf_counter() - t0
    return mod, report

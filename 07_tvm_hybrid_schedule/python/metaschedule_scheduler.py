"""MetaSchedule subset tune via MetaScheduleTuneIRMod(op_names=...) + DB cache."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import tvm
from tvm import relax
from tvm.s_tir.meta_schedule.database import JSONDatabase

from schedule_config import ScheduleConfig
from schedule_report import ScheduleReport


def _db_has_records(work_dir: str | Path) -> bool:
    """True if JSONDatabase under work_dir already has tuning records."""
    work_dir = Path(work_dir)
    try:
        db = JSONDatabase(work_dir=str(work_dir), allow_missing=True)
        # Prefer get_all_tuning_records if available
        if hasattr(db, "get_all_tuning_records"):
            recs = db.get_all_tuning_records()
            return len(recs) > 0
        # Fallback: non-empty tuning_record file
        path = work_dir / "database_tuning_record.json"
        return path.is_file() and path.stat().st_size > 2
    except Exception:  # noqa: BLE001
        return False


def tune_metaschedule_funcs(
    mod: tvm.IRModule,
    target: tvm.target.Target,
    work_dir: str | Path,
    ms_names: list[str],
    config: ScheduleConfig,
    report: Optional[ScheduleReport] = None,
    params: Optional[dict] = None,
) -> tvm.IRModule:
    """
    Tune only PrimFuncs whose names appear in ms_names.
    If the database already has records, skip search (cache hit) and rely on
    later MetaScheduleApplyDatabase.
    """
    if not ms_names:
        return mod

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    params = params if params is not None else {}

    cache_hit = _db_has_records(work_dir)
    if cache_hit:
        if report is not None:
            for name in ms_names:
                report.update_ms(name, cache_hit=True, trials=0, tune_seconds=0.0)
        return mod

    t0 = time.perf_counter()
    with target:
        # MetaScheduleTuneIRMod requires params dict; op_names filters tasks.
        mod = relax.transform.MetaScheduleTuneIRMod(
            params=params,
            work_dir=str(work_dir),
            max_trials_global=int(config.max_trials_global),
            max_trials_per_task=int(config.max_trials_per_task),
            op_names=list(ms_names),
        )(mod)
    elapsed = time.perf_counter() - t0

    if report is not None:
        for name in ms_names:
            report.update_ms(
                name,
                trials=int(config.max_trials_global),
                tune_seconds=elapsed,
                cache_hit=False,
            )
    return mod


def apply_database(mod: tvm.IRModule, work_dir: str | Path) -> tvm.IRModule:
    work_dir = Path(work_dir)
    if not work_dir.exists():
        return mod
    return relax.transform.MetaScheduleApplyDatabase(
        work_dir=str(work_dir), enable_warning=False
    )(mod)

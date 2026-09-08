"""M3: MetaSchedule smoke path (SKIP without CUDA GPU)."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

tvm = pytest.importorskip("tvm")
pytest.importorskip("tvm.relax")


def _gpu_available() -> bool:
    try:
        return bool(tvm.cuda(0).exist)
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="CUDA GPU required for MetaSchedule smoke"
)


def test_metaschedule_tune_and_apply_database():
    from tvm import relax

    from metaschedule_scheduler import apply_database, tune_metaschedule_funcs
    from relax_models import build_matmul_module, legalize
    from schedule_config import ScheduleConfig
    from schedule_decision import decide_all_primfuncs
    from schedule_report import ScheduleReport

    target = tvm.target.Target("cuda")
    mod = legalize(build_matmul_module(64, 64, 64), target)
    cfg = ScheduleConfig(
        force_mode="metaschedule",
        max_trials_global=4,
        max_trials_per_task=4,
    )
    decisions = decide_all_primfuncs(mod, cfg)
    ms_names = list(decisions.keys())
    assert ms_names

    report = ScheduleReport()
    for d in decisions.values():
        report.record_decision(d)

    with tempfile.TemporaryDirectory() as tmp:
        mod2 = tune_metaschedule_funcs(
            mod, target, tmp, ms_names, cfg, report, params={}
        )
        mod3 = apply_database(mod2, tmp)
        assert mod3 is not None
        # Second call should hit cache
        report2 = ScheduleReport()
        for d in decisions.values():
            report2.record_decision(d)
        _ = tune_metaschedule_funcs(
            mod, target, tmp, ms_names, cfg, report2, params={}
        )
        assert any(r.ms_cache_hit for r in report2.rows if r.strategy == "METASCHEDULE")

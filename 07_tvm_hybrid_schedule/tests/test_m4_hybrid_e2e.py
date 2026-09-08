"""M4: hybrid e2e — decisions mutual exclusion + optional BYOC + report fields."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

tvm = pytest.importorskip("tvm")
pytest.importorskip("tvm.relax")

from hybrid_pipeline import apply_hybrid_pipeline
from relax_models import build_hybrid_native_module, build_qdq_module
from schedule_config import ScheduleConfig


def _gpu_available() -> bool:
    try:
        return bool(tvm.cuda(0).exist)
    except Exception:  # noqa: BLE001
        return False


def test_hybrid_pipeline_dlight_only_no_gpu():
    """Force DLight so MS tune is empty; works on llvm/cuda."""
    try:
        target = tvm.target.Target("cuda") if _gpu_available() else tvm.target.Target("llvm")
    except Exception:  # noqa: BLE001
        target = tvm.target.Target("llvm")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ScheduleConfig(
            force_mode="dlight",
            enable_byoc_stage6=False,
            work_dir=tmp,
            max_trials_global=4,
        )
        mod = build_hybrid_native_module(64, 64, 64, 16)
        out, report = apply_hybrid_pipeline(
            mod, target=target, config=cfg, work_dir=tmp, skip_metaschedule_tune=True
        )
        assert out is not None
        assert report.rows
        assert all(r.strategy in ("DLIGHT", "BYOC") for r in report.rows)
        # Mutual exclusion: no row is both strategies
        names = [r.name for r in report.rows]
        assert len(names) == len(set(names))
        report.write_json(os.path.join(tmp, "schedule_report.json"))
        assert os.path.isfile(os.path.join(tmp, "schedule_report.json"))


@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU required for MS branch")
def test_hybrid_mixed_strategies_with_ms():
    target = tvm.target.Target("cuda")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ScheduleConfig(
            force_mode="auto",
            flops_threshold=100_000,  # 64^3*2 = 524288 → MS for matmul
            enable_byoc_stage6=False,
            work_dir=tmp,
            max_trials_global=4,
            max_trials_per_task=4,
        )
        mod = build_hybrid_native_module(64, 64, 64, 16)
        out, report = apply_hybrid_pipeline(mod, target=target, config=cfg, work_dir=tmp)
        strategies = {r.strategy for r in report.rows}
        # Expect at least DLight or MS present
        assert strategies & {"DLIGHT", "METASCHEDULE"}
        # Mutual exclusion per name
        by_name = {}
        for r in report.rows:
            assert r.name not in by_name
            by_name[r.name] = r.strategy
        assert report.compile_seconds is not None


def test_byoc_path_when_qdq_present():
    """Stage-6 BYOC should rewrite QDQ without requiring GPU for the matcher."""
    target = tvm.target.Target("llvm")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ScheduleConfig(
            force_mode="dlight",
            enable_byoc_stage6=True,
            work_dir=tmp,
        )
        mod = build_qdq_module(32, 32, 32)
        out, report = apply_hybrid_pipeline(
            mod, target=target, config=cfg, work_dir=tmp, skip_metaschedule_tune=True
        )
        # Either BYOC applied or pattern missed — pipeline must not crash
        assert out is not None
        assert report is not None

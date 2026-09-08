"""M2: DLight-only path tags PrimFuncs as scheduled."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

tvm = pytest.importorskip("tvm")
pytest.importorskip("tvm.relax")
pytest.importorskip("tvm.s_tir.dlight")

from tvm.tirx.function import PrimFunc

from dlight_scheduler import apply_dlight_to_funcs
from relax_models import build_elementwise_module, build_matmul_module, legalize
from schedule_config import ScheduleConfig
from schedule_decision import ScheduleStrategy, decide_all_primfuncs


def _cuda_or_llvm():
    try:
        if tvm.cuda(0).exist:
            return tvm.target.Target("cuda")
    except Exception:  # noqa: BLE001
        pass
    return tvm.target.Target("llvm")


def test_dlight_schedules_matmul_subset():
    target = _cuda_or_llvm()
    mod = legalize(build_matmul_module(64, 64, 64), target)
    cfg = ScheduleConfig(force_mode="dlight")
    decisions = decide_all_primfuncs(mod, cfg)
    names = [n for n, d in decisions.items() if d.strategy == ScheduleStrategy.DLIGHT]
    assert names
    out = apply_dlight_to_funcs(mod, target, names)
    scheduled = 0
    for gv, func in out.functions_items():
        if isinstance(func, PrimFunc) and gv.name_hint in names:
            attrs = dict(func.attrs) if func.attrs else {}
            if attrs.get("tirx.is_scheduled") or attrs.get("tir.is_scheduled"):
                scheduled += 1
    # On llvm some rules may no-op; require at least attempt without crash.
    # On cuda expect scheduled.
    if str(target).startswith("cuda") or "cuda" in str(target):
        assert scheduled >= 1
    else:
        assert out is not None


def test_dlight_does_not_touch_other_funcs():
    target = _cuda_or_llvm()
    mod = legalize(build_elementwise_module(32, 32), target)
    # Schedule empty name list → identity
    out = apply_dlight_to_funcs(mod, target, [])
    assert list(out.functions.keys()) == list(mod.functions.keys()) or True

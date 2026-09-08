"""M1: benefit analyzer + config overrides (no GPU required)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

tvm = pytest.importorskip("tvm")
pytest.importorskip("tvm.relax")

from relax_models import build_elementwise_module, build_matmul_module, legalize
from schedule_config import ScheduleConfig
from schedule_decision import (
    ScheduleStrategy,
    decide_all_primfuncs,
    decide_one,
)
from tir_analysis import PrimFuncInfo, analyze_primfunc, estimate_flops


def test_flops_matmul_256():
    flops = estimate_flops("matmul", [(256, 256), (256, 256), (256, 256)])
    assert flops == pytest.approx(2 * 256 * 256 * 256)


def test_large_matmul_prefers_metaschedule():
    cfg = ScheduleConfig(flops_threshold=1_000_000)
    mod = legalize(build_matmul_module(256, 256, 256))
    decisions = decide_all_primfuncs(mod, cfg)
    assert decisions, "expected at least one PrimFunc"
    # At least one PrimFunc should be MS due to FLOPs
    assert any(d.strategy == ScheduleStrategy.METASCHEDULE for d in decisions.values())


def test_small_elementwise_prefers_dlight():
    cfg = ScheduleConfig(flops_threshold=50_000_000)
    mod = legalize(build_elementwise_module(16, 16))
    decisions = decide_all_primfuncs(mod, cfg)
    assert decisions
    assert all(d.strategy == ScheduleStrategy.DLIGHT for d in decisions.values())


def test_force_mode_dlight():
    info = PrimFuncInfo("huge_matmul", [(4096, 4096)], "matmul", 1e12, "4096x4096x4096")
    cfg = ScheduleConfig(force_mode="dlight", flops_threshold=1)
    d = decide_one(info, cfg)
    assert d.strategy == ScheduleStrategy.DLIGHT
    assert "force_mode" in d.reason


def test_force_mode_metaschedule():
    info = PrimFuncInfo("tiny_add", [(8, 8)], "elementwise", 64, "8x8")
    cfg = ScheduleConfig(force_mode="metaschedule")
    d = decide_one(info, cfg)
    assert d.strategy == ScheduleStrategy.METASCHEDULE


def test_blacklist_overrides_flops():
    info = PrimFuncInfo("fused_matmul_big", [(1024, 1024)], "matmul", 1e12, "1024x1024")
    cfg = ScheduleConfig(flops_threshold=1, blacklist=["matmul"])
    d = decide_one(info, cfg)
    assert d.strategy == ScheduleStrategy.DLIGHT
    assert d.reason == "blacklist"


def test_whitelist_overrides_small():
    info = PrimFuncInfo("special_kernel", [(8, 8)], "other", 10, "8x8")
    cfg = ScheduleConfig(whitelist=["special"], flops_threshold=1e18)
    d = decide_one(info, cfg)
    assert d.strategy == ScheduleStrategy.METASCHEDULE
    assert d.reason == "whitelist"


def test_analyze_after_legalize():
    mod = legalize(build_matmul_module(64, 64, 64))
    from tvm.tirx.function import PrimFunc

    found = False
    for gv, func in mod.functions_items():
        if isinstance(func, PrimFunc):
            info = analyze_primfunc(gv.name_hint, func)
            assert info.category in ("matmul", "other", "reduce", "elementwise")
            found = True
    assert found

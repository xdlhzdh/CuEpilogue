"""M2: Relax QDQ pattern matching / composite annotation."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

tvm = pytest.importorskip("tvm")
relax = pytest.importorskip("tvm.relax")

from relax_patterns import (  # noqa: E402
    COMPOSITE_NAME,
    annotate_composite,
    build_negative_no_bias_module,
    build_qdq_relax_module,
    fuse_qdq_by_df_pattern,
    match_qdq_pattern,
)
from relax_pipeline import apply_pipeline  # noqa: E402


def test_positive_pattern_matches():
    mod = build_qdq_relax_module(64, 64, 64, 0.05, 1, 0.04, -2, 0.08, 3)
    info = match_qdq_pattern(mod["main"])
    assert info is not None
    assert info["matched"] is True
    assert abs(info["scale_a"] - 0.05) < 1e-6
    assert info["zp_a"] == 1
    assert abs(info["scale_b"] - 0.04) < 1e-6
    assert info["zp_b"] == -2


def test_annotate_sets_composite():
    mod = build_qdq_relax_module(32, 32, 32, 0.05, 1, 0.04, -2, 0.08, 3)
    fused, info = annotate_composite(mod)
    assert info is not None
    assert fused["main"].attrs["Composite"] == COMPOSITE_NAME


def test_negative_no_bias_does_not_match():
    mod = build_negative_no_bias_module(32, 32, 32)
    info = match_qdq_pattern(mod["main"])
    assert info is None
    fused, info2 = annotate_composite(mod)
    assert info2 is None
    attrs = fused["main"].attrs
    assert attrs is None or attrs.get("Composite") != COMPOSITE_NAME


def test_dfpattern_fuses_positive():
    mod = build_qdq_relax_module(32, 32, 32, 0.05, 1, 0.04, -2, 0.08, 3)
    fused, matched = fuse_qdq_by_df_pattern(mod)
    assert matched is True
    codegen_funcs = [
        fused[gv]
        for gv in fused.get_global_vars()
        if fused[gv].attrs is not None and fused[gv].attrs.get("Codegen") == "cutlass"
    ]
    assert codegen_funcs, "expected Codegen=cutlass after FuseOpsByPattern"
    text = str(fused)
    assert COMPOSITE_NAME in text
    # attrs_getter should have attached scales/zps at fuse time
    assert "cutlass.scale_a" in text
    assert "cutlass.zp_d" in text


def test_dfpattern_rejects_no_bias():
    mod = build_negative_no_bias_module(32, 32, 32)
    fused, matched = fuse_qdq_by_df_pattern(mod)
    assert matched is False
    assert len(list(fused.get_global_vars())) == 1


def test_dfpattern_pipeline_rewrites_to_packed():
    mod = build_qdq_relax_module(32, 32, 32, 0.05, 1, 0.04, -2, 0.08, 3)
    rewritten, info = apply_pipeline(mod, method="dfpattern")
    assert info is not None and info["matched"] is True
    text = str(rewritten)
    # RunCodegen lowers Codegen=cutlass via relax.ext.cutlass → call_dps_packed(symbol)
    assert "call_dps_packed" in text
    assert "_cutlass" in text
    assert tvm.get_global_func("relax.ext.cutlass", allow_missing=True) is not None
    # external BYOC module attached by RunCodegen
    attrs = rewritten.attrs
    assert attrs is not None and "external_mods" in dict(attrs)

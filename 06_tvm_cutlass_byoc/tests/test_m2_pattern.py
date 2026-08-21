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
    match_qdq_pattern,
)


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

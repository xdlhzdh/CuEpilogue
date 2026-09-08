"""Stage-6 CUTLASS BYOC wrapper (highest-priority bypass before TIR scheduling)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import tvm

from schedule_config import ScheduleConfig

_STAGE6_PY = Path(__file__).resolve().parents[2] / "06_tvm_cutlass_byoc" / "python"
if str(_STAGE6_PY) not in sys.path:
    sys.path.insert(0, str(_STAGE6_PY))


def _looks_like_qdq(mod: tvm.IRModule) -> bool:
    """Heuristic: Relax main with int8 params and clip/quantize-ish ops."""
    from tvm import relax

    for gv, func in mod.functions_items():
        if gv.name_hint != "main":
            continue
        if not isinstance(func, relax.Function):
            continue
        # Already rewritten to packed call?
        text = str(func)
        if "cu_epilogue.qgemm_bias_gelu" in text or "call_dps_packed" in text:
            return False
        if "int8" in text and ("clip" in text or "round" in text):
            return True
        # Composite already annotated by stage6 matcher
        attrs = func.attrs
        if attrs is not None and attrs.get("Composite") == "cutlass.qgemm_bias_gelu":
            return True
    return False


def maybe_apply_stage6_byoc(
    mod: tvm.IRModule, config: ScheduleConfig
) -> Tuple[tvm.IRModule, Optional[dict[str, Any]]]:
    """
    If enabled and module looks like stage-5 QDQ semantics, run stage-6
    apply_pipeline → call_dps_packed. Returns (mod, meta_or_None).
    """
    if not config.enable_byoc_stage6:
        return mod, None
    if not _looks_like_qdq(mod):
        return mod, None

    try:
        from relax_pipeline import apply_pipeline  # stage6
    except ImportError as exc:  # pragma: no cover
        return mod, {"error": f"stage6 import failed: {exc}", "applied": False}

    try:
        rewritten, info = apply_pipeline(mod)
        meta = {
            "applied": True,
            "name": "cu_epilogue.qgemm_bias_gelu",
            "info": info,
            "shapes": [],
        }
        return rewritten, meta
    except Exception as exc:  # noqa: BLE001
        # Pattern did not match or rewrite failed — leave module unchanged
        return mod, {"applied": False, "error": str(exc)}

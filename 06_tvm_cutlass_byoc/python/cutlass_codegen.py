"""Register ``relax.ext.cutlass`` Codegen hook for FuseOpsByPattern + RunCodegen.

Flow::

    FuseOpsByPattern(annotate_codegen=True)
      → Composite attrs via attrs_getter (scale/zp)
      → Codegen="cutlass"
    RunCodegen
      → calls this hook ``relax.ext.cutlass``
      → rewrites to ``call_dps_packed(<global_symbol>)``
    Runtime
      → packed ``global_symbol`` (bound here) → stage-5 CUTLASS ``.so``

Note: ``RunCodegen`` requires the hook to return ``runtime.Module`` list for
``external_mods``. We do not compile C here — the executable path is the
Python packed func bound to ``global_symbol``. The returned module is only a
placeholder to satisfy that API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import tvm
from tvm import relax
from tvm_ffi import register_global_func

from relax_patterns import COMPOSITE_NAME

_LAST_META: Optional[dict] = None

_META_ATTR_KEYS = (
    "cutlass.scale_a",
    "cutlass.zp_a",
    "cutlass.scale_b",
    "cutlass.zp_b",
    "cutlass.scale_d",
    "cutlass.zp_d",
)


def _walk_functions(expr: relax.Expr, out: List[relax.Function]) -> None:
    if isinstance(expr, relax.Function):
        out.append(expr)
        _walk_functions(expr.body, out)
    elif isinstance(expr, relax.SeqExpr):
        for block in expr.blocks:
            for binding in block.bindings:
                if isinstance(binding, relax.VarBinding):
                    _walk_functions(binding.value, out)
        _walk_functions(expr.body, out)
    elif isinstance(expr, relax.Call):
        for arg in expr.args:
            _walk_functions(arg, out)


def _meta_from_attrs(attrs) -> Optional[dict]:
    """Build runtime meta dict from Composite attrs written by attrs_getter."""
    if attrs is None:
        return None
    if any(k not in attrs for k in _META_ATTR_KEYS):
        return None
    return {
        "scale_a": float(attrs["cutlass.scale_a"]),
        "zp_a": int(attrs["cutlass.zp_a"]),
        "scale_b": float(attrs["cutlass.scale_b"]),
        "zp_b": int(attrs["cutlass.zp_b"]),
        "scale_d": float(attrs["cutlass.scale_d"]),
        "zp_d": int(attrs["cutlass.zp_d"]),
        "matched": True,
    }


def extract_meta_from_codegen_func(func: relax.Function) -> dict:
    """Read fuse-time attrs from nested Composite (or the function itself)."""
    funcs: List[relax.Function] = []
    _walk_functions(func, funcs)
    for f in funcs:
        attrs = f.attrs
        if attrs is not None and attrs.get("Composite") == COMPOSITE_NAME:
            meta = _meta_from_attrs(attrs)
            if meta is not None:
                return meta
    meta = _meta_from_attrs(func.attrs)
    if meta is not None:
        return meta
    raise ValueError(
        "cutlass codegen: Composite missing cutlass.scale_*/zp_* attrs "
        "(FusionPattern.attrs_getter did not run?)"
    )


def extract_meta_from_fused_module(mod: tvm.IRModule) -> Optional[dict]:
    """Find Codegen/Composite functions and read fuse-time attrs."""
    for gv in mod.get_global_vars():
        func = mod[gv]
        if not isinstance(func, relax.Function):
            continue
        attrs = func.attrs
        if attrs is None:
            continue
        if attrs.get("Codegen") == "cutlass" or attrs.get("Composite") == COMPOSITE_NAME:
            try:
                return extract_meta_from_codegen_func(func)
            except ValueError:
                continue
    return None


def _make_runtime_packed(meta: dict):
    """Closure: call_dps_packed(symbol) → stage-5 .so via relax_pipeline."""

    def _packed(a, b, bias, out):
        from relax_pipeline import _qgemm_packed, register_qgemm_meta

        register_qgemm_meta(
            meta["scale_a"],
            meta["zp_a"],
            meta["scale_b"],
            meta["zp_b"],
            meta["scale_d"],
            meta["zp_d"],
        )
        return _qgemm_packed(a, b, bias, out)

    return _packed


def _bind_runtime_symbol(symbol: str, meta: dict) -> None:
    """Bind the name RunCodegen will emit in ``call_dps_packed``."""
    register_global_func(symbol, _make_runtime_packed(meta), override=True)


def _placeholder_external_module(symbol: str) -> tvm.runtime.Module:
    """Satisfy RunCodegen's ``external_mods`` requirement (not the real kernel)."""
    create = tvm.get_global_func("runtime.CSourceModuleCreate")
    return create(
        f"/* placeholder for {symbol}; runtime is Python packed → CUTLASS .so */\n",
        "c",
        [symbol],
        [],
    )


@register_global_func("relax.ext.cutlass", override=True)
def cutlass_compiler(
    funcs: List[relax.Function],
    options: Dict[str, Any],
    constant_names: Dict[Any, str],
):
    """BYOC Codegen hook: read fuse attrs, bind runtime symbol, return placeholders."""
    del options, constant_names
    global _LAST_META

    modules: List[tvm.runtime.Module] = []
    for func in funcs:
        if not isinstance(func, relax.Function):
            raise TypeError(f"expected relax.Function, got {type(func)}")
        if func.attrs is None or "global_symbol" not in func.attrs:
            raise ValueError("Codegen function missing global_symbol attr")

        symbol = str(func.attrs["global_symbol"])
        meta = extract_meta_from_codegen_func(func)
        _LAST_META = meta

        # Executable path (required). Placeholder module (API-only).
        _bind_runtime_symbol(symbol, meta)
        modules.append(_placeholder_external_module(symbol))

    return modules


def ensure_cutlass_codegen_registered() -> None:
    """Importing this module registers ``relax.ext.cutlass`` via decorator."""
    assert tvm.get_global_func("relax.ext.cutlass", allow_missing=True) is not None


def last_codegen_meta() -> Optional[dict]:
    return _LAST_META

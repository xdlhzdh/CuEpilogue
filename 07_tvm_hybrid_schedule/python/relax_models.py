"""Hand-built Relax IRModules for hybrid-schedule tests (no .mlir files)."""
from __future__ import annotations

import sys
from pathlib import Path

import tvm
from tvm import relax

_STAGE6_PY = Path(__file__).resolve().parents[2] / "06_tvm_cutlass_byoc" / "python"
if str(_STAGE6_PY) not in sys.path:
    sys.path.insert(0, str(_STAGE6_PY))


def build_matmul_module(m: int, n: int, k: int) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    a = relax.Var("A", relax.TensorType((m, k), "float32"))
    b = relax.Var("B", relax.TensorType((k, n), "float32"))
    with bb.function("main", [a, b]):
        with bb.dataflow():
            mm = bb.emit(relax.op.matmul(a, b, out_dtype="float32"))
            gv = bb.emit_output(mm)
        bb.emit_func_output(gv)
    return bb.get()


def build_elementwise_module(m: int, n: int) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    a = relax.Var("A", relax.TensorType((m, n), "float32"))
    b = relax.Var("B", relax.TensorType((m, n), "float32"))
    with bb.function("main", [a, b]):
        with bb.dataflow():
            s = bb.emit(relax.op.add(a, b))
            r = bb.emit(relax.op.nn.relu(s))
            gv = bb.emit_output(r)
        bb.emit_func_output(gv)
    return bb.get()


def build_hybrid_native_module(
    big_m: int = 256, big_n: int = 256, big_k: int = 256, small: int = 32
) -> tvm.IRModule:
    """
    IRModule with two entry functions: large matmul (likely MS after legalize)
    and small elementwise (likely DLight). Pipeline legalizes both PrimFuncs.
    """
    mod_mm = build_matmul_module(big_m, big_n, big_k)
    mod_el = build_elementwise_module(small, small)
    # Rename elementwise main → elem_main to avoid GlobalVar clash
    updates = {}
    for gv, func in mod_mm.functions_items():
        updates[gv] = func
    for gv, func in mod_el.functions_items():
        if gv.name_hint == "main":
            new_gv = tvm.ir.GlobalVar("elem_main")
            updates[new_gv] = func.with_attr("global_symbol", "elem_main")
        else:
            updates[tvm.ir.GlobalVar(gv.name_hint + "_elem")] = func
    return tvm.IRModule(updates)


def build_qdq_module(
    m: int = 64,
    n: int = 64,
    k: int = 64,
    sa: float = 0.05,
    za: int = 1,
    sb: float = 0.04,
    zb: int = -2,
    sd: float = 0.08,
    zd: int = 3,
) -> tvm.IRModule:
    from relax_patterns import build_qdq_relax_module

    return build_qdq_relax_module(m, n, k, sa, za, sb, zb, sd, zd)


def legalize(mod: tvm.IRModule, target: tvm.target.Target | None = None) -> tvm.IRModule:
    target = target or tvm.target.Target("llvm")
    with target:
        try:
            mod = relax.get_pipeline("zero")(mod)
        except Exception:  # noqa: BLE001
            mod = relax.transform.LegalizeOps()(mod)
    return mod

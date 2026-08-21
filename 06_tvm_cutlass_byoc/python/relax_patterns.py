"""Relax DFPattern for DQ + matmul + bias + FastGELU + Q (stage-5 semantics)."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import tvm
from tvm import relax

COMPOSITE_NAME = "cutlass.qgemm_bias_gelu"
FAST_GELU_CONST = -2.455492


def build_qdq_relax_module(
    m: int,
    n: int,
    k: int,
    scale_a: float,
    zp_a: int,
    scale_b: float,
    zp_b: int,
    scale_d: float,
    zp_d: int,
) -> tvm.IRModule:
    """Hand-built Relax IR matching the fused QDQ chain (BlockBuilder API)."""
    bb = relax.BlockBuilder()
    a = relax.Var("A", relax.TensorType((m, k), "int8"))
    b = relax.Var("B", relax.TensorType((k, n), "int8"))
    bias = relax.Var("bias", relax.TensorType((n,), "float32"))

    with bb.function("main", [a, b, bias]):
        with bb.dataflow():
            a_f32 = bb.emit(relax.op.astype(a, "float32"))
            b_f32 = bb.emit(relax.op.astype(b, "float32"))
            a_dq = bb.emit(
                relax.op.multiply(
                    relax.op.subtract(a_f32, relax.const(float(zp_a), "float32")),
                    relax.const(scale_a, "float32"),
                )
            )
            b_dq = bb.emit(
                relax.op.multiply(
                    relax.op.subtract(b_f32, relax.const(float(zp_b), "float32")),
                    relax.const(scale_b, "float32"),
                )
            )
            mm = bb.emit(relax.op.matmul(a_dq, b_dq, out_dtype="float32"))
            biased = bb.emit(relax.op.add(mm, bias))
            # FastGELU: x / (1 + exp2(c*x))  via power(2, c*x)
            t = bb.emit(
                relax.op.multiply(biased, relax.const(FAST_GELU_CONST, "float32"))
            )
            e = bb.emit(relax.op.power(relax.const(2.0, "float32"), t))
            den = bb.emit(relax.op.add(e, relax.const(1.0, "float32")))
            gelu = bb.emit(relax.op.divide(biased, den))
            q = bb.emit(
                relax.op.add(
                    relax.op.divide(gelu, relax.const(scale_d, "float32")),
                    relax.const(float(zp_d), "float32"),
                )
            )
            q_r = bb.emit(relax.op.round(q))
            q_c = bb.emit(relax.op.clip(q_r, -128.0, 127.0))
            out = bb.emit(relax.op.astype(q_c, "int8"))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)

    return bb.get()


def build_negative_no_bias_module(m: int, n: int, k: int) -> tvm.IRModule:
    """Negative case: matmul + gelu without bias — must not fuse."""
    bb = relax.BlockBuilder()
    a = relax.Var("A", relax.TensorType((m, k), "float32"))
    b = relax.Var("B", relax.TensorType((k, n), "float32"))
    with bb.function("main", [a, b]):
        with bb.dataflow():
            mm = bb.emit(relax.op.matmul(a, b, out_dtype="float32"))
            t = bb.emit(
                relax.op.multiply(mm, relax.const(FAST_GELU_CONST, "float32"))
            )
            e = bb.emit(relax.op.power(relax.const(2.0, "float32"), t))
            den = bb.emit(relax.op.add(e, relax.const(1.0, "float32")))
            gelu = bb.emit(relax.op.divide(mm, den))
            gv = bb.emit_output(gelu)
        bb.emit_func_output(gv)
    return bb.get()


def _extract_scalar(expr) -> Optional[float]:
    if isinstance(expr, relax.Constant):
        data = expr.data.numpy()
        if data.size == 1:
            return float(data.reshape(()))
    return None


def _op_name(call: relax.Call) -> str:
    if isinstance(call.op, tvm.ir.Op):
        return call.op.name
    return str(call.op)


def match_qdq_pattern(func: relax.Function) -> Optional[dict]:
    """Walk dataflow and detect DQ-matmul-bias-FastGELU-Q."""
    if not isinstance(func.body, relax.SeqExpr):
        return None
    blocks = func.body.blocks
    if len(blocks) != 1 or not isinstance(blocks[0], relax.DataflowBlock):
        return None
    bindings = list(blocks[0].bindings)
    by_var = {}
    for b in bindings:
        if isinstance(b, relax.VarBinding):
            by_var[b.var] = b.value

    matmul_var = None
    matmul_call = None
    for b in bindings:
        if not isinstance(b, relax.VarBinding):
            continue
        v = b.value
        if isinstance(v, relax.Call) and _op_name(v) == "relax.matmul":
            matmul_var = b.var
            matmul_call = v
            break
    if matmul_call is None:
        return None

    bias_add_var = None
    bias_arg = None
    for b in bindings:
        if not isinstance(b, relax.VarBinding):
            continue
        v = b.value
        if (
            isinstance(v, relax.Call)
            and _op_name(v) == "relax.add"
            and len(v.args) == 2
            and (v.args[0] == matmul_var or v.args[1] == matmul_var)
        ):
            bias_add_var = b.var
            bias_arg = v.args[1] if v.args[0] == matmul_var else v.args[0]
            break
    if bias_add_var is None:
        return None

    # FastGELU: divide(biased, add(power(2, mul(biased, c)), 1))
    gelu_var = None
    for b in bindings:
        if not isinstance(b, relax.VarBinding):
            continue
        v = b.value
        if not (isinstance(v, relax.Call) and _op_name(v) == "relax.divide"):
            continue
        if v.args[0] != bias_add_var:
            continue
        den = v.args[1]
        den_val = by_var.get(den, den)
        if not (isinstance(den_val, relax.Call) and _op_name(den_val) == "relax.add"):
            continue
        has_fast = False
        for arg in den_val.args:
            av = by_var.get(arg, arg)
            if isinstance(av, relax.Call) and _op_name(av) == "relax.power":
                # power(2, mul(...))
                base = _extract_scalar(by_var.get(av.args[0], av.args[0]))
                if base is None and isinstance(av.args[0], relax.Constant):
                    base = _extract_scalar(av.args[0])
                exp = by_var.get(av.args[1], av.args[1])
                if base is not None and abs(base - 2.0) < 1e-6 and isinstance(
                    exp, relax.Call
                ) and _op_name(exp) == "relax.multiply":
                    for ma in exp.args:
                        sc = _extract_scalar(
                            by_var.get(ma, ma) if isinstance(ma, relax.Var) else ma
                        )
                        if sc is None and isinstance(ma, relax.Constant):
                            sc = _extract_scalar(ma)
                        if sc is not None and abs(sc - FAST_GELU_CONST) < 1e-5:
                            has_fast = True
        if has_fast:
            gelu_var = b.var
            break
    if gelu_var is None:
        return None

    has_quant = False
    for b in bindings:
        if not isinstance(b, relax.VarBinding):
            continue
        v = b.value
        if isinstance(v, relax.Call) and _op_name(v) == "relax.clip":
            has_quant = True
            break
        if isinstance(v, relax.Call) and _op_name(v) == "relax.astype":
            src = by_var.get(v.args[0], v.args[0])
            if isinstance(src, relax.Call) and _op_name(src) == "relax.clip":
                has_quant = True
                break
    if not has_quant:
        return None

    def parse_dq(arg) -> Optional[Tuple[Any, float, float]]:
        val = by_var.get(arg, arg)
        if not (isinstance(val, relax.Call) and _op_name(val) == "relax.multiply"):
            return None
        sub = None
        scale = None
        for a in val.args:
            av = by_var.get(a, a)
            if isinstance(av, relax.Call) and _op_name(av) == "relax.subtract":
                sub = av
            else:
                scale = _extract_scalar(av if not isinstance(a, relax.Var) else by_var.get(a, a))
                if scale is None:
                    scale = _extract_scalar(a if isinstance(a, relax.Constant) else av)
        if sub is None or scale is None:
            return None
        zp = None
        src = None
        for a in sub.args:
            av = by_var.get(a, a)
            sc = _extract_scalar(av if not isinstance(a, relax.Var) else by_var.get(a, a))
            if sc is None and isinstance(a, relax.Constant):
                sc = _extract_scalar(a)
            if sc is not None:
                zp = sc
            else:
                src = a
        if zp is None or src is None:
            return None
        src_v = by_var.get(src, src)
        if isinstance(src_v, relax.Call) and _op_name(src_v) == "relax.astype":
            src = src_v.args[0]
        return src, scale, zp

    dq_a = parse_dq(matmul_call.args[0])
    dq_b = parse_dq(matmul_call.args[1])
    if dq_a is None or dq_b is None:
        return None

    scale_d, zp_d = None, None
    for b in bindings:
        if not isinstance(b, relax.VarBinding):
            continue
        v = b.value
        if isinstance(v, relax.Call) and _op_name(v) == "relax.divide" and v.args[0] == gelu_var:
            scale_d = _extract_scalar(by_var.get(v.args[1], v.args[1]))
            if scale_d is None and isinstance(v.args[1], relax.Constant):
                scale_d = _extract_scalar(v.args[1])
        if isinstance(v, relax.Call) and _op_name(v) == "relax.add":
            for a in v.args:
                sc = _extract_scalar(by_var.get(a, a) if isinstance(a, relax.Var) else a)
                if sc is not None and abs(sc - 0.0) > 0.5:
                    # likely zp_d (integer-ish)
                    if abs(sc - FAST_GELU_CONST) > 1.0:
                        zp_d = sc

    params = list(func.params)
    a_src, sa, za = dq_a
    b_src, sb, zb = dq_b

    def param_index(expr) -> Optional[int]:
        for i, p in enumerate(params):
            if expr == p:
                return i
        return None

    ia, ib, ibias = param_index(a_src), param_index(b_src), param_index(bias_arg)
    if ia is None or ib is None or ibias is None:
        return None

    return {
        "param_A": ia,
        "param_B": ib,
        "param_bias": ibias,
        "scale_a": float(sa),
        "zp_a": int(round(za)),
        "scale_b": float(sb),
        "zp_b": int(round(zb)),
        "scale_d": float(scale_d) if scale_d is not None else 0.08,
        "zp_d": int(round(zp_d)) if zp_d is not None else 0,
        "matched": True,
    }


def annotate_composite(mod: tvm.IRModule) -> Tuple[tvm.IRModule, Optional[dict]]:
    """If main matches, attach composite attr metadata for codegen."""
    gvs = [gv for gv in mod.get_global_vars() if gv.name_hint == "main"]
    if not gvs:
        # single-function module
        main = mod[mod.get_global_vars()[0]]
        main_gv = mod.get_global_vars()[0]
    else:
        main_gv = gvs[0]
        main = mod[main_gv]

    info = match_qdq_pattern(main)
    if info is None:
        return mod, None

    new_main = main.with_attr("Composite", COMPOSITE_NAME)
    for key in (
        "scale_a",
        "zp_a",
        "scale_b",
        "zp_b",
        "scale_d",
        "zp_d",
        "param_A",
        "param_B",
        "param_bias",
    ):
        new_main = new_main.with_attr(f"cutlass.{key}", info[key])

    new_mod = tvm.IRModule({main_gv: new_main})
    return new_mod, info

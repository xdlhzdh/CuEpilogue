"""Relax DFPattern for DQ + matmul + bias + FastGELU + Q (stage-5 semantics)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import tvm
from tvm import relax

COMPOSITE_NAME = "cutlass.qgemm_bias_gelu"
# Stage-5 FastGELU: x / (1 + exp2(c*x)) with c = -2.455492
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


def _extract_scalar(expr: relax.Expr) -> Optional[float]:
    """Extract a scalar constant; returns None for non-scalars."""
    if isinstance(expr, relax.Constant):
        data = expr.data.numpy()
        if data.size == 1:
            return float(data.reshape(()))
    return None


def _op_name(call: relax.Call) -> str:
    if isinstance(call.op, tvm.ir.Op):
        return call.op.name
    return str(call.op)


def _same_expr(a: relax.Expr, b: relax.Expr) -> bool:
    """SSA / structural identity check without depending on missing APIs."""
    if a is b or a == b:
        return True
    try:
        tvm.ir.assert_structural_equal(a, b)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def match_qdq_pattern(
    func: relax.Function, debug: bool = False
) -> Optional[dict]:
    """Reverse-trace DQ-matmul-bias-FastGELU-Q; independent of binding order.

    Expected chain (stage-5 semantics)::

        DQ(A), DQ(B) -> matmul -> +bias -> FastGELU
          -> /scale_d + zp_d -> round -> clip [-> astype int8]

    Concrete Relax IR example (from ``build_qdq_relax_module(64, 64, 64,
    scale_a=0.05, zp_a=1, scale_b=0.04, zp_b=-2, scale_d=0.08, zp_d=3)``;
    all bindings live in a single ``R.dataflow()`` block)::

        @R.function
        def main(
            A: R.Tensor((64, 64), dtype="int8"),
            B: R.Tensor((64, 64), dtype="int8"),
            bias: R.Tensor((64,), dtype="float32"),
        ) -> R.Tensor((64, 64), dtype="int8"):
            with R.dataflow():
                # DQ(A): (astype(A) - zp_a) * scale_a
                lv = R.astype(A, dtype="float32")
                lv2 = R.subtract(lv, R.const(1.0, "float32"))
                lv3 = R.multiply(lv2, R.const(0.05, "float32"))
                # DQ(B): (astype(B) - zp_b) * scale_b
                lv1 = R.astype(B, dtype="float32")
                lv4 = R.subtract(lv1, R.const(-2.0, "float32"))
                lv5 = R.multiply(lv4, R.const(0.04, "float32"))
                # matmul + bias
                lv6 = R.matmul(lv3, lv5, out_dtype="float32")
                lv7 = R.add(lv6, bias)
                # FastGELU: x / (1 + 2^(c*x)), c = -2.455492; numerator shares lv7
                lv8 = R.multiply(lv7, R.const(-2.455492, "float32"))
                lv9 = R.power(R.const(2.0, "float32"), lv8)
                lv10 = R.add(lv9, R.const(1.0, "float32"))
                lv11 = R.divide(lv7, lv10)
                # Q: clip(round(gelu / scale_d + zp_d)) -> int8
                lv12 = R.divide(lv11, R.const(0.08, "float32"))
                lv13 = R.add(lv12, R.const(3.0, "float32"))
                lv14 = R.round(lv13)
                lv15 = R.clip(lv14, -128.0, 127.0)
                lv16 = R.astype(lv15, dtype="int8")
                gv = lv16
                R.output(gv)
            return gv
    """
    if not isinstance(func.body, relax.SeqExpr):
        if debug:
            print("[match] fail: body not SeqExpr")
        return None
    blocks = func.body.blocks
    if len(blocks) != 1 or not isinstance(blocks[0], relax.DataflowBlock):
        if debug:
            print("[match] fail: not single DataflowBlock")
        return None
    bindings = list(blocks[0].bindings)

    by_var: Dict[relax.Var, relax.Expr] = {}
    for b in bindings:
        if isinstance(b, relax.VarBinding):
            by_var[b.var] = b.value

    def resolve(expr: relax.Expr) -> relax.Expr:
        while isinstance(expr, relax.Var) and expr in by_var:
            expr = by_var[expr]
        return expr

    # ---- collect output quant clip nodes (clip or astype(clip)) ----
    quant_candidates: list[tuple[relax.Var, relax.Call]] = []
    for b in bindings:
        if not isinstance(b, relax.VarBinding):
            continue
        val = resolve(b.value)
        if not isinstance(val, relax.Call):
            continue
        if _op_name(val) == "relax.clip":
            quant_candidates.append((b.var, val))
        elif _op_name(val) == "relax.astype":
            inner = resolve(val.args[0])
            if isinstance(inner, relax.Call) and _op_name(inner) == "relax.clip":
                quant_candidates.append((b.var, inner))

    if not quant_candidates:
        if debug:
            print("[match] fail: no quant clip node found")
        return None

    def parse_dq(arg: relax.Expr) -> Optional[Tuple[relax.Expr, float, float]]:
        """Parse (x - zp) * scale, optionally through astype."""
        val = resolve(arg)
        if not (isinstance(val, relax.Call) and _op_name(val) == "relax.multiply"):
            return None
        sub = None
        scale = None
        for a in val.args:
            av = resolve(a)
            if isinstance(av, relax.Call) and _op_name(av) == "relax.subtract":
                sub = av
            else:
                scale = _extract_scalar(av)
        if sub is None or scale is None:
            return None
        zp = None
        src = None
        for a in sub.args:
            av = resolve(a)
            sc = _extract_scalar(av)
            if sc is not None:
                zp = sc
            else:
                src = a
        if zp is None or src is None:
            return None
        src_v = resolve(src)
        if isinstance(src_v, relax.Call) and _op_name(src_v) == "relax.astype":
            src = src_v.args[0]
        return src, scale, zp

    params = list(func.params)

    def param_index(e: relax.Expr) -> Optional[int]:
        for idx, p in enumerate(params):
            if _same_expr(e, p):
                return idx
        return None

    for _, clip_call in quant_candidates:
        # Peel optional round: clip(round(q)) or clip(q)
        quant_expr = resolve(clip_call.args[0])
        if isinstance(quant_expr, relax.Call) and _op_name(quant_expr) == "relax.round":
            quant_expr = resolve(quant_expr.args[0])

        # Quant: add(divide(gelu, scale_d), zp_d)  (either operand order)
        if not (
            isinstance(quant_expr, relax.Call)
            and _op_name(quant_expr) == "relax.add"
            and len(quant_expr.args) == 2
        ):
            if debug:
                print("[match] fail: quant node is not binary add(gelu/scale, zp)")
            continue

        lhs, rhs = resolve(quant_expr.args[0]), resolve(quant_expr.args[1])

        def as_quant_pair(
            a: relax.Expr, b: relax.Expr
        ) -> Optional[Tuple[relax.Expr, float, float]]:
            # a = divide(gelu, scale), b = zp scalar
            if not (isinstance(a, relax.Call) and _op_name(a) == "relax.divide"):
                return None
            zp = _extract_scalar(b)
            scale = _extract_scalar(resolve(a.args[1]))
            if zp is None or scale is None:
                return None
            return resolve(a.args[0]), scale, zp

        parsed = as_quant_pair(lhs, rhs) or as_quant_pair(rhs, lhs)
        if parsed is None:
            if debug:
                print("[match] fail: cannot split quant add into gelu/scale + zp")
            continue
        gelu_div_expr, scale_d, zp_d = parsed

        if not (
            isinstance(gelu_div_expr, relax.Call)
            and _op_name(gelu_div_expr) == "relax.divide"
        ):
            if debug:
                print("[match] fail: FastGELU root is not divide")
            continue

        # FastGELU: divide(bias_add, add(power(2, mul(x, c)), 1))
        bias_add_expr = resolve(gelu_div_expr.args[0])
        den_expr = resolve(gelu_div_expr.args[1])
        if not (isinstance(den_expr, relax.Call) and _op_name(den_expr) == "relax.add"):
            if debug:
                print("[match] fail: gelu denominator not add")
            continue

        pow_term: Optional[relax.Call] = None
        for arg in den_expr.args:
            a_res = resolve(arg)
            if isinstance(a_res, relax.Call) and _op_name(a_res) == "relax.power":
                pow_term = a_res
                break
        if pow_term is None:
            if debug:
                print("[match] fail: cannot find power term in denominator")
            continue

        base_val = _extract_scalar(resolve(pow_term.args[0]))
        exp_expr = resolve(pow_term.args[1])
        if base_val is None or abs(base_val - 2.0) > 1e-6:
            if debug:
                print(f"[match] fail: power base != 2, got {base_val}")
            continue
        if not (
            isinstance(exp_expr, relax.Call) and _op_name(exp_expr) == "relax.multiply"
        ):
            if debug:
                print("[match] fail: power exp not multiply")
            continue

        c_const: Optional[float] = None
        mul_input_x: Optional[relax.Expr] = None
        for a in exp_expr.args:
            av = resolve(a)
            sc = _extract_scalar(av)
            if sc is not None and abs(sc - FAST_GELU_CONST) < 1e-5:
                c_const = sc
            else:
                mul_input_x = av
        if c_const is None or mul_input_x is None:
            if debug:
                print("[match] fail: FastGELU constant missing")
            continue
        # DAG: multiply input must be the same bias_add used as gelu numerator
        if not _same_expr(mul_input_x, bias_add_expr):
            if debug:
                print("[match] fail: FastGELU mul input != bias_add")
            continue

        if not (
            isinstance(bias_add_expr, relax.Call)
            and _op_name(bias_add_expr) == "relax.add"
        ):
            if debug:
                print("[match] fail: gelu numerator not add(bias_add)")
            continue

        matmul_call: Optional[relax.Call] = None
        bias_arg_expr: Optional[relax.Expr] = None
        for arg in bias_add_expr.args:
            av = resolve(arg)
            if isinstance(av, relax.Call) and _op_name(av) == "relax.matmul":
                matmul_call = av
            else:
                bias_arg_expr = arg
        if matmul_call is None or bias_arg_expr is None:
            if debug:
                print("[match] fail: bias_add cannot split matmul / bias")
            continue

        dq_a = parse_dq(matmul_call.args[0])
        dq_b = parse_dq(matmul_call.args[1])
        if dq_a is None or dq_b is None:
            if debug:
                print("[match] fail: dq_a or dq_b parse failed")
            continue
        a_src, sa, za = dq_a
        b_src, sb, zb = dq_b

        ia = param_index(a_src)
        ib = param_index(b_src)
        ibias = param_index(bias_arg_expr)
        if ia is None or ib is None or ibias is None:
            if debug:
                print(f"[match] fail: param index ia={ia}, ib={ib}, ibias={ibias}")
            continue

        return {
            "param_A": ia,
            "param_B": ib,
            "param_bias": ibias,
            "scale_a": float(sa),
            "zp_a": int(round(za)),
            "scale_b": float(sb),
            "zp_b": int(round(zb)),
            "scale_d": float(scale_d),
            "zp_d": int(round(zp_d)),
            "matched": True,
        }

    if debug:
        print("[match] all quant candidates tried, no match")
    return None



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


def _check_qdq_fusion(ctx) -> bool:
    """Reject matches that are not stage-5 FastGELU (c and power base)."""
    annotated = ctx.annotated_expr
    c = _extract_scalar(annotated.get("gelu_c"))
    base = _extract_scalar(annotated.get("pow_base"))
    return (
        c is not None
        and abs(c - FAST_GELU_CONST) < 1e-5
        and base is not None
        and abs(base - 2.0) < 1e-6
    )


def _qdq_attrs_getter(annotated: dict) -> dict:
    """Attach affine params during FuseOpsByPattern (no second hand matcher)."""

    def scalar(key: str) -> float:
        val = _extract_scalar(annotated[key])
        if val is None:
            raise ValueError(f"FusionPattern attrs_getter: missing scalar {key}")
        return float(val)

    return {
        "cutlass.scale_a": str(scalar("scale_a")),
        "cutlass.zp_a": str(int(round(scalar("zp_a")))),
        "cutlass.scale_b": str(scalar("scale_b")),
        "cutlass.zp_b": str(int(round(scalar("zp_b")))),
        "cutlass.scale_d": str(scalar("scale_d")),
        "cutlass.zp_d": str(int(round(scalar("zp_d")))),
        "cutlass.matched": "1",
    }


def build_qdq_fusion_pattern():
    """DFPattern / FusionPattern for FuseOpsByPattern (official BYOC path).

    ``attrs_getter`` writes ``cutlass.scale_*`` / ``cutlass.zp_*`` onto the
    fused Composite function so ``relax.ext.cutlass`` can read them directly.
    """
    from tvm.relax.dpl import is_const, is_op, wildcard
    from tvm.relax.transform import FusionPattern

    a = wildcard()
    b = wildcard()
    bias = wildcard()

    zp_a = is_const()
    scale_a = is_const()
    zp_b = is_const()
    scale_b = is_const()
    a_dq = is_op("relax.multiply")(
        is_op("relax.subtract")(is_op("relax.astype")(a), zp_a),
        scale_a,
    )
    b_dq = is_op("relax.multiply")(
        is_op("relax.subtract")(is_op("relax.astype")(b), zp_b),
        scale_b,
    )
    mm = is_op("relax.matmul")(a_dq, b_dq)
    biased = is_op("relax.add")(mm, bias)

    gelu_c = is_const()
    gelu_mul = is_op("relax.multiply")(biased, gelu_c)
    pow_base = is_const()
    gelu_pow = is_op("relax.power")(pow_base, gelu_mul)
    gelu_den = is_op("relax.add")(gelu_pow, is_const())
    gelu = is_op("relax.divide")(biased, gelu_den)

    scale_d = is_const()
    zp_d = is_const()
    # clip(value, min, max) — arity must be 3 for TVM 0.26 matching
    out = is_op("relax.astype")(
        is_op("relax.clip")(
            is_op("relax.round")(
                is_op("relax.add")(
                    is_op("relax.divide")(gelu, scale_d),
                    zp_d,
                )
            ),
            wildcard(),
            wildcard(),
        )
    )

    return FusionPattern(
        COMPOSITE_NAME,
        out,
        annotation_patterns={
            "A": a,
            "B": b,
            "bias": bias,
            "zp_a": zp_a,
            "scale_a": scale_a,
            "zp_b": zp_b,
            "scale_b": scale_b,
            "gelu_c": gelu_c,
            "pow_base": pow_base,
            "gelu_mul": gelu_mul,
            "gelu_pow": gelu_pow,
            "scale_d": scale_d,
            "zp_d": zp_d,
        },
        check=_check_qdq_fusion,
        attrs_getter=_qdq_attrs_getter,
    )


def _module_has_cutlass_offload(mod: tvm.IRModule) -> bool:
    for gv in mod.get_global_vars():
        func = mod[gv]
        attrs = func.attrs
        if attrs is None:
            continue
        if attrs.get("Codegen") == "cutlass":
            return True
        if attrs.get("Composite") == COMPOSITE_NAME:
            return True
    return False


def fuse_qdq_by_df_pattern(mod: tvm.IRModule) -> Tuple[tvm.IRModule, bool]:
    """Fuse QDQ via FuseOpsByPattern(annotate_codegen=True).

    On success the module contains a ``Codegen="cutlass"`` wrapper (and nested
    ``Composite=cutlass.qgemm_bias_gelu``). Does not rewrite to packed call.
    """
    from tvm.relax.transform import FuseOpsByPattern

    pattern = build_qdq_fusion_pattern()
    fused = FuseOpsByPattern(
        [pattern],
        bind_constants=True,
        annotate_codegen=True,
    )(mod)
    return fused, _module_has_cutlass_offload(fused)

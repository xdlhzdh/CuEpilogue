"""Relax pipeline: pattern annotate → rewrite main to external packed call."""
from __future__ import annotations

from typing import Optional, Tuple

import tvm
from tvm import relax

from relax_patterns import (
    COMPOSITE_NAME,
    annotate_composite,
)
from so_wrapper import launch

_META = {
    "scale_a": 0.05,
    "zp_a": 1,
    "scale_b": 0.04,
    "zp_b": -2,
    "scale_d": 0.08,
    "zp_d": 3,
}


def register_qgemm_meta(sa, za, sb, zb, sd, zd) -> None:
    _META.update(
        {
            "scale_a": float(sa),
            "zp_a": int(za),
            "scale_b": float(sb),
            "zp_b": int(zb),
            "scale_d": float(sd),
            "zp_d": int(zd),
        }
    )


@tvm.register_global_func("cu_epilogue.qgemm_bias_gelu", override=True)
def _qgemm_packed(a, b, bias, out):
    """Packed func: host NDArrays → fused CUTLASS .so (via so_wrapper)."""
    import numpy as np

    A = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    B = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
    Bias = bias.numpy() if hasattr(bias, "numpy") else np.asarray(bias)
    result = launch(
        A,
        B,
        Bias,
        _META["scale_a"],
        _META["zp_a"],
        _META["scale_b"],
        _META["zp_b"],
        _META["scale_d"],
        _META["zp_d"],
    )
    if hasattr(out, "copyfrom"):
        out.copyfrom(result)
    else:
        np.copyto(np.asarray(out), result)
    return out


def fuse_and_partition(mod: tvm.IRModule) -> Tuple[tvm.IRModule, Optional[dict]]:
    return annotate_composite(mod)


def _meta_from_func(func: relax.Function) -> dict:
    return {
        "scale_a": float(func.attrs["cutlass.scale_a"]),
        "zp_a": int(func.attrs["cutlass.zp_a"]),
        "scale_b": float(func.attrs["cutlass.scale_b"]),
        "zp_b": int(func.attrs["cutlass.zp_b"]),
        "scale_d": float(func.attrs["cutlass.scale_d"]),
        "zp_d": int(func.attrs["cutlass.zp_d"]),
        "param_A": int(func.attrs["cutlass.param_A"]),
        "param_B": int(func.attrs["cutlass.param_B"]),
        "param_bias": int(func.attrs["cutlass.param_bias"]),
    }


def rewrite_to_extern(mod: tvm.IRModule, info: Optional[dict] = None) -> tvm.IRModule:
    """Replace matched main with call_dps_packed to cu_epilogue.qgemm_bias_gelu."""
    gvs = list(mod.get_global_vars())
    main_gv = next((g for g in gvs if g.name_hint == "main"), gvs[0])
    main = mod[main_gv]
    if main.attrs is None or main.attrs.get("Composite") != COMPOSITE_NAME:
        raise ValueError("module is not annotated as cutlass.qgemm_bias_gelu composite")
    meta = info or _meta_from_func(main)

    params = list(main.params)
    a_ty = params[meta["param_A"]].ty
    shape_a = [int(x) for x in a_ty.shape]
    b_ty = params[meta["param_B"]].ty
    shape_b = [int(x) for x in b_ty.shape]
    m, k = shape_a
    _k2, n = shape_b

    register_qgemm_meta(
        meta["scale_a"],
        meta["zp_a"],
        meta["scale_b"],
        meta["zp_b"],
        meta["scale_d"],
        meta["zp_d"],
    )

    bb = relax.BlockBuilder()
    a = relax.Var("A", relax.TensorType((m, k), "int8"))
    b = relax.Var("B", relax.TensorType((k, n), "int8"))
    bias = relax.Var("bias", relax.TensorType((n,), "float32"))
    with bb.function("main", [a, b, bias]):
        with bb.dataflow():
            out = bb.emit(
                relax.op.call_dps_packed(
                    "cu_epilogue.qgemm_bias_gelu",
                    [a, b, bias],
                    out_ty=relax.TensorType((m, n), "int8"),
                )
            )
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)

    out_mod = bb.get()
    new_main = out_mod["main"]
    new_main = new_main.with_attr("Composite", COMPOSITE_NAME)
    new_main = new_main.with_attr("Codegen", "cutlass")
    # Replace in a fresh module preserving GlobalVar name "main".
    return tvm.IRModule({out_mod.get_global_vars()[0]: new_main})


def apply_pipeline(mod: tvm.IRModule) -> Tuple[tvm.IRModule, dict]:
    fused, info = fuse_and_partition(mod)
    if info is None:
        raise ValueError("QDQ pattern did not match")
    rewritten = rewrite_to_extern(fused, info)
    return rewritten, info


def build_vm(mod: tvm.IRModule):
    target = tvm.target.Target("llvm")
    try:
        mod = relax.transform.LegalizeOps()(mod)
    except Exception:
        pass
    try:
        ex = relax.build(mod, target=target)
        return relax.VirtualMachine(ex, tvm.cpu())
    except Exception:
        ex = relax.vm.build(mod, target=target)
        return relax.VirtualMachine(ex, tvm.cpu())

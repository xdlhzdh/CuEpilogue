"""Relax pipeline: pattern annotate → rewrite main to external packed call."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import tvm
from tvm import relax

from relax_patterns import (
    COMPOSITE_NAME,
    annotate_composite,
    fuse_qdq_by_df_pattern,
)
from so_wrapper import device_synchronize, launch, launch_device

PipelineMethod = Literal["handwritten", "dfpattern"]

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


def _is_cuda_tensor(x) -> bool:
    """True if *x* is a TVM Tensor/NDArray already resident on a CUDA device."""
    if not hasattr(x, "data_ptr") or not hasattr(x, "device"):
        return False
    return "cuda" in str(x.device).lower()


def default_target_and_device() -> Tuple[tvm.target.Target, tvm.runtime.Device]:
    """Prefer CUDA (sm_70+ CUTLASS 2.x path); fall back to llvm/cpu only if no GPU."""
    try:
        if tvm.cuda(0).exist:
            # Explicit host keeps Relax VM host codegen on llvm while device is CUDA —
            # same split as stage-5 runtime (host driver + device CUTLASS .so).
            return tvm.target.Target("cuda", host="llvm"), tvm.cuda(0)
    except Exception:  # noqa: BLE001
        pass
    return tvm.target.Target("llvm"), tvm.cpu()


@tvm.register_global_func("cu_epilogue.qgemm_bias_gelu", override=True)
def _qgemm_packed(a, b, bias, out):
    """Packed func → stage-5 CUTLASS .so.

    Fast path: CUDA Tensors pass device pointers straight into the C ABI
    (no host round-trip), matching stage-5's device-pointer calling convention.
    Slow path: CPU/host arrays go through so_wrapper.launch (H2D → kernel → D2H).
    """
    if (
        _is_cuda_tensor(a)
        and _is_cuda_tensor(b)
        and _is_cuda_tensor(bias)
        and _is_cuda_tensor(out)
    ):
        m, k = int(a.shape[0]), int(a.shape[1])
        n = int(b.shape[1])
        launch_device(
            a.data_ptr(),
            b.data_ptr(),
            bias.data_ptr(),
            out.data_ptr(),
            m,
            n,
            k,
            _META["scale_a"],
            _META["zp_a"],
            _META["scale_b"],
            _META["zp_b"],
            _META["scale_d"],
            _META["zp_d"],
        )
        device_synchronize()
        return out

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


def _main_function(mod: tvm.IRModule):
    gvs = list(mod.get_global_vars())
    main_gv = next((g for g in gvs if g.name_hint == "main"), gvs[0])
    return main_gv, mod[main_gv]


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
    _main_gv, main = _main_function(mod)
    # Handwritten path annotates main; DFPattern path may only annotate the
    # fused Codegen wrapper — then *info* must be supplied by the caller.
    if info is None:
        if main.attrs is None or main.attrs.get("Composite") != COMPOSITE_NAME:
            raise ValueError(
                "module is not annotated as cutlass.qgemm_bias_gelu composite"
            )
        meta = _meta_from_func(main)
    else:
        meta = info

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


def apply_pipeline(
    mod: tvm.IRModule,
    method: PipelineMethod = "handwritten",
) -> Tuple[tvm.IRModule, dict]:
    """Match QDQ and offload to packed CUTLASS call.

    Parameters
    ----------
    method:
        ``"handwritten"`` — SSA reverse-trace matcher + Composite attrs →
        ``rewrite_to_extern`` (default).
        ``"dfpattern"`` — ``FuseOpsByPattern(annotate_codegen=True)`` →
        registered ``relax.ext.cutlass`` via ``RunCodegen`` (official BYOC
        Codegen hook; runtime still hits stage-5 ``.so``).
    """
    if method == "handwritten":
        fused, info = fuse_and_partition(mod)
        if info is None:
            raise ValueError("QDQ pattern did not match")
        rewritten = rewrite_to_extern(fused, info)
        return rewritten, info

    if method == "dfpattern":
        # Register Codegen hook before RunCodegen looks up relax.ext.cutlass.
        import cutlass_codegen as _cutlass_codegen

        _cutlass_codegen.ensure_cutlass_codegen_registered()

        fused, matched = fuse_qdq_by_df_pattern(mod)
        if not matched:
            raise ValueError("QDQ DFPattern / FuseOpsByPattern did not match")

        info = _cutlass_codegen.extract_meta_from_fused_module(fused)
        if info is None:
            raise ValueError("QDQ meta extract failed after FuseOpsByPattern")

        # Official BYOC lowering: invoke relax.ext.cutlass for Codegen=cutlass.
        lowered = relax.transform.RunCodegen()(fused)
        # Hook may refine meta; prefer last codegen snapshot when present.
        info = _cutlass_codegen.last_codegen_meta() or info
        return lowered, info

    raise ValueError(f"unknown pipeline method: {method!r}")


def build_vm(
    mod: tvm.IRModule,
    target: Optional[tvm.target.Target] = None,
    device: Optional[tvm.runtime.Device] = None,
):
    """Lower / build Relax VM for CUDA by default (llvm/cpu only if no GPU)."""
    if target is None or device is None:
        def_t, def_d = default_target_and_device()
        target = target or def_t
        device = device or def_d
    try:
        mod = relax.transform.LegalizeOps()(mod)
    except Exception:
        pass
    try:
        ex = relax.build(mod, target=target)
        return relax.VirtualMachine(ex, device)
    except Exception:
        ex = relax.vm.build(mod, target=target)
        return relax.VirtualMachine(ex, device)
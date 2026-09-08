"""CUDA sm_70 codegen + optional cuda(0) VM launch after hybrid schedule."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tvm
from tvm import relax
from tvm import runtime as rt


def cuda_sm70_target() -> tvm.target.Target:
    """Project CLI default: CUDA device / llvm host, arch=sm_70 (V100 tag)."""
    return tvm.target.Target("nvidia/nvidia-v100", host="llvm")


def compute_version_to_arch(version: str) -> str:
    """'7.0' → sm_70, '8.6' → sm_86. TVM Device.compute_version is major.minor."""
    parts = str(version).strip().split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return f"sm_{major}{minor}"


def detect_cuda_target_from_device(index: int = 0) -> Optional[tvm.target.Target]:
    """Build a CUDA Target from the live GPU, or None if cuda(index) is missing."""
    try:
        dev = tvm.cuda(index)
        if not bool(dev.exist):
            return None
        arch = compute_version_to_arch(str(dev.compute_version))
        attrs: dict = {
            "kind": "cuda",
            "arch": arch,
            "max_num_threads": int(dev.max_threads_per_block or 1024),
            "max_threads_per_block": int(dev.max_threads_per_block or 1024),
            "thread_warp_size": int(dev.warp_size or 32),
            "max_shared_memory_per_block": int(
                dev.max_shared_memory_per_block or 49152
            ),
        }
        return tvm.target.Target(attrs, host="llvm")
    except Exception:  # noqa: BLE001
        return None


_WMMA_MARKERS = (
    "wmma_",
    "wmma::",
    "tvm_mma_sync",
    "tvm_load_matrix_sync",
    "tvm_store_matrix_sync",
    "tvm_fill_fragment",
)


def classify_kernel_kind(mod: tvm.IRModule) -> dict[str, str]:
    """Inspect scheduled TIR only: wmma | simt | unscheduled. Does not tensorize."""
    from tvm.tirx.function import PrimFunc

    out: dict[str, str] = {}
    for gv, func in mod.functions_items():
        if not isinstance(func, PrimFunc):
            continue
        text = str(func)
        low = text.lower()
        if any(m in text or m in low for m in _WMMA_MARKERS):
            out[gv.name_hint] = "wmma"
        elif "thread_binding" in text or "blockIdx" in text:
            out[gv.name_hint] = "simt"
        else:
            out[gv.name_hint] = "unscheduled"
    return out


def parse_target(spec: Optional[str]) -> tvm.target.Target:
    """CLI target. Default / 'cuda' / 'sm_70' → cuda sm_70. 'llvm' stays host-only."""
    if spec is None or spec in ("cuda", "sm70", "sm_70", "cuda/sm70", "cuda/sm_70"):
        return cuda_sm70_target()
    if spec == "llvm":
        return tvm.target.Target("llvm")
    if spec.startswith("nvidia/") or spec == "cuda":
        return tvm.target.Target(spec, host="llvm")
    return tvm.target.Target(spec)


def is_cuda_target(target: tvm.target.Target) -> bool:
    return target.kind.name == "cuda"


def prepare_cuda_compile_env(dump_dir: str | Path) -> None:
    """Prefer nvcc when TVM_CUDA_COMPILE_MODE unset (this venv has no cuda-bindings)."""
    os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TVM_KERNEL_DUMP", str(dump_dir))


def find_cuda_source(mod: Any) -> Optional[str]:
    """Walk VMExecutable → llvm → cuda and return CUDA C."""
    stack = [mod]
    seen: set[int] = set()
    while stack:
        m = stack.pop()
        mid = id(m)
        if mid in seen:
            continue
        seen.add(mid)
        kind = getattr(m, "kind", None)
        if kind == "cuda":
            try:
                src = m.inspect_source()
            except Exception:  # noqa: BLE001
                src = None
            if src:
                return src
        for im in getattr(m, "imports", None) or []:
            stack.append(im)
    return None


def build_cuda_executable(mod: tvm.IRModule, target: tvm.target.Target):
    """Codegen scheduled TIR. Must run inside `with target:` so nvcc sees arch."""
    with target:
        try:
            return relax.build(mod, target=target)
        except Exception:
            return relax.vm.build(mod, target=target)


def export_artifacts(
    ex: Any, work_dir: str | Path, so_name: str = "hybrid_cuda_sm70.so"
) -> dict[str, str]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    so_path = work_dir / so_name
    ex.export_library(str(so_path))
    artifacts = {"so": str(so_path)}
    src = find_cuda_source(ex.mod)
    if src:
        cu_path = work_dir / "tvm_kernels.cu"
        cu_path.write_text(src, encoding="utf-8")
        artifacts["cu"] = str(cu_path)
    dump = os.environ.get("TVM_KERNEL_DUMP")
    if dump:
        artifacts["kernel_dump"] = dump
    return artifacts


def cuda0_device() -> Optional[tvm.runtime.Device]:
    try:
        dev = tvm.cuda(0)
        if dev.exist:
            return dev
    except Exception:  # noqa: BLE001
        pass
    return None


def _rand_tensor(shape: tuple[int, ...], dtype: str, device) -> Any:
    if dtype in ("float32", "float16"):
        arr = np.random.randn(*shape).astype(dtype)
    elif dtype == "int8":
        arr = np.random.randint(-8, 8, size=shape, dtype=np.int8)
    else:
        arr = np.zeros(shape, dtype=dtype)
    return rt.tensor(arr, device=device)


def run_on_cuda0(ex: Any, kind: str) -> dict[str, Any]:
    """Launch VM on cuda(0). Returns status dict; does not raise if GPU is missing."""
    dev = cuda0_device()
    if dev is None:
        return {"ran": False, "reason": "cuda(0) not present (compiler-only host)"}

    vm = relax.VirtualMachine(ex, dev)
    specs: dict[str, list[tuple[tuple[int, ...], str]]]
    if kind == "matmul":
        specs = {"main": [((256, 256), "float32"), ((256, 256), "float32")]}
    elif kind == "elem":
        specs = {"main": [((32, 32), "float32"), ((32, 32), "float32")]}
    elif kind == "hybrid":
        specs = {
            "main": [((256, 256), "float32"), ((256, 256), "float32")],
            "elem_main": [((32, 32), "float32"), ((32, 32), "float32")],
        }
    elif kind == "qdq":
        specs = {
            "main": [
                ((64, 64), "int8"),
                ((64, 64), "int8"),
                ((64,), "float32"),
            ]
        }
    else:
        return {"ran": False, "reason": f"unknown kind {kind}"}

    outputs = {}
    for name, params in specs.items():
        args = [_rand_tensor(shape, dtype, dev) for shape, dtype in params]
        out = vm[name](*args)
        arr = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        outputs[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return {"ran": True, "device": "cuda(0)", "outputs": outputs}

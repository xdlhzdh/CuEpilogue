"""ctypes wrapper around stage-5 libcutlass_qgemm_bias_gelu_runtime.so."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

_LIB: Optional[ctypes.CDLL] = None
_LAUNCH_COUNT = 0


def reset_launch_count() -> None:
    global _LAUNCH_COUNT
    _LAUNCH_COUNT = 0


def launch_count() -> int:
    return _LAUNCH_COUNT


def _default_so_candidates() -> list[Path]:
    candidates: list[Path] = []
    build = os.environ.get("CU_EPILOGUE_BUILD_DIR")
    if build:
        candidates.append(
            Path(build) / "05_qdq_fusion" / "libcutlass_qgemm_bias_gelu_runtime.so"
        )
    here = Path(__file__).resolve()
    repo = here.parents[2]
    candidates.append(
        repo / "build" / "05_qdq_fusion" / "libcutlass_qgemm_bias_gelu_runtime.so"
    )
    env_so = os.environ.get("CU_EPILOGUE_QGEMM_SO")
    if env_so:
        candidates.insert(0, Path(env_so))
    return candidates


def find_runtime_so() -> Path:
    for p in _default_so_candidates():
        if p.is_file():
            return p
    tried = ", ".join(str(p) for p in _default_so_candidates())
    raise FileNotFoundError(
        "libcutlass_qgemm_bias_gelu_runtime.so not found. Tried: " + tried
    )


def load_library(so_path: Optional[str | Path] = None) -> ctypes.CDLL:
    global _LIB
    if _LIB is not None and so_path is None:
        return _LIB
    path = Path(so_path) if so_path else find_runtime_so()
    lib = ctypes.CDLL(str(path))
    lib.cutlass_qgemm_bias_gelu.argtypes = [
        ctypes.POINTER(ctypes.c_int8),  # A
        ctypes.POINTER(ctypes.c_int8),  # B
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_int8),  # D
        ctypes.c_int64,  # M
        ctypes.c_int64,  # N
        ctypes.c_int64,  # K
        ctypes.c_float,  # scale_a
        ctypes.c_int32,  # zp_a
        ctypes.c_float,  # scale_b
        ctypes.c_int32,  # zp_b
        ctypes.c_float,  # scale_d
        ctypes.c_int32,  # zp_d
    ]
    lib.cutlass_qgemm_bias_gelu.restype = None
    _LIB = lib
    return lib


def _as_c_int8(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int8)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def _as_c_float(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def launch(
    A: np.ndarray,
    B: np.ndarray,
    bias: np.ndarray,
    scale_a: float,
    zp_a: int,
    scale_b: float,
    zp_b: int,
    scale_d: float,
    zp_d: int,
    out: Optional[np.ndarray] = None,
    *,
    device_ptrs: bool = False,
) -> np.ndarray:
    """Run fused QGEMM.

    By default A/B/bias/out are host numpy arrays and this helper copies via
    CUDA runtime through a small path: we require *device* buffers when
    device_ptrs=True. For the host path we allocate device memory with
    cudaMalloc via libcudart.
    """
    global _LAUNCH_COUNT
    lib = load_library()

    A = np.ascontiguousarray(A, dtype=np.int8)
    B = np.ascontiguousarray(B, dtype=np.int8)
    bias = np.ascontiguousarray(bias, dtype=np.float32)
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"K mismatch: A {A.shape} B {B.shape}")
    if bias.shape != (n,):
        raise ValueError(f"bias shape {bias.shape} != ({n},)")

    if out is None:
        out = np.empty((m, n), dtype=np.int8)
    else:
        out = np.ascontiguousarray(out, dtype=np.int8)
        if out.shape != (m, n):
            raise ValueError(f"out shape {out.shape} != ({m}, {n})")

    if device_ptrs:
        # Caller already passed device pointers packed as numpy object? Not used.
        raise NotImplementedError("device_ptrs path reserved for TVM DLTensor binding")

    # Host → device → kernel → host via cuda runtime.
    cudart = _load_cudart()
    dA = _cuda_malloc(cudart, A.nbytes)
    dB = _cuda_malloc(cudart, B.nbytes)
    dBias = _cuda_malloc(cudart, bias.nbytes)
    dD = _cuda_malloc(cudart, out.nbytes)
    try:
        _cuda_memcpy_h2d(cudart, dA, A)
        _cuda_memcpy_h2d(cudart, dB, B)
        _cuda_memcpy_h2d(cudart, dBias, bias)
        lib.cutlass_qgemm_bias_gelu(
            ctypes.cast(dA, ctypes.POINTER(ctypes.c_int8)),
            ctypes.cast(dB, ctypes.POINTER(ctypes.c_int8)),
            ctypes.cast(dBias, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(dD, ctypes.POINTER(ctypes.c_int8)),
            ctypes.c_int64(m),
            ctypes.c_int64(n),
            ctypes.c_int64(k),
            ctypes.c_float(scale_a),
            ctypes.c_int32(zp_a),
            ctypes.c_float(scale_b),
            ctypes.c_int32(zp_b),
            ctypes.c_float(scale_d),
            ctypes.c_int32(zp_d),
        )
        _cuda_device_sync(cudart)
        _cuda_memcpy_d2h(cudart, out, dD)
    finally:
        _cuda_free(cudart, dA)
        _cuda_free(cudart, dB)
        _cuda_free(cudart, dBias)
        _cuda_free(cudart, dD)

    _LAUNCH_COUNT += 1
    return out


def launch_device(
    dA: int,
    dB: int,
    dBias: int,
    dD: int,
    m: int,
    n: int,
    k: int,
    scale_a: float,
    zp_a: int,
    scale_b: float,
    zp_b: int,
    scale_d: float,
    zp_d: int,
) -> None:
    """Launch with raw device pointers (ints from DLTensor.data)."""
    global _LAUNCH_COUNT
    lib = load_library()
    lib.cutlass_qgemm_bias_gelu(
        ctypes.cast(dA, ctypes.POINTER(ctypes.c_int8)),
        ctypes.cast(dB, ctypes.POINTER(ctypes.c_int8)),
        ctypes.cast(dBias, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(dD, ctypes.POINTER(ctypes.c_int8)),
        ctypes.c_int64(m),
        ctypes.c_int64(n),
        ctypes.c_int64(k),
        ctypes.c_float(scale_a),
        ctypes.c_int32(zp_a),
        ctypes.c_float(scale_b),
        ctypes.c_int32(zp_b),
        ctypes.c_float(scale_d),
        ctypes.c_int32(zp_d),
    )
    _LAUNCH_COUNT += 1


# --- minimal CUDA runtime bindings -------------------------------------------------

_CUDART: Optional[ctypes.CDLL] = None


def _load_cudart() -> ctypes.CDLL:
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            _CUDART = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if _CUDART is None:
        raise OSError("libcudart.so not found")
    return _CUDART


def _check(cudart: ctypes.CDLL, err: int, what: str) -> None:
    if err != 0:
        cudart.cudaGetErrorString.restype = ctypes.c_char_p
        msg = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"{what} failed ({err}): {msg!r}")


def _cuda_malloc(cudart: ctypes.CDLL, nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    _check(cudart, cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
    return ptr.value or 0


def _cuda_free(cudart: ctypes.CDLL, ptr: int) -> None:
    if ptr:
        _check(cudart, cudart.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")


def _cuda_memcpy_h2d(cudart: ctypes.CDLL, dst: int, src: np.ndarray) -> None:
    # cudaMemcpyHostToDevice = 1
    _check(
        cudart,
        cudart.cudaMemcpy(
            ctypes.c_void_p(dst),
            src.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(src.nbytes),
            ctypes.c_int(1),
        ),
        "cudaMemcpy H2D",
    )


def _cuda_memcpy_d2h(cudart: ctypes.CDLL, dst: np.ndarray, src: int) -> None:
    # cudaMemcpyDeviceToHost = 2
    _check(
        cudart,
        cudart.cudaMemcpy(
            dst.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(src),
            ctypes.c_size_t(dst.nbytes),
            ctypes.c_int(2),
        ),
        "cudaMemcpy D2H",
    )


def _cuda_device_sync(cudart: ctypes.CDLL) -> None:
    _check(cudart, cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

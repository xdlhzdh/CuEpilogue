"""Static TIR PrimFunc analysis: shapes, category, FLOPs estimate."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from tvm.tirx.function import PrimFunc


@dataclass
class PrimFuncInfo:
    name: str
    shapes: list[tuple[int, ...]]
    category: str  # matmul | elementwise | reduce | other
    flops: float
    shape_key: str


_BUFFER_RE = re.compile(
    r"T\.Buffer\(\(([^)]+)\)",
    re.MULTILINE,
)
_INT_RE = re.compile(r"T\.int64\((\d+)\)|(\d+)")


def _parse_shape_tuple(inner: str) -> Optional[tuple[int, ...]]:
    dims: list[int] = []
    for m in _INT_RE.finditer(inner):
        dims.append(int(m.group(1) or m.group(2)))
    if not dims:
        return None
    return tuple(dims)


def buffer_shapes(func: PrimFunc) -> list[tuple[int, ...]]:
    """Collect concrete buffer shapes from PrimFunc TVMScript text."""
    shapes: list[tuple[int, ...]] = []
    buf_map = getattr(func, "buffer_map", None)
    if buf_map is not None:
        try:
            for _, buf in buf_map.items():
                shape = []
                ok = True
                for d in buf.shape:
                    v = int(d.value) if hasattr(d, "value") else int(d)
                    if v <= 0:
                        ok = False
                        break
                    shape.append(v)
                if ok and shape:
                    shapes.append(tuple(shape))
            if shapes:
                return shapes
        except Exception:  # noqa: BLE001
            shapes = []

    text = str(func)
    for m in _BUFFER_RE.finditer(text):
        parsed = _parse_shape_tuple(m.group(1))
        if parsed:
            shapes.append(parsed)
    return shapes


def classify_primfunc(name: str, shapes: list[tuple[int, ...]], func: PrimFunc) -> str:
    n = name.lower()
    if "matmul" in n or "dense" in n or "gemm" in n:
        return "matmul"
    if any(k in n for k in ("reduce", "sum", "norm", "softmax")):
        return "reduce"
    if any(k in n for k in ("add", "mul", "relu", "gelu", "exp", "cast", "elem")):
        return "elementwise"

    try:
        text = str(func)
        if 'remap("SSR"' in text or "remap('SSR'" in text:
            return "matmul"
        if "axis.R" in text or 'remap("SR"' in text:
            return "reduce"
    except Exception:  # noqa: BLE001
        pass
    return "other"


def estimate_flops(category: str, shapes: list[tuple[int, ...]]) -> float:
    """Rough FLOPs. Matmul uses 2*M*N*K when 2D buffers look like GEMM."""
    if category == "matmul":
        twod = [s for s in shapes if len(s) == 2]
        if len(twod) >= 2:
            a, b = twod[0], twod[1]
            if a[1] == b[0]:
                m, k, n = a[0], a[1], b[1]
                return float(2 * m * n * k)
            if a[0] == b[1]:
                m, k, n = a[1], a[0], b[0]
                return float(2 * m * n * k)
            # Identical square buffers A,B,C → treat as MxMxM
            if a == b and a[0] == a[1]:
                m = a[0]
                return float(2 * m * m * m)
            m, n = twod[-1]
            k = twod[0][1] if twod[0][0] == m else max(twod[0])
            return float(2 * m * n * k)
        if len(twod) == 1:
            m, n = twod[0]
            return float(2 * m * n * min(m, n))
        return 0.0

    if category in ("elementwise", "reduce", "other"):
        if not shapes:
            return 0.0
        vols = []
        for s in shapes:
            v = 1
            for d in s:
                v *= max(d, 1)
            vols.append(v)
        return float(max(vols) if vols else 0)

    return 0.0


def shape_key(shapes: list[tuple[int, ...]]) -> str:
    if not shapes:
        return ""
    twod = [s for s in shapes if len(s) == 2]
    if len(twod) >= 2 and twod[0][1] == twod[1][0]:
        m, k, n = twod[0][0], twod[0][1], twod[1][1]
        return f"{m}x{n}x{k}"
    if twod:
        return "x".join(str(d) for d in twod[0])
    return "|".join("x".join(str(d) for d in s) for s in shapes[:3])


def analyze_primfunc(name: str, func: PrimFunc) -> PrimFuncInfo:
    shapes = buffer_shapes(func)
    category = classify_primfunc(name, shapes, func)
    flops = estimate_flops(category, shapes)
    return PrimFuncInfo(
        name=name,
        shapes=shapes,
        category=category,
        flops=flops,
        shape_key=shape_key(shapes),
    )

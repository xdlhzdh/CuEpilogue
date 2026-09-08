"""Apply DLight rules only to PrimFuncs marked DLIGHT (subset schedule)."""
from __future__ import annotations

from typing import Iterable, Optional

import tvm
from tvm.s_tir import dlight as dl
from tvm.tirx.function import PrimFunc

from schedule_report import ScheduleReport


def default_gpu_rules():
    return (
        dl.gpu.Matmul(),
        dl.gpu.GEMV(),
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Fallback(),
    )


def extract_submodule(mod: tvm.IRModule, names: Iterable[str]) -> tvm.IRModule:
    name_set = set(names)
    funcs = {}
    for gv, func in mod.functions_items():
        if gv.name_hint in name_set and isinstance(func, PrimFunc):
            funcs[gv] = func
    return tvm.IRModule(funcs)


def merge_primfuncs(dst: tvm.IRModule, src: tvm.IRModule) -> tvm.IRModule:
    """Replace PrimFuncs in dst with those from src (by GlobalVar name_hint)."""
    by_name = {gv.name_hint: func for gv, func in src.functions_items()}
    updates = {}
    for gv, func in dst.functions_items():
        if gv.name_hint in by_name:
            updates[gv] = by_name[gv.name_hint]
        else:
            updates[gv] = func
    return tvm.IRModule(updates, dst.attrs)


def apply_dlight_to_funcs(
    mod: tvm.IRModule,
    target: tvm.target.Target,
    dlight_names: list[str],
    report: Optional[ScheduleReport] = None,
) -> tvm.IRModule:
    """Schedule only the named PrimFuncs with DLight; leave others untouched."""
    if not dlight_names:
        return mod

    sub = extract_submodule(mod, dlight_names)
    if not list(sub.functions_items()):
        return mod

    with target:
        scheduled = dl.ApplyDefaultSchedule(*default_gpu_rules())(sub)

    out = merge_primfuncs(mod, scheduled)
    if report is not None:
        for name in dlight_names:
            # dlight_skipped_search already set at decision time; ensure True
            for row in report.rows:
                if row.name == name and row.strategy == "DLIGHT":
                    row.dlight_skipped_search = True
    return out

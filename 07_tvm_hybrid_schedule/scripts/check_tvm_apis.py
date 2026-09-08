#!/usr/bin/env python3
"""Probe TVM 0.26 APIs required by stage-7 hybrid scheduling."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import tvm
        from tvm import relax
        from tvm.s_tir import dlight as dl
        from tvm.s_tir.meta_schedule.database import JSONDatabase
        from tvm.tirx.function import PrimFunc
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: import error: {exc}")
        return 1

    checks = [
        ("tvm version", getattr(tvm, "__version__", "?")),
        ("dlight.ApplyDefaultSchedule", hasattr(dl, "ApplyDefaultSchedule")),
        ("dlight.gpu.Matmul", hasattr(dl.gpu, "Matmul")),
        ("MetaScheduleTuneIRMod", hasattr(relax.transform, "MetaScheduleTuneIRMod")),
        ("MetaScheduleApplyDatabase", hasattr(relax.transform, "MetaScheduleApplyDatabase")),
        ("LegalizeOps", hasattr(relax.transform, "LegalizeOps")),
        ("JSONDatabase", JSONDatabase is not None),
        ("PrimFunc", PrimFunc is not None),
    ]
    ok = True
    for name, val in checks:
        print(f"  {name}: {val}")
        if val is False:
            ok = False
    if not ok:
        print("FAIL: missing required APIs (consider building TVM from source)")
        return 1
    print("OK: stage-7 TVM APIs available")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Benefit analyzer: assign DLIGHT / METASCHEDULE per PrimFunc (mutually exclusive)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import tvm
from tvm.tirx.function import PrimFunc

from schedule_config import ScheduleConfig
from tir_analysis import PrimFuncInfo, analyze_primfunc

STRATEGY_ATTR = "cu_epilogue.schedule_strategy"


class ScheduleStrategy(str, Enum):
    BYOC = "BYOC"
    DLIGHT = "DLIGHT"
    METASCHEDULE = "METASCHEDULE"


@dataclass
class Decision:
    name: str
    strategy: ScheduleStrategy
    reason: str
    info: PrimFuncInfo


def _name_matches(name: str, patterns: list[str]) -> bool:
    n = name.lower()
    for p in patterns:
        if p.lower() in n:
            return True
    return False


def decide_one(info: PrimFuncInfo, config: ScheduleConfig) -> Decision:
    """Apply force_mode → whitelist/blacklist → automatic heuristics."""
    name = info.name

    if config.force_mode == "dlight":
        return Decision(name, ScheduleStrategy.DLIGHT, "force_mode=dlight", info)
    if config.force_mode == "metaschedule":
        return Decision(
            name, ScheduleStrategy.METASCHEDULE, "force_mode=metaschedule", info
        )

    # Blacklist forces DLight (checked before whitelist so ops can be pinned down)
    if _name_matches(name, config.blacklist):
        return Decision(name, ScheduleStrategy.DLIGHT, "blacklist", info)

    if _name_matches(name, config.whitelist):
        return Decision(name, ScheduleStrategy.METASCHEDULE, "whitelist", info)

    # Automatic: any of these → MetaSchedule
    if info.flops > config.flops_threshold:
        return Decision(
            name,
            ScheduleStrategy.METASCHEDULE,
            f"flops={info.flops:.0f}>{config.flops_threshold:.0f}",
            info,
        )

    if info.shape_key and info.shape_key in config.irregular_shapes:
        return Decision(
            name, ScheduleStrategy.METASCHEDULE, f"irregular_shape={info.shape_key}", info
        )

    if _name_matches(name, config.hotspot_ops):
        return Decision(name, ScheduleStrategy.METASCHEDULE, "hotspot_ops", info)

    # Default: small / elementwise / reduce → DLight
    if info.category in ("elementwise", "reduce"):
        return Decision(
            name, ScheduleStrategy.DLIGHT, f"category={info.category}", info
        )

    return Decision(
        name,
        ScheduleStrategy.DLIGHT,
        f"default (flops={info.flops:.0f}<={config.flops_threshold:.0f})",
        info,
    )


def iter_primfuncs(mod: tvm.IRModule) -> Dict[str, PrimFunc]:
    out: Dict[str, PrimFunc] = {}
    for gv, func in mod.functions_items():
        if isinstance(func, PrimFunc):
            out[gv.name_hint] = func
    return out


def decide_all_primfuncs(
    mod: tvm.IRModule, config: ScheduleConfig
) -> Dict[str, Decision]:
    decisions: Dict[str, Decision] = {}
    for name, func in iter_primfuncs(mod).items():
        info = analyze_primfunc(name, func)
        decisions[name] = decide_one(info, config)
    return decisions


def attach_strategy_attrs(
    mod: tvm.IRModule, decisions: Dict[str, Decision]
) -> tvm.IRModule:
    """Tag each PrimFunc with cu_epilogue.schedule_strategy."""
    updates = {}
    for gv, func in mod.functions_items():
        name = gv.name_hint
        if name in decisions and isinstance(func, PrimFunc):
            tagged = func.with_attr(STRATEGY_ATTR, decisions[name].strategy.value)
            updates[gv] = tagged
        else:
            updates[gv] = func
    return tvm.IRModule(updates, mod.attrs)

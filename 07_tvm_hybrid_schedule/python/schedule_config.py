"""Load hybrid-schedule TOML config with optional CLI overrides."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class ScheduleConfig:
    force_mode: str = "auto"  # auto | dlight | metaschedule
    flops_threshold: float = 50_000_000.0
    hotspot_ratio_threshold: float = 0.10
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    irregular_shapes: list[str] = field(default_factory=list)
    hotspot_ops: list[str] = field(default_factory=list)
    max_trials_global: int = 16
    max_trials_per_task: int = 16
    enable_byoc_stage6: bool = True
    work_dir: str = "docs/work_dirs/default"

    def validate(self) -> None:
        if self.force_mode not in ("auto", "dlight", "metaschedule"):
            raise ValueError(f"invalid force_mode: {self.force_mode}")
        if self.flops_threshold < 0:
            raise ValueError("flops_threshold must be >= 0")


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "default_schedule.toml"


def load_config(path: Optional[str | Path] = None, **overrides: Any) -> ScheduleConfig:
    cfg_path = Path(path) if path else _default_config_path()
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open("rb") as f:
            data = tomllib.load(f) or {}

    cfg = ScheduleConfig(
        force_mode=str(data.get("force_mode", "auto")),
        flops_threshold=float(data.get("flops_threshold", 50_000_000)),
        hotspot_ratio_threshold=float(data.get("hotspot_ratio_threshold", 0.10)),
        whitelist=list(data.get("whitelist", []) or []),
        blacklist=list(data.get("blacklist", []) or []),
        irregular_shapes=list(data.get("irregular_shapes", []) or []),
        hotspot_ops=list(data.get("hotspot_ops", []) or []),
        max_trials_global=int(data.get("max_trials_global", 16)),
        max_trials_per_task=int(data.get("max_trials_per_task", 16)),
        enable_byoc_stage6=bool(data.get("enable_byoc_stage6", True)),
        work_dir=str(data.get("work_dir", "docs/work_dirs/default")),
    )
    if overrides:
        # Filter None overrides so CLI defaults don't wipe config.
        clean = {k: v for k, v in overrides.items() if v is not None}
        cfg = replace(cfg, **clean)
    cfg.validate()
    return cfg

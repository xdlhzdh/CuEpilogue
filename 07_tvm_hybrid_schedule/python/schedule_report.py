"""Schedule decision report: JSON (+ optional CSV)."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from schedule_decision import Decision, ScheduleStrategy


@dataclass
class ReportRow:
    name: str
    shapes: list[Any]
    flops: float
    strategy: str
    reason: str
    ms_trials: Optional[int] = None
    tune_seconds: Optional[float] = None
    best_latency_ms: Optional[float] = None
    ms_cache_hit: Optional[bool] = None
    dlight_skipped_search: Optional[bool] = None


@dataclass
class ScheduleReport:
    rows: list[ReportRow] = field(default_factory=list)
    compile_seconds: Optional[float] = None
    work_dir: Optional[str] = None

    def record_decision(self, decision: Decision) -> None:
        info = decision.info
        row = ReportRow(
            name=decision.name,
            shapes=[list(s) for s in info.shapes],
            flops=info.flops,
            strategy=decision.strategy.value,
            reason=decision.reason,
        )
        if decision.strategy == ScheduleStrategy.DLIGHT:
            row.dlight_skipped_search = True
        self.rows.append(row)

    def record_byoc(self, name: str, shapes: list[Any], reason: str = "stage6_cutlass") -> None:
        self.rows.append(
            ReportRow(
                name=name,
                shapes=shapes,
                flops=0.0,
                strategy=ScheduleStrategy.BYOC.value,
                reason=reason,
            )
        )

    def update_ms(
        self,
        name: str,
        *,
        trials: Optional[int] = None,
        tune_seconds: Optional[float] = None,
        best_latency_ms: Optional[float] = None,
        cache_hit: Optional[bool] = None,
    ) -> None:
        for row in self.rows:
            if row.name == name and row.strategy == ScheduleStrategy.METASCHEDULE.value:
                if trials is not None:
                    row.ms_trials = trials
                if tune_seconds is not None:
                    row.tune_seconds = tune_seconds
                if best_latency_ms is not None:
                    row.best_latency_ms = best_latency_ms
                if cache_hit is not None:
                    row.ms_cache_hit = cache_hit
                return

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_seconds": self.compile_seconds,
            "work_dir": self.work_dir,
            "operators": [asdict(r) for r in self.rows],
        }

    def write_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            p.write_text("", encoding="utf-8")
            return
        fields = list(asdict(self.rows[0]).keys())
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(asdict(r))

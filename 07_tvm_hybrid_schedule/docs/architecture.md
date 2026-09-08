# Stage 7 architecture — hybrid DLight / MetaSchedule

## Why this stage exists

Stage 6 binds a **specific** fused INT8 kernel via BYOC (`call_dps_packed`).
Stage 7 schedules the **remaining** TIR PrimFuncs after Relax legalization:

- **DLight**: fast rule-based tiling (`ApplyDefaultSchedule`), no search.
- **MetaSchedule**: search + hardware measure (`MetaScheduleTuneIRMod` + DB).

Constraint: **one PrimFunc → one strategy**. BYOC has highest priority and never
enters the DLight/MS decision loop.

## Pass / function order

```text
Relax IRModule
  → get_pipeline("zero")
  → maybe stage6 apply_pipeline          # BYOC
  → LegalizeOps                          # PrimFuncs
  → decide_all_primfuncs + with_attr     # NEW decision layer
  → apply_dlight_to_funcs(subset)        # only DLIGHT names
  → MetaScheduleTuneIRMod(op_names=MS)   # only METASCHEDULE names
  → MetaScheduleApplyDatabase
```

There is **no** MLIR `.mlir` input. Graphs are built with Relax `BlockBuilder`
(same style as stage 6).

## Mutual exclusion implementation

| Strategy | Mechanism |
| --- | --- |
| DLight | Extract PrimFunc submodule by name → `ApplyDefaultSchedule` → merge back |
| MetaSchedule | `MetaScheduleTuneIRMod(..., op_names=[...])` (not `MetaScheduleTuneTIR`, which tunes everything) |
| BYOC | Stage-6 rewrite removes QDQ from TIR path before LegalizeOps consumes it as TIR |

## Decision rules (auto)

1. `force_mode` global override
2. `blacklist` → DLight; `whitelist` → MetaSchedule
3. Else MetaSchedule if FLOPs > threshold, irregular shape, or hotspot name
4. Else DLight (small / elementwise / reduce / default)

`hotspot_ratio_threshold` is reserved for a future runtime profiler; today
hotspots are static name lists in TOML.

## Relation to stage 6

| | Stage 6 | Stage 7 |
| --- | --- | --- |
| Focus | Pattern → packed CUTLASS `.so` | TIR tiling policy mix |
| External call | `cu_epilogue.qgemm_bias_gelu` | optional reuse of stage 6 |
| Kernel authoring | No (bind existing `.so`) | No (reuse DLight / MS) |

## Observability

`ScheduleReport` writes `schedule_report.json` / `.csv` under `work_dir` with
strategy, reason, FLOPs, MS trials / tune seconds / cache hit, and
`dlight_skipped_search=true` for DLight ops.

# 阶段七验收与基准报告 — DLight + MetaSchedule 混合调度

## 范围

- **编译路径**：Relax IRModule →（可选）阶段六 CUTLASS BYOC → `LegalizeOps` → 按 **PrimFunc** 打策略标签（`DLIGHT` / `METASCHEDULE`）→ 子集走 DLight / 子集走 MetaSchedule → `MetaScheduleApplyDatabase`。
- **约束**：同一 PrimFunc 只能选一种策略（互斥）。已被 BYOC 匹配的 QDQ 子图 **不进入** DLight/MetaSchedule。
- **本报告主要测什么**：编译期策略分流（谁走 DLight、谁走 MetaSchedule）、调优耗时、数据库缓存命中，以及决策/流水线正确性——**不能替代**阶段六「融合 `.so` vs NumPy」的 kernel 延迟对比。
- **TVM**：apache-tvm 0.26.0（复用阶段六 venv）。关键 API：`tvm.s_tir.dlight`、`relax.transform.MetaScheduleTuneIRMod`、`MetaScheduleApplyDatabase`、`JSONDatabase`。
- **硬件**：MetaSchedule 调优/回放需要 CUDA；仅 DLight / 决策类测试可不依赖 GPU。

> 阶段六回答的是：「BYOC 是否绑上 INT8 `.so`、这次调用有多快？」  
> 阶段七回答的是：「对 **剩余** TIR 算子，DLight / MetaSchedule 分得对不对、编译成本怎样？」

## 环境

```bash
./06_tvm_cutlass_byoc/scripts/setup_tvm.sh
source 06_tvm_cutlass_byoc/.venv/bin/activate
export PYTHONPATH=$PWD/07_tvm_hybrid_schedule/python:$PWD/06_tvm_cutlass_byoc/python:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/05_qdq_fusion:$LD_LIBRARY_PATH

python 07_tvm_hybrid_schedule/scripts/check_tvm_apis.py
```

可选：用与阶段六相同的 CMake 开关注册 ctest：

```bash
cmake -B build -DCU_EPILOGUE_ENABLE_TVM=ON
cmake --build build -j$(nproc)
```

| 项 | 说明 |
| --- | --- |
| 主机 / GPU | 跑完后填写（例如 Tesla V100） |
| TVM | 项目 venv 中的 0.26.x |
| 配置 | `config/default_schedule.toml`（`flops_threshold`、`force_mode`、`max_trials_*`、`enable_byoc_stage6` 等） |
| 工作目录 / MS 库 | `docs/work_dirs/...`（JSON 调优库写在 work_dir 下） |

## 正确性（pytest / ctest）

| 里程碑 | 文件 | 检查内容 |
| --- | --- | --- |
| **M1** | `tests/test_m1_decision.py` | FLOPs / elementwise / `force_mode` / 白黑名单 → 策略 |
| **M2** | `tests/test_m2_dlight.py` | DLight 只作用于指定子集；其它 PrimFunc 不被改写 |
| **M3** | `tests/test_m3_metaschedule.py` | 调优 + 数据库回放 + 缓存命中（**需要 CUDA**） |
| **M4** | `tests/test_m4_hybrid_e2e.py` | 仅 DLight；混合 MetaSchedule；可选阶段六 BYOC 旁路 |

```bash
pytest 07_tvm_hybrid_schedule/tests -q
# 或：
cd build && ctest -R stage7 --output-on-failure
# 或：
./07_tvm_hybrid_schedule/scripts/run_pytest.sh
```

| 条件 | 预期 |
| --- | --- |
| 无 TVM / DLight | `run_pytest.sh` 以 0 退出（SKIP） |
| 无 CUDA | 跑 M1/M2 及部分 M4；M3 与偏重 MS 的 M4 跳过 |
| 有 CUDA 且阶段五 `.so` 可用 | 全量用例（含 BYOC 分支） |

## 编译耗时对比

混合调度目标：只对值得搜索的算子走 MetaSchedule，其余走 DLight——编译应比「全部 MetaSchedule」更省，对热点 matmul 又好于「全部 DLight」。

```bash
# 仓库根目录；venv 与 PYTHONPATH 同上
python 07_tvm_hybrid_schedule/scripts/bench_hybrid.py --max-trials 16 \
  --work-root 07_tvm_hybrid_schedule/docs/work_dirs/bench
```

在真实 GPU 上跑完后填写：

| 模式 | 编译秒数 | 出现的策略 | 说明 |
| --- | ---: | --- | --- |
| 仅 DLight（`force_mode=dlight`） | _(填写)_ | `DLIGHT` | 无 MetaSchedule 搜索 |
| 仅 MetaSchedule（`force_mode=metaschedule`） | _(填写)_ | `METASCHEDULE` | 需 CUDA；耗时随 `max_trials` 上升 |
| 混合自动（`force_mode=auto`） | _(填写)_ | 混合 | 由 FLOPs / 名单等决定分流 |

无 CUDA 时，`bench_hybrid.py` 会提示并只跑 DLight / 跳过 MS 调优的混合路径。

### 单模块 CLI 与 JSON/CSV 报表

```bash
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind hybrid \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/demo \
  --force-mode auto

# 强制全部 DLight（冒烟）：
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind hybrid \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/demo_dlight \
  --force-mode dlight
```

`--work-dir` 下产物：

- `schedule_report.json` / `schedule_report.csv`：每个算子的策略、原因、FLOPs、MS trial 数 / 调优秒数 / `ms_cache_hit`、`dlight_skipped_search`

### 决策层冒烟示例（示意）

来自偏 DLight 的冒烟 `schedule_report.json`（示例；请在本机重跑后替换数值）：

| 名称 | Shape / FLOPs | 策略 | 原因 | MS 缓存命中 | 调优秒数 |
| --- | --- | --- | --- | --- | ---: |
| `fused_add_relu` | 64×32 / 2048 | `DLIGHT` | `category=elementwise` | — | — |
| `matmul` | （视规模） | `DLIGHT` 或 `METASCHEDULE` | FLOPs 相对 `flops_threshold`（默认 50M） | 第二次同 work_dir 编译可为 true | _(填写)_ |

数据库复用：使用 **同一** `work_dir` 再编译一次时，MetaSchedule 算子应出现 `ms_cache_hit=true`（跳过搜索）。

## 与 BYOC 的衔接

```bash
# 形似 QDQ 的模块：可能走阶段六 BYOC，该子图不再进 TIR 混合调度
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind qdq \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/byoc_demo
```

仅当 `enable_byoc_stage6=true`（TOML 默认开启）且模块像阶段六 QDQ 链时，报表才会出现 BYOC 行。这些算子 **不会** 再打 DLight/MS 标签。

## 与阶段六延迟指标的边界

| | 阶段六 | 阶段七 |
| --- | --- | --- |
| 主指标 | 融合 `.so` vs NumPy 端到端 ms（如 4.57× / 11.2×） | 编译策略混合 + 决策可观测性 |
| Kernel 来源 | 绑定阶段五 `.so` | 复用 DLight / MetaSchedule（不新写 CUTLASS） |
| Profile（nsys/ncu） | 已记录 RowSum+Gemm+ColSum | 非本阶段重点；若需可对调度后的 TIR 另做 |

融合 INT8 调用的延迟 / DRAM 请继续看 [阶段六报告](../../06_tvm_cutlass_byoc/docs/benchmarking_report.md)。

## 架构说明

流水线顺序、互斥机制与决策规则见 [architecture.md](architecture.md)。

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `python/schedule_decision.py` | 收益判定与打标 |
| `python/dlight_scheduler.py` | 子集 DLight |
| `python/metaschedule_scheduler.py` | 子集 MetaSchedule 调优与数据库 |
| `python/hybrid_pipeline.py` | 完整 Pass 顺序 |
| `python/schedule_report.py` | 写出 JSON/CSV |
| `scripts/bench_hybrid.py` | DLight / MS / 混合编译耗时对比 |
| `config/default_schedule.toml` | 阈值、名单、trial 数、BYOC 开关 |

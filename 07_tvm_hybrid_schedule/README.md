# 阶段七 — TVM DLight + MetaSchedule 混合调度

本阶段不手工实现新 kernel。Legalize 后每个剩余 TIR `PrimFunc` 仅选择 **DLight** 或 **MetaSchedule** 之一。  
阶段六 CUTLASS BYOC 优先：匹配到的 QDQ 子图直接旁路，不进入 TIR 调度。

```text
构造 Relax 图
  → get_pipeline("zero")
  → [可选] 阶段六 BYOC：QDQ → call_dps_packed(.so)     ← 旁路，不进 TIR
  → LegalizeOps                                         ← 剩余算子变成 PrimFunc
  → 收益判定 → 属性 cu_epilogue.schedule_strategy
  → 子集 DLight：ApplyDefaultSchedule
  → 子集 MetaSchedule：TuneIRMod(op_names=...) + ApplyDatabase
  → [CLI 默认] relax.build(target=nvidia/nvidia-v100)   ← arch=sm_70，CodegenCUDA + nvcc
  → [若 cuda(0) 存在] Relax VM launch
```

测得的数字见 [docs/hybrid_schedule_report.md](docs/hybrid_schedule_report.md)。架构：[docs/architecture.md](docs/architecture.md)。

本阶段主要评估 **策略分流正确性与编译开销**，不以 kernel 延迟为主指标。融合 INT8 的 nsys/ncu 数据见阶段六。

---

## 环境准备

仓库根目录。复用阶段六 venv（TVM 0.26）：

```bash
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70 -DCU_EPILOGUE_ENABLE_TVM=ON
cmake --build build -j$(nproc)

./06_tvm_cutlass_byoc/scripts/setup_tvm.sh
source 06_tvm_cutlass_byoc/.venv/bin/activate
export CU_EPILOGUE_BUILD_DIR=$PWD/build
export PYTHONPATH=$PWD/07_tvm_hybrid_schedule/python:$PWD/06_tvm_cutlass_byoc/python:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/05_qdq_fusion:$LD_LIBRARY_PATH

python 07_tvm_hybrid_schedule/scripts/check_tvm_apis.py
# 期望：OK: stage-7 TVM APIs available
```

`check_tvm_apis.py` 确认：`dlight.ApplyDefaultSchedule`、`MetaScheduleTuneIRMod`、`MetaScheduleApplyDatabase`、`JSONDatabase`、`LegalizeOps`。

---

## 策略决策

入口是 `hybrid_pipeline.apply_hybrid_pipeline()`（`python/hybrid_pipeline.py`）。Legalize 之后、真正调度之前：

```text
LegalizeOps()                          # Relax 算子 → 未调度的 TIR PrimFunc
  → decide_all_primfuncs(mod, config)  # python/schedule_decision.py
       对每个 PrimFunc:
         analyze_primfunc()            # python/tir_analysis.py：形状 / 类别 / FLOPs
         decide_one(info, config)      # 选出 DLIGHT 或 METASCHEDULE
  → attach_strategy_attrs()            # PrimFunc.with_attr("cu_epilogue.schedule_strategy", ...)
  → 按 strategy 拆成 dlight_names / ms_names
  → apply_dlight_to_funcs(dlight_names)
  → tune_metaschedule_funcs(ms_names) + apply_database()
```

决策是纯静态 Python 逻辑，仅使用 PrimFunc 名称、buffer 形状和 TOML 配置；不编译、不执行 GPU kernel，也不读取 profiler 数据。因此该决策开销较低，但属于启发式判断。

### 第一步：`analyze_primfunc` → `PrimFuncInfo`

`python/tir_analysis.py`。对单个 PrimFunc 抽出 5 个字段：

| 字段 | 计算方式 |
| --- | --- |
| `name` | PrimFunc 的 `GlobalVar.name_hint`（Legalize/融合后的名字，如 `matmul`、`fused_add_relu`） |
| `shapes` | `buffer_shapes()`：优先读 `func.buffer_map` 里每个 buffer 的静态 shape；有动态维（`int(d)` 失败或 ≤0）就整体回退到正则扫 TVMScript 文本里的 `T.Buffer((...))` |
| `category` | `classify_primfunc()`：见下 |
| `flops` | `estimate_flops(category, shapes)`：见下 |
| `shape_key` | `shape_key()`：两个 2D buffer 且 `twod[0][1] == twod[1][0]` → `"MxNxK"`；否则取第一个 2D 的 `"dxd"`；再否则拼前 3 个 shape。用于 `irregular_shapes` 精确匹配 |

**`category` 只有 4 个取值**，判定顺序固定（先名字子串，再 TIR 文本兜底）：

| 顺序 | 条件 | 结果 |
| --- | --- | --- |
| 1 | 名字含 `matmul` / `dense` / `gemm` | `matmul` |
| 2 | 名字含 `reduce` / `sum` / `norm` / `softmax` | `reduce` |
| 3 | 名字含 `add` / `mul` / `relu` / `gelu` / `exp` / `cast` / `elem` | `elementwise` |
| 4 | TIR 文本含 `remap("SSR"` | `matmul` |
| 5 | TIR 文本含 `axis.R` 或 `remap("SR"` | `reduce` |
| 6 | 都不命中 | `other` |

因为是 **子串匹配且 matmul 优先**，融合算子的分类由名字里第一个命中的关键字决定：`fused_add_relu` → `elementwise`，而假设出现 `fused_matmul_add` 会判成 `matmul`。

**`flops` 是量级估计，不是精确 FLOPs**：

- `category == "matmul"`：根据 2D buffer 推断 `M/N/K`，返回 `2*M*N*K`。依次检查「`a[1]==b[0]`」→「`a[0]==b[1]`（转置）」→「`a==b` 且为方阵 → `2*m³`」；兜底路径将最后一个 2D buffer 视为输出，并从第一个 2D buffer 推断 K。仅有一个 2D buffer 时估算为 `2*m*n*min(m,n)`。
- `elementwise` / `reduce` / `other`：返回 **最大 buffer 的元素个数**（各维相乘取最大值），量纲上是「访存元素数」而非乘加次数。
- 无法提取形状时返回 `0.0`，自动决策将落入 DLight 默认路径。

### 第二步：`decide_one` 决策优先级

配置见 [config/default_schedule.toml](config/default_schedule.toml)。规则按顺序匹配，首个命中项立即返回；`reason` 记录命中的规则：

| 优先级 | 条件 | strategy | `reason` 实际取值 |
| --- | --- | --- | --- |
| 1 | `force_mode == "dlight"` | DLIGHT | `force_mode=dlight` |
| 1 | `force_mode == "metaschedule"` | METASCHEDULE | `force_mode=metaschedule` |
| 2 | 名字子串命中 `blacklist` | DLIGHT | `blacklist` |
| 3 | 名字子串命中 `whitelist` | METASCHEDULE | `whitelist` |
| 4 | `flops > flops_threshold`（默认 **50M**） | METASCHEDULE | `flops=524288>100000` |
| 5 | `shape_key` 命中 `irregular_shapes`（**精确**相等） | METASCHEDULE | `irregular_shape=128x128x128` |
| 6 | 名字子串命中 `hotspot_ops` | METASCHEDULE | `hotspot_ops` |
| 7 | `category` 是 `elementwise` / `reduce` | DLIGHT | `category=elementwise` |
| 8 | 兜底（`matmul` / `other` 且未超阈值） | DLIGHT | `default (flops=33554432<=50000000)` |

边界条件：

- **blacklist 先于 whitelist**，且两者均先于 FLOPs 阈值，因此名称级强制规则优先于自动启发式。
- `irregular_shapes` 是 `shape_key in config.irregular_shapes` 的**精确字符串**比较，不是子串；名字类的 `whitelist` / `blacklist` / `hotspot_ops` 才是子串（`_name_matches`，大小写不敏感）。
- `decide_one` 不返回 `BYOC`。BYOC 子图在 Phase 1 已替换为 `call_dps_packed`，不会生成供 `decide_all_primfuncs` 遍历的 PrimFunc；报表中的 `BYOC` 行由 `report.record_byoc()` 单独写入（`reason=stage6_cutlass`）。

### `Decision` 数据结构

```python
@dataclass
class Decision:
    name: str                  # PrimFunc GlobalVar 名
    strategy: ScheduleStrategy # BYOC | DLIGHT | METASCHEDULE（决策只产出后两种）
    reason: str                # 命中的那一档，见上表
    info: PrimFuncInfo         # 决策依据：shapes / category / flops / shape_key
```

`decide_all_primfuncs` 返回 `Dict[str, Decision]`（key = PrimFunc 名，逐个独立决策），随后被消费三次：

| 消费方 | 用了 `Decision` 的哪部分 |
| --- | --- |
| `attach_strategy_attrs()` | `strategy.value` → `PrimFunc.with_attr("cu_epilogue.schedule_strategy", ...)` |
| `dlight_names` / `ms_names` 拆分 | `strategy`，决定这个 PrimFunc 进哪条调度路径 |
| `report.record_decision()` | `name` / `strategy` / `reason` + `info.shapes` / `info.flops` → `schedule_report.json` 的一行；DLIGHT 行额外置 `dlight_skipped_search=true` |

`info.category` 和 `info.shape_key` 不直接写入报表，仅可能出现在 `reason` 中（如 `category=elementwise`）。MetaSchedule 记录的 `ms_trials`、`tune_seconds` 和 `ms_cache_hit` 在调优阶段由 `report.update_ms()` 回填，决策阶段均为 `null`。

属性 `cu_epilogue.schedule_strategy` 仅用于本仓库的策略标记与报表记录，TVM Pass 不读取该属性。互斥分流由 `dlight_names` 与 `MetaScheduleTuneIRMod(op_names=ms_names)` 实现。

`hotspot_ratio_threshold` 为未来 profiler 集成预留，当前实现未使用。

> 默认 `--kind hybrid` 包含 256³ matmul（约 33.5M FLOPs）与小型 elementwise。由于 33.5M 低于默认 50M 阈值，auto 模式下 matmul 也选择 DLight。可降低 `flops_threshold`（benchmark 使用 `100_000`）或增大形状以观察混合分流。

### 多算子 dataflow 的决策粒度

决策粒度是 **PrimFunc**，不是 Relax 层面的「算子」。一个 dataflow 里有 N 个不同类型算子时：

1. `get_pipeline("zero")` 和 `LegalizeOps` 会把 Relax 算子 lower 成 TIR PrimFunc。TVM 在此过程中可能做 **算子融合**（例如 `add` + `relu` 融合成 `fused_add_relu`），也可能不融合（大 matmul 单独一个 PrimFunc）。最终 IRModule 里有 **K 个 PrimFunc**（K ≤ N）。
2. `decide_all_primfuncs` 遍历这 K 个 PrimFunc，**每个独立决策**：对该 PrimFunc 做 `analyze_primfunc`（名字分类 + FLOPs 估算），然后 `decide_one` 走上面的优先级阶梯。
3. 不同 PrimFunc **可以走不同策略**。没有「整图统一」的约束。

示例：`build_hybrid_native_module(64, 64, 64, 16)`（M4 用例 `test_hybrid_mixed_strategies_with_ms`，`flops_threshold=100_000`）：

```text
Relax 图：
  main:     matmul(A[64,64], B[64,64])        → FP32
  elem_main: add(A[16,16], B[16,16]) → relu   → FP32

LegalizeOps 之后 IRModule 含 2 个 PrimFunc：
  "matmul"        category=matmul   flops=2*64*64*64 = 524288  > 100000 → METASCHEDULE
  "fused_add_relu" category=elementwise flops=256            < 100000 → DLIGHT
```

`schedule_report.json` 输出：

```json
{"operators": [
  {"name": "fused_add_relu", "strategy": "DLIGHT",       "reason": "category=elementwise", "flops": 256},
  {"name": "matmul",         "strategy": "METASCHEDULE", "reason": "flops=524288>100000",   "flops": 524288}
]}
```

如果算子更多（例如 matmul → bias_add → relu → layernorm → matmul → softmax），LegalizeOps 后可能产生 4–6 个 PrimFunc：每个独立走决策。大 matmul 超阈值 → MS，小 add/relu 被融合后 elementwise → DLight，layernorm（名字含 `norm`，分类为 reduce）→ DLight。TOML 的 `whitelist` / `blacklist` 可以按名字强制覆盖。

> 当前测试模块是 `relax_models.py` 中构造的 2–3 算子组合，不代表 LLM 规模 workload。决策逻辑仍会对任意数量的 PrimFunc 逐个独立执行，并在报表中记录各 PrimFunc 的分流结果。

---

## DLight 调度

实现：`python/dlight_scheduler.py` → `apply_dlight_to_funcs`。

1. 按 `dlight_names` 提取只包含目标 PrimFunc 的 submodule。
2. 在当前 `target` 下调用 TVM：

   ```python
   dl.ApplyDefaultSchedule(
       dl.gpu.Matmul(),
       dl.gpu.GEMV(),
       dl.gpu.Reduction(),
       dl.gpu.GeneralReduction(),
       dl.gpu.Fallback(),
   )(sub)
   ```

3. 将调度后的 PrimFunc 合并回原 IRModule。空名单时不修改模块。

`ApplyDefaultSchedule`（TVM `tvm.s_tir.dlight`）对每个**尚未**带 `tirx.is_scheduled=1` 的 PrimFunc：按上面规则 **从前往后试**，第一个 `apply()` 返回非 `None` 的就用，打上 `tirx.is_scheduled`，**不再搜索、不上 GPU 测延迟**。

各规则实际改的是 **TIR 循环嵌套**（`tvm.s_tir.Schedule`），不是写 `.cu`：

| 规则 | 匹配什么 | 调度动作（摘要） |
| --- | --- | --- |
| `gpu.Matmul` | 规约块可规范化为 `C[S,I,J] += A[S,I,K]*B[S,J,K]` | split/reorder；绑定 `blockIdx.{x,y,z}` / `threadIdx.{x,y}`；可选 shared-memory cache 和 vectorize。满足 Tensor Core 条件时进一步执行 WMMA tensorize（见 [Tensorize / WMMA](#tensorize-wmma-的触发条件)） |
| `gpu.GEMV` | 矩阵×向量 | 类似的 GPU 线程映射 |
| `gpu.Reduction` / `GeneralReduction` | 带归约轴 | 归约循环分解 + 线程绑定 |
| `gpu.Fallback` | 前面都不匹配 | fuse 空间轴，`split` 成 `[None, max_threads]`，`bind(bx,"blockIdx.x")`、`bind(tx,"threadIdx.x")` |

这些规则继承 `GPUScheduleRule`。当 `target.keys` 不包含 `"gpu"`（例如 `llvm`）时，`apply()` 返回 `None`，TIR 保持不变。因此 M2 仅在 CUDA target 上断言 `tirx.is_scheduled`。

调度后 PrimFunc 里会出现 `T.thread_binding(..., thread="blockIdx.x")` 这类注解。这仍然是 **TIR**，还不是 PTX / cubin。

---

## MetaSchedule 调优

实现：`python/metaschedule_scheduler.py`。

**调优** `tune_metaschedule_funcs`：

```python
relax.transform.MetaScheduleTuneIRMod(
    params={},
    work_dir=str(work_dir),
    max_trials_global=...,
    max_trials_per_task=...,
    op_names=list(ms_names),   # 仅调优指定 PrimFunc
)(mod)
```

TVM 内部完整的 tune 流程（`tune_relax` → `tune_tasks`）：

1. **提取 task**：`MetaScheduleExtractTask` 从 IRModule 抽出需要 tune 的 PrimFunc（被 `op_names` 过滤，只含 `ms_names`）。
2. **搜索空间**：`SpaceGenerator`（默认 `post-order-apply`）对每个 task 生成调度搜索空间——tile 大小、线程绑定、cache 位置、unroll 步长、vectorize 宽度等。比 DLight 的固定 heuristic 宽得多。
3. **搜索策略**：默认 `EvolutionarySearch`（进化搜索），由 `TaskScheduler`（默认 `gradient`）在多个 task 间分配 trial 预算。
4. **硬件测量**：
   - `Builder`（默认 `LocalBuilder`）把候选 TIR 编成 `.so`（`tvm.build` → nvcc）。
   - `Runner`（默认 `LocalRunner`）在本机 GPU 上 **加载 .so → 分配输入 → `time_evaluator` 多次运行 → 取中位延迟**。
   - 评测标准：**kernel 执行时间（秒）**。`RunnerResult` 记录每次 trial 的 `run_secs: List[float]`；TaskScheduler 选延迟最小的作为 best。
   - 默认 `EvaluatorConfig` 会多次重复执行取均值（减少噪声）。
5. **CostModel**（默认 `XGBModel`）：利用已采集的硬件测量数据训练 XGBoost 模型，指导后续 trial 采样，减少硬件测量次数。
6. **写入数据库**：每条 trial 的 trace + 延迟写入 `work_dir` 下两个文件：

   | 文件 | 内容 |
   | --- | --- |
   | `database_workload.json` | 每个 PrimFunc 的结构指纹（用于匹配） |
   | `database_tuning_record.json` | 每条 trial 的完整调度 trace（split/bind/cache 指令序列）+ 测得延迟 |

   两个文件共同构成 `JSONDatabase`。M3 测试验证数据库写入与缓存命中。

MetaSchedule 会在 GPU 上编译并运行候选 kernel。默认 `max_trials_global=16`，即全局最多执行 16 个 trial；`bench_hybrid.py --max-trials 4` 将预算降低为 4，用于快速验证。

搜索并非每次编译都执行；搜索与回放的触发条件见「开发阶段与部署阶段」。

**回放** `apply_database`：

```python
relax.transform.MetaScheduleApplyDatabase(work_dir=..., enable_warning=False)(mod)
```

根据数据库中的最优 trace 改写对应 PrimFunc（split/bind/cache），生成带线程绑定与已调度标记的 TIR。`TuneIRMod` 主要写入数据库，`ApplyDatabase` 负责将最优调度回放到 IR。

查看 tune 结果：

```bash
# work_dir 下产物
cat work_dir/database_tuning_record.json | python -m json.tool | head -40
# 每条记录包含：workload hash、调度 trace（指令序列）、run_secs（延迟数组）
```

---

## 开发阶段与部署阶段

MetaSchedule 搜索需要 GPU 编译并反复测量候选 kernel，通常不应在每次部署编译时执行。本仓库将搜索与结果回放拆分为两个 Pass，并通过 `work_dir` 下的 `JSONDatabase` 持久化调优结果：

| | 搜索（高开销，通常一次） | 回放（低开销，每次编译） |
| --- | --- | --- |
| Pass | `MetaScheduleTuneIRMod` | `MetaScheduleApplyDatabase` |
| 需要 GPU | **需要**（LocalBuilder + LocalRunner 执行 kernel） | 不需要（仅改写 TIR） |
| 对 `work_dir` | **写** trace + 延迟 | 只 **读** |
| 典型耗时 | 秒～分钟（`max_trials_global` 决定） | 毫秒级 |

调优产物由 `work_dir/database_workload.json` 和 `work_dir/database_tuning_record.json` 组成。开发环境完成调优后，可将这两个文件随构建输入一起交付；部署编译仅执行 `MetaScheduleApplyDatabase`，无需重新搜索。

### 两个独立的开关

`tune_metaschedule_funcs` 是否执行搜索由两种跳过机制决定，任一生效即不再搜索：

**① 显式开关 `skip_metaschedule_tune`（调用方指定）**

`apply_hybrid_pipeline(..., skip_metaschedule_tune=True)`，CLI 对应 `--skip-ms-tune`，用于部署编译阶段：

```python
if ms_names and not skip_metaschedule_tune:
    mod = tune_metaschedule_funcs(...)   # 开发：搜索 + 写库
    mod = apply_database(mod, work_dir)  # 再回放
elif ms_names and skip_metaschedule_tune:
    mod = apply_database(mod, work_dir)  # 部署：只回放，绝不搜索
    for name in ms_names:
        report.update_ms(name, cache_hit=True, trials=0, tune_seconds=0.0)
```

两条分支均调用 `apply_database`，差异仅在于是否先执行搜索。

**② 自动缓存命中 `_db_has_records(work_dir)`（函数内部自检）**

即使没加 `--skip-ms-tune`，`tune_metaschedule_funcs` 一进来也会先看库：

```python
cache_hit = _db_has_records(work_dir)   # JSONDatabase(allow_missing=True).get_all_tuning_records()
if cache_hit:
    report.update_ms(name, cache_hit=True, trials=0, tune_seconds=0.0)
    return mod                          # 直接返回，不搜索
```

判定方式：`JSONDatabase(work_dir, allow_missing=True)` 有 `get_all_tuning_records()` 就看条数 > 0；取不到就退化成「`database_tuning_record.json` 存在且大小 > 2 字节」。

### 数据库作用域

`JSONDatabase` **没有名字、没有命名空间**，它由两个文件路径唯一确定（TVM `json_database.py`）：

```python
if work_dir is not None:
    path_workload      = osp.join(work_dir, "database_workload.json")
    path_tuning_record = osp.join(work_dir, "database_tuning_record.json")
```

一个 `work_dir` 对应一个 JSONDatabase。CLI 的 `--work-dir` 或 TOML 中的 `work_dir` 决定数据库路径。

库里两张表，**跨所有算子共用**：

| 文件 | 一行是什么 |
| --- | --- |
| `database_workload.json` | 一个 **workload**：某个 PrimFunc 的 IRModule 结构指纹（默认 `module_equality="structural"`，即 `StructuralEqual` / `StructuralHash`） |
| `database_tuning_record.json` | 一条 **tuning record**：引用某个 workload 的 index + 调度 trace + `run_secs` |

一个 `work_dir` 可以保存多个算子的 workload。依次调优 matmul 和 conv 时，两者的 workload 与 tuning record 会写入同一组文件。

而 `get_all_tuning_records()` 的签名是 `() -> list[TuningRecord]`，**不接受任何参数**，返回整张 record 表（跨所有 workload）。`_db_has_records` 只判断它的条数 > 0，因此回答的是「这个 `work_dir` 里有没有人 tune 过**任何**东西」，而不是「有没有**我现在要编的这个算子**的记录」。

TVM 还提供按 workload 查询的 API，但当前实现未使用：

| API | 粒度 |
| --- | --- |
| `get_all_tuning_records()` | 整库，无参数 ← 现在用的 |
| `has_workload(mod)` | 单个算子，需传 IRModule 做结构匹配 |
| `query_tuning_record(mod, target, workload_name)` | 单个算子的最优记录，找不到返回 `None` |
| `len(db)` | 整库记录条数 |

要做到「精确到算子」，就得对每个 `ms_name` 抽出子模块分别调 `has_workload` / `query_tuning_record`。

文件大小兜底条件 `> 2` 用于排除空文件以及 `[]` / `{}`。`JSONDatabase(allow_missing=True)` 在构造时即可创建文件，因此不能仅根据文件是否存在判断是否已有调优记录。

### 缓存命中与重新调优

只要 `work_dir` 中存在调优记录，后续编译即判定为 cache hit 并直接回放；报表记录 `ms_cache_hit=true`、`ms_trials=0`、`tune_seconds=0.0`。M3 测试验证同一 `work_dir` 的第二次调用命中缓存。

想**重新** tune 只有两条路：换一个 `--work-dir`，或者手工删掉旧 `work_dir` 里的两个 `database_*.json`。本仓库 **没有** `--force-retune` 之类的开关。

```bash
# 开发：首次搜索（需要 GPU，写入数据库）
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind hybrid \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/demo --force-mode metaschedule --max-trials 16
#   matmul: METASCHEDULE ... cache=False    ← tune_seconds > 0

# 开发：同 work_dir 再编一次，自动 cache hit（不再搜索）
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind hybrid \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/demo --force-mode metaschedule
#   matmul: METASCHEDULE ... cache=True     ← tune_seconds = 0

# 部署：禁止搜索，仅使用已有数据库
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind hybrid \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/demo --force-mode metaschedule --skip-ms-tune
```

### 当前实现的限制

- **缓存检测是数据库级，而非 workload 级。** 在已调优 matmul 的 `work_dir` 中编译新算子时，只要数据库中已有任意记录，新算子也会被视为 cache hit，从而跳过调优：

  ```bash
  # 第 1 次：tune matmul，库里有了 matmul 的 workload + record
  build_and_run.py --kind matmul --work-dir wd/shared --force-mode metaschedule

  # 第 2 次：换成 elem 图，本该 tune 一个新算子
  build_and_run.py --kind elem --work-dir wd/shared --force-mode metaschedule
  #   _db_has_records(wd/shared) → True（那是 matmul 的记录！）→ 直接 return，新算子从未被 tune
  #   → ApplyDatabase 找不到匹配 workload，enable_warning=False 静默跳过
  #   → 报表显示 METASCHEDULE + cache=True，但 TIR 里没有任何 thread_binding
  ```

  多套形状或多个模型应使用独立 `work_dir`，避免已有记录阻止新 workload 进入调优。
- **ApplyDatabase 匹配失败不会报告警告。** 匹配依赖 workload 结构指纹；形状变化可能导致匹配失败。由于调用设置 `enable_warning=False`，对应 PrimFunc 保持原样，而报表仍显示 `METASCHEDULE`。可通过 TIR 中的 `thread_binding` 检查是否实际应用调度。
- **`--skip-ms-tune` 分支无条件记录 `ms_cache_hit=true`。** 当前实现未验证数据库中是否存在匹配记录；空库情况下报表仍显示 `cache=True`，但 IR 不会获得 MetaSchedule 调度。

### 部署边界

`apply_hybrid_pipeline` 在 ApplyDatabase 后返回已调度 TIR；CLI 随后默认执行 `relax.build`。产物形态见 [`relax.build` Codegen 与产物格式](#relaxbuild-codegen-与产物格式)。部署时可仅交付 `.so`，运行时无需携带 `work_dir`。

---

## `ScheduleReport` 与 MetaSchedule 数据库

`report` 是本仓库定义的观测对象（`python/schedule_report.py`），实现为 Python dataclass。TVM Pass 不读取该对象。它记录各算子的调度策略、决策原因及相关编译数据，供命令行展示、测试断言和文档报告使用。

### 与 MetaSchedule 数据库的关系

| | `ScheduleReport` | MetaSchedule `JSONDatabase` |
| --- | --- | --- |
| 谁写的 | 本仓库 `schedule_report.py` | TVM `MetaScheduleTuneIRMod` |
| 落盘文件 | `work_dir/schedule_report.json` + `.csv` | `work_dir/database_workload.json` + `database_tuning_record.json` |
| 内容 | 每个算子的**策略决策**（DLIGHT / METASCHEDULE / BYOC + 原因） | 每个 workload 的**调度 trace + 实测延迟** |
| 谁读它 | 人、`bench_hybrid.py`、M3/M4 测试 | `MetaScheduleApplyDatabase` |
| 影响编译结果吗 | **不影响**，纯旁路观测 | **影响**，回放的调度就来自这里 |
| 跨次编译累积吗 | 不，每次编译新建一个空 report | 是，记录持续追加 |

数据仅从数据库状态流向报表：`_db_has_records()` 的结果写入 `ms_cache_hit` / `tune_seconds`。报表不写数据库，也不修改 IR；删除报表不会影响编译结果。

### 字段写入阶段

| Phase | 调用 | 写入 |
| --- | --- | --- |
| 1 BYOC | `report.record_byoc(name, shapes, "stage6_cutlass")` | 追加一行 `strategy=BYOC` |
| 3 决策 | `report.record_decision(dec)`（每个 PrimFunc 一次） | 追加一行 `strategy=DLIGHT/METASCHEDULE`；DLIGHT 行置 `dlight_skipped_search=true` |
| 4 DLight | `apply_dlight_to_funcs` 内部 | 再确认一次 `dlight_skipped_search=true` |
| 4 MS | `report.update_ms(name, ...)` | **回填** `ms_trials` / `tune_seconds` / `ms_cache_hit` 到已有行 |
| 收尾 | `report.compile_seconds = ...` | `apply_hybrid_pipeline` 的墙钟时间：包含调度及 MetaSchedule 搜索/回放，不包含 CLI 后续的 `relax.build` 与 `export_library` |

`record_*` 追加记录，`update_ms` 更新已有记录。因此 BYOC 记录位于前部，其余记录按 IRModule 的 PrimFunc 遍历顺序排列。

### `ReportRow` 字段格式

```python
@dataclass
class ReportRow:
    name: str                              # 算子名
    shapes: list[Any]                      # buffer 形状，list of list
    flops: float
    strategy: str                          # "BYOC" | "DLIGHT" | "METASCHEDULE"
    reason: str                            # 命中的决策档位
    ms_trials: Optional[int] = None
    tune_seconds: Optional[float] = None
    best_latency_ms: Optional[float] = None
    ms_cache_hit: Optional[bool] = None
    dlight_skipped_search: Optional[bool] = None
```

各字段在三种 strategy 下的实际取值：

| 字段 | BYOC 行 | DLIGHT 行 | METASCHEDULE 行 |
| --- | --- | --- | --- |
| `name` | 固定 `cu_epilogue.qgemm_bias_gelu` | PrimFunc 名 | PrimFunc 名 |
| `shapes` | `[]`（当前 BYOC 元数据未采集形状） | `info.shapes` | `info.shapes` |
| `flops` | **恒为 `0.0`** | `info.flops` | `info.flops` |
| `reason` | `stage6_cutlass` | 如 `category=elementwise` | 如 `flops=524288>100000` |
| `ms_trials` / `tune_seconds` | `null` | `null` | 执行调优时为 `max_trials_global` / 秒数；cache hit 时为 `0` / `0.0` |
| `ms_cache_hit` | `null` | `null` | `true`（命中或 `--skip-ms-tune`）/ `false`（执行过搜索） |
| `dlight_skipped_search` | `null` | `true` | `null` |
| `best_latency_ms` | `null` | `null` | 当前实现为 `null` |

JSON 顶层包含 `compile_seconds`、`work_dir` 和由 `ReportRow` 序列化得到的 `operators`：

```json
{
  "compile_seconds": 0.0227,
  "work_dir": "docs/work_dirs/default",
  "operators": [
    {"name": "fused_add_relu", "shapes": [[32,32],[32,32],[32,32]], "flops": 1024.0,
     "strategy": "DLIGHT", "reason": "category=elementwise",
     "ms_trials": null, "tune_seconds": null, "best_latency_ms": null,
     "ms_cache_hit": null, "dlight_skipped_search": true},
    {"name": "matmul", "shapes": [[256,256],[256,256],[256,256]], "flops": 33554432.0,
     "strategy": "DLIGHT", "reason": "default (flops=33554432<=50000000)",
     "ms_trials": null, "tune_seconds": null, "best_latency_ms": null,
     "ms_cache_hit": null, "dlight_skipped_search": true}
  ]
}
```

CSV 是同样内容的扁平表，表头取 `asdict(rows[0]).keys()`（`compile_seconds` / `work_dir` **不在 CSV 里**，只在 JSON 里）。

### 报表的作用

报表用于评估策略分流与编译开销，是本阶段的主要可观测产物：

| 消费者 | 用了什么 |
| --- | --- |
| 人 / CLI 终端 | `build_and_run.py` 逐行打印 `name: strategy reason=... flops=... cache=...` |
| `bench_hybrid.py` | `compile_seconds`、`{r.strategy}` 集合、`len(rows)` → 三侧对比表 |
| **M4** `test_m4_hybrid_e2e.py` | 断言每个名称仅出现一次、`strategy` 取值合法、`compile_seconds is not None`，并验证 JSON 写出 |
| **M3** `test_m3_metaschedule.py` | 断言二次调用后 `any(r.ms_cache_hit)` |
| `docs/hybrid_schedule_report.md` | 引用报表中的测量结果 |

当前测试通过报表中每个名称仅出现一次，验证 DLight 与 MetaSchedule 的互斥分流。

### 字段限制

- **`best_latency_ms` 当前始终为 `null`。** `update_ms()` 和 `ReportRow` 已定义该字段，但当前调用方未传入数值。MetaSchedule 实测延迟保存在 `database_tuning_record.json` 的 `run_secs` 中，尚未汇总到报表。
- **`update_ms` 按「name + strategy=='METASCHEDULE'」匹配，命中第一行就 return。** 如果某个名字被记成了 DLIGHT 或 BYOC，MS 侧的回填会静默失败（找不到行，函数直接走完）。
- **BYOC 行的 `shapes=[]` / `flops=0.0` 表示未采集**：`maybe_apply_stage6_byoc` 返回的 meta 将 `shapes` 设为空列表，且 BYOC 子图不产生 PrimFunc，因此没有 `PrimFuncInfo`。该 FLOPs 值不适合用于性能比较。

---

## Tensorize / WMMA 的触发条件

两条路径均要求：

1. target 满足各自的 Tensor Core 架构门槛；
2. PrimFunc 可识别为 **matmul 语义的 TIR 归约**（`C[S,I,J] += A*B`）；
3. 输入与累加 dtype 存在匹配的 WMMA intrinsic。

Relax 算子名和本仓库的 `category=matmul` 仅用于前置策略分流，不能保证后续 tensorize。

本仓库不直接调用 `sch.tensorize`。`classify_kernel_kind()` 仅检查调度后的 TIR。

| | DLight | MetaSchedule |
| --- | --- | --- |
| 执行组件 | `dlight.gpu.Matmul` | `MultiLevelTilingTensorCore` + postproc `RewriteTensorize` |
| 实现方式 | 形态与约束均满足后调用 `sch.tensorize(WMMA 16×16×16)` | 将 Tensor Core 候选加入搜索空间；最终 trace 由 `RewriteTensorize` 写回 |
| target 条件 | CUDA 且 **`arch >= sm_70`** | `from-target` 在 **`arch >= sm_75`** 时加入 Tensor Core 规则 |
| TIR 条件 | 可规范化为 matmul 归约；GEMV、Reduction 与 Fallback 不执行 tensorize | matmul-like 归约且存在已注册的 WMMA intrinsic |
| 其他条件 | 非 batch 静态维均 **> 64**；未设置 `dlight.do_not_tensorize` | 与 SIMT 候选共同测速，最终结果不保证使用 WMMA |
| 本仓库入口 | `ApplyDefaultSchedule(Matmul, …)` | `MetaScheduleTuneIRMod`（该接口不暴露 `space_generator`） |

若 TIR 形态不匹配（如 add/relu），DLight 尝试后续规则，MetaSchedule 仅生成普通 tiling 候选。若 matmul 形态匹配但 dtype、架构或尺寸约束不满足，DLight 使用 Matmul 规则的 SIMT 分支，而非 Fallback。

当前 CLI 默认 target 为 sm_70，示例 Relax 图均使用 FP32，因此两条路径均不会生成 WMMA，输出为 `kernel matmul: simt`。

验证方式：TIR 中出现 `tvm_mma_sync` 或 `wmma_` 表示 WMMA；仅出现 `thread_binding` 表示 SIMT。CUDA C 可检查 `nvcuda::wmma`，PTX 可检查 `wmma.` 指令。

---

## `relax.build` Codegen 与产物格式

调度只产出 **已调度 TIR**。GPU 上能跑的二进制来自 CLI 里的 `cuda_codegen.build_cuda_executable` → `relax.build`（`--no-codegen` 则整段跳过）。

```text
已调度 TIR
  → SplitHostDevice          host 图 / device 图拆开
  → CodegenCUDA              device 图 → CUDA C 字符串
  → tvm_callback_cuda_compile
        nvrtc → cubin        或
        nvcc  → fatbin       （fatbin 里通常同时带着 PTX + cubin）
  → 链进 VMExecutable        host=llvm，device blob 嵌在 cuda 模块里
  → export_library           写成可加载的 .so
```

`classify_kernel_kind` / DLight / MetaSchedule **都不选** `.cu` 还是 cubin。它们只改 TIR。形态由下面两件事决定。

### 1. 编译器：`TVM_CUDA_COMPILE_MODE`

TVM 回调 `tvm_callback_cuda_compile`（`tvm/support/nvcc.py`）：

| 环境变量 | 调用 | 嵌进 VM 的 device 二进制 | 前置 |
| --- | --- | --- | --- |
| 未设（**TVM 默认**） | NVRTC | **cubin** | `pip install cuda-bindings` |
| `nvcc`（**本仓库 CLI 默认**） | `nvcc` 子进程 | **fatbin** | PATH 里有 `nvcc` |
| `nvrtc` | 同第一行 | cubin | 同上 |

CLI 的 `prepare_cuda_compile_env` 用 `setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")`，因为当前 venv 没有 `cuda-bindings`。已经 export 过该变量则不改。

因此：

- **cubin**：单 arch 的 CUDA 机器码（SASS）。NVRTC 路径的正式产物。
- **fatbin**（有时写作 nvfatbin）：NVIDIA 的 fat binary 容器，nvcc `--fatbin` 生成；里面可以有 cubin，也可以带一份 PTX 供 JIT。
- **PTX**：虚拟 ISA 文本。nvcc 编译过程中会经过它；**正式嵌进 VM 的不是裸 `.ptx` 文件**，除非你去 dump 目录里看中间文件。
- **`.cu`**：CodegenCUDA 生成的 CUDA C 源码，随后交给 NVRTC 或 nvcc；该文件不是运行时二进制。
- **`.so`**：host 侧共享库。`export_library` 把 VMExecutable（llvm host + 已嵌入的 cubin/fatbin）打成 ELF。运行时加载的是这个，不是单独去 `dlopen` cubin。

必须在 `with target:` 上下文中调用 `relax.build(...)`，否则 CUDA 编译回调无法获取 `arch`，并报告 `No CUDA architecture was specified`。若绕过本 CLI 的 target 解析而直接构造裸 `Target("cuda")`，无 GPU 时会回退为 `sm_50`。

### 2. 落盘：哪些是正式产物，哪些是 dump

`--work-dir DIR`（缺省则 TOML 的 `work_dir`）。codegen 相关文件：

```text
DIR/
├── hybrid_cuda_sm70.so      # 正式：export_library，VM 加载这个
├── tvm_kernels.cu           # 副本：本仓库从 ex.mod 抽出的 CUDA C（inspect_source）
└── kernel_dump/             # dump：仅当 TVM_KERNEL_DUMP 有值
    ├── tvm_kernels.cu       # 送给 nvcc 的那份源（和上面内容同类，路径不同）
    ├── tvm_kernels.ptx      # nvcc 中间 PTX（.target sm_70）
    ├── tvm_kernels.cubin    # nvcc 也会写出
    ├── tvm_kernels.fatbin   # nvcc 模式生成并嵌入 .so 的设备代码容器
    └── tvm_kernels.cudafe1.* / .cpp1.ii   # nvcc 前端中间文件
```

`prepare_cuda_compile_env` 通过 `setdefault` 将 `TVM_KERNEL_DUMP` 设为 `DIR/kernel_dump`，因此 CLI 默认输出 dump。若环境变量已指向其他目录，则文件写入该目录。

| 你想要 | 看哪个 | 谁决定有没有 |
| --- | --- | --- |
| 能 `VirtualMachine` 加载的库 | `DIR/hybrid_cuda_sm70.so` | 每次 codegen 都写 |
| 可读的 CUDA C | `DIR/tvm_kernels.cu` | 抽出 cuda 模块成功才写 |
| 核对 PTX arch / kernel 名 | `kernel_dump/tvm_kernels.ptx` | 开了 `TVM_KERNEL_DUMP` |
| 确认 VM 嵌入 cubin 或 fatbin | 检查 `TVM_CUDA_COMPILE_MODE` 及 dump 中的对应文件 | 见上一节 |

内存里模块树（抽出 `.cu` 时 DFS 这一层）：

```text
relax.VMExecutable
└── llvm     host
    └── cuda  device（blob = cubin 或 fatbin）
```

launch：`VirtualMachine(ex, cuda(0))`。没驱动就 skip，`.so` 已经编好。`--no-run` 编完不 launch。

```bash
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind matmul \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/codegen \
  --force-mode dlight --skip-ms-tune
# --no-codegen 停在 TIR；--no-run 只编不跑
```

---

## 三条路径

每个 PrimFunc 仅进入以下路径之一：

| | BYOC（`--kind qdq`） | DLight | MetaSchedule |
| --- | --- | --- | --- |
| 何时 | 模块匹配阶段六 QDQ 模式，且 `enable_byoc_stage6=true` | 决策结果为 `DLIGHT` | 决策结果为 `METASCHEDULE` |
| 实际跑什么 | 阶段六 `apply_pipeline` → `.so` | 规则 tiling，无搜索 | GPU 上 search + JSON 库 |
| 需要 CUDA？ | 匹配本身不需要 | llvm 也能跑（cuda 上才会标 `tirx.is_scheduled`） | **要** |

---

## 正确性（pytest M1–M5）

```bash
pytest 07_tvm_hybrid_schedule/tests -q
```

| 项 | 文件 | 测什么 | 通过标准 | GPU |
| --- | --- | --- | --- | --- |
| **M1** | `tests/test_m1_decision.py` | 判定：FLOPs / force_mode / 白黑名单 | 大 matmul→MS；小 add+relu→DLight；blacklist 优先于 FLOPs | 否 |
| **M2** | `tests/test_m2_dlight.py` | DLight 只调度点名的 PrimFunc | cuda 上至少 1 个 `tirx.is_scheduled`；空名单不改其它 func | 否（cuda 更严） |
| **M3** | `tests/test_m3_metaschedule.py` | 调优及同一 `work_dir` 的缓存命中 | 调优成功；第二次 `ms_cache_hit=true` | **需要**，否则跳过 |
| **M4** | `tests/test_m4_hybrid_e2e.py` | 整条流水线 + 互斥 + 可选 BYOC | 每名字只出现一次策略；写出 `schedule_report.json` | 仅混合 MS 用例要 |
| **M5** | `tests/test_m5_codegen.py` | CLI target=sm_70；nvcc 存在则 `relax.build` 出 `__global__` | arch=sm_70；`.so` + CUDA C | codegen 要 nvcc |

M1 验证决策规则；M2 验证 DLight 子集隔离；M3 验证 MetaSchedule 数据库写入与缓存命中；M4 验证端到端互斥分流；M5 验证 CUDA sm_70 Codegen。

| 条件 | 结果 |
| --- | --- |
| 无 TVM / DLight | CTest 包装脚本 **exit 0（SKIP）** |
| 无 CUDA | M1、M2、M4 的 DLight/BYOC 仍跑；M3 与混合 MS **pytest skip** |
| 有 CUDA + 阶段五 `.so` | 全量运行（包含 BYOC IR 改写） |

---

## CTest 集成

根 `CMakeLists.txt`：`-DCU_EPILOGUE_ENABLE_TVM=ON` 才 `add_subdirectory(07_tvm_hybrid_schedule)`。  
`07_tvm_hybrid_schedule/CMakeLists.txt` 在 **`CU_EPILOGUE_BUILD_TESTS=ON`**（默认开）时注册 **`stage7_tvm_tests`**：

```cmake
add_test(
  NAME stage7_tvm_tests
  COMMAND bash scripts/run_pytest.sh
          ${CMAKE_BINARY_DIR} ${STAGE7_DIR} ${STAGE6_DIR}
)
set_tests_properties(stage7_tvm_tests PROPERTIES
  WORKING_DIRECTORY ${STAGE7_DIR}
  ENVIRONMENT "CU_EPILOGUE_BUILD_DIR=${CMAKE_BINARY_DIR};PYTHONPATH=${STAGE7_DIR}/python:${STAGE6_DIR}/python:..."
)
```

`scripts/run_pytest.sh` 做四件事：

1. 用阶段六 `.venv` 的 python（`CU_EPILOGUE_TVM_VENV` 可覆盖）
2. `import tvm; from tvm import relax; from tvm.s_tir import dlight` → 失败则 **exit 0（SKIP）**
3. 设 `PYTHONPATH`（7 + 6）和 `LD_LIBRARY_PATH`（若 `build/05_qdq_fusion` 存在）
4. `exec python -m pytest tests/ -q`（即 **M1–M5**）

与阶段六不同，阶段七测试不以 `.so` 是否存在作为整体运行条件；缺少 `.so` 时，BYOC 用例仍验证流水线可正常返回。

```bash
ctest --test-dir build -R stage7 --output-on-failure
```

> 测试注册在 build 目录。在仓库根目录直接运行 `ctest` 将提示 `No tests were found!!!`。

CTest **不跑** `bench_hybrid.py` / `build_and_run.py`。

---

## 编译 CLI

```bash
# 混合图：大 matmul + 小 add/relu；force dlight 保证无搜索
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind hybrid \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/demo \
  --force-mode dlight

# auto（默认阈值 50M；256³ 仍可能全是 DLight，见上）
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind hybrid \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/demo \
  --force-mode auto --max-trials 16

# QDQ：可能走 BYOC，该子图不再打 DLight/MS
python 07_tvm_hybrid_schedule/python/build_and_run.py --kind qdq \
  --work-dir 07_tvm_hybrid_schedule/docs/work_dirs/byoc_demo
```

主要参数：

- `--kind matmul|elem|hybrid|qdq`：选择示例模块；
- `--target cuda|sm_70|llvm|<TVM target>`：选择编译 target；`cuda` 与 `sm_70` 均映射到 `nvidia/nvidia-v100`（sm_70）；
- `--skip-ms-tune`：跳过搜索，但仍执行 ApplyDatabase；
- `--no-codegen`：返回已调度 TIR，不执行 `relax.build`；
- `--no-run`：完成 Codegen，但不在 `cuda(0)` 启动 Relax VM。

CLI 固定使用参数指定的 target；相比之下，`bench_hybrid.py` 会检测 `cuda(0)` 的实际架构，无 GPU 时回退到 LLVM。

### 输出文件

`--work-dir` 下包含三类互相独立的输出：

| 文件 | 谁写 | 是什么 |
| --- | --- | --- |
| `schedule_report.json` / `.csv` | 本仓库 | 策略分流（与 kernel 二进制无关） |
| `database_workload.json` / `database_tuning_record.json` | TVM MetaSchedule | 调参库；没 tune 过可能是空文件 |
| `hybrid_cuda_sm70.so`、`tvm_kernels.cu`、`kernel_dump/` | CLI `relax.build` | Codegen 产物；cubin、fatbin 与各文件含义见 [`relax.build` Codegen 与产物格式](#relaxbuild-codegen-与产物格式) |

终端类似：

```text
target=cuda -keys=cuda,gpu -arch=sm_70 ...
arch=sm_70
compile_seconds=0.023
  fused_add_relu: DLIGHT reason=category=elementwise flops=1024 cache=None
  matmul: DLIGHT reason=default (flops=33554432<=50000000) flops=33554432 cache=None
report → .../schedule_report.json
primfuncs after: ['fused_add_relu', 'matmul', ...]
  kernel fused_add_relu: simt
  kernel matmul: simt
codegen so → .../hybrid_cuda_sm70.so
codegen cu → .../tvm_kernels.cu
```

同一 `work_dir` 再编一次 MS 算子：应出现 `cache=True`。

---

## Bench（编译耗时，不是 kernel ms）

```bash
python 07_tvm_hybrid_schedule/scripts/bench_hybrid.py --max-trials 16 \
  --work-root 07_tvm_hybrid_schedule/docs/work_dirs/bench
```

图固定为 `build_hybrid_native_module(128, 128, 128, 32)`，`enable_byoc_stage6=False`。

| 侧 | 实际跑什么 |
| --- | --- |
| **dlight_only** | `force_mode=dlight`，`skip_ms=True` |
| **metaschedule_only** | `force_mode=metaschedule`，执行调优（无 GPU 时跳过） |
| **hybrid_auto** | `force_mode=auto`，`flops_threshold=100_000`（128³ matmul 选择 MetaSchedule） |

无 CUDA 时仅输出 `dlight_only` 与 `hybrid_auto_dlight_fallback` 两行，并打印
`NOTE: no CUDA GPU — skipped MetaSchedule-only / full hybrid tune`。

终端输出：

```text
label,compile_seconds,strategies,n_ops
dlight_only,0.xxx,"DLIGHT",2
metaschedule_only,x.xxx,"METASCHEDULE",2
hybrid_auto,x.xxx,"DLIGHT|METASCHEDULE",2
```

加速比含义是「混合编译 vs 全量 MS 搜索」，**不是**阶段六那种 GPU fused vs NumPy。

本阶段 **没有** nsys / ncu。融合 INT8 延迟 / DRAM 继续看 [阶段六 README](../06_tvm_cutlass_byoc/README.md)。

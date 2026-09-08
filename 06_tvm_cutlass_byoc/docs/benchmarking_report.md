# 阶段六验收与基准报告 — TVM Relax + CUTLASS QGEMM BYOC

环境准备、pytest / ctest 命令见 [README](../README.md)。本文记 **验收口径、如何确认跑在阶段五 `.so`、以及已测数字**。

## 范围

- **Kernel**：复用阶段五 `libcutlass_qgemm_bias_gelu_runtime.so`（SIMT INT8 + SFU FastGELU，affine 零点）。TVM **不**重写 CUTLASS，也 **不**走 TVM 自带 CUTLASS codegen / nvcc。
- **编译路径**：手写 Relax QDQ 图 → pattern 匹配 → `R.call_dps_packed("cu_epilogue.qgemm_bias_gelu")` → packed func 调阶段五 C ABI。每个复合算子对应 **一次 `.so` API 调用**。
- **TVM 后端**：默认 `Target("cuda", host="llvm")` + `tvm.cuda(0)`（V100 / sm_70）。CUDA Tensor 用 `data_ptr()` 直调 C ABI；无 GPU 时回退 llvm/cpu + host H2D。
- **硬件**：Tesla V100（sm_70）。无 INT8 Tensor Core，验收路径为 SIMT。
- **TVM**：apache-tvm 0.26.0（`06_tvm_cutlass_byoc/.venv`）。

> 一次 `.so` 调用 **不等于** 一次 CUDA kernel。C ABI 内部为 affine 零点固定启动 **RowSum + Gemm + ColSum** 三个 kernel。

## 怎么确认跑在 CUTLASS `.so` 里

```text
Relax main
  → R.call_dps_packed("cu_epilogue.qgemm_bias_gelu")
  → Python packed func `_qgemm_packed`
  → launch_device（CUDA Tensor）或 launch（host numpy）
  → libcutlass_qgemm_bias_gelu_runtime.so::cutlass_qgemm_bias_gelu
  → GPU: RowSumKernel + CUTLASS Gemm + ColSumKernel
```

| 层 | 看什么 | 期望 |
| --- | --- | --- |
| IR | `apply_pipeline` 之后的 `main` | `call_dps_packed("cu_epilogue.qgemm_bias_gelu")`，`Codegen=cutlass`；不再是 DQ/matmul/GELU/Q 链 |
| 计数 | `so_wrapper.launch_count()` | M3 / `build_and_run.py` 为 **1**（C 入口次数，不是 kernel 个数） |
| nsys | `cuda_gpu_kern_sum` | 只有 RowSum / Gemm / ColSum；**没有**独立 DQ、Bias、GELU、Q compute kernel |
| 加载 | `find_runtime_so()` / `LD_LIBRARY_PATH` | `build/05_qdq_fusion/libcutlass_qgemm_bias_gelu_runtime.so` |

两条执行路径不要混：

| 路径 | 谁用 | 张量 | 额外拷贝 |
| --- | --- | --- | --- |
| **CUDA VM** | `build_and_run.py` / M3 | `tvm.runtime.tensor(..., cuda)` | packed 内 `launch_device`，无 H2D |
| **Host `launch()`** | `bench_e2e.py`、`profile_nsys.sh` | numpy | 每次 `cudaMalloc` + H2D/D2H |

`build_and_run.py` 若打印 `VM path failed ... using direct packed launch`，说明 Relax VM 失败、退到 host `launch()`——仍是同一 `.so`，但不是 CUDA VM 快路径。成功时不应出现这行。

## 正确性

三项 pytest 均已通过。需 GPU + 阶段五 `.so`；M2 仅依赖 TVM。

| 项 | 文件 | 验证 |
| --- | --- | --- |
| **M1** | `tests/test_m1_so_correctness.py` | 直接调 `.so` vs NumPy affine 参考，INT8 绝对差 ≤ 1 |
| **M2** | `tests/test_m2_pattern.py` | 完整 QDQ+bias+FastGELU 打上 `cutlass.qgemm_bias_gelu`；缺 bias 不匹配 |
| **M3** | `tests/test_m3_e2e.py` | Relax → packed → `.so`；差 ≤ 1；`launch_count() == 1` |

```bash
# 仓库根；venv 与环境变量见 README
pytest 06_tvm_cutlass_byoc/tests -q
python 06_tvm_cutlass_byoc/python/build_and_run.py --m 64 --n 64 --k 64
# 期望：max_abs=0（或 ≤1）  launches=1  OK
```

## Kernel 数量（nsys，已测）

```bash
./06_tvm_cutlass_byoc/scripts/profile_nsys.sh
```

负载：形状 **256³**，**3** 次 `launch()` → **9** 个 GPU 计算 kernel（每次调用 3 个）。脚本走 host `launch()`，时间线上会有 memcpy；那是胶水，不代表融合失败。

| 时间占比 | 实例数 | Kernel |
| --- | --- | --- |
| ~50% | 3 | `RowSumKernel`（affine 零点准备） |
| ~42% | 3 | CUTLASS `Gemm<...>`（INT8 MMA + Bias + FastGELU + Q epilogue） |
| ~8% | 3 | `ColSumKernel`（affine 零点准备） |

与阶段五相同的 **三 kernel** 结构。中间 f32 不落 DRAM。产物：`06_tvm_cutlass_byoc/docs/profile_artifacts/stage6_nsys.nsys-rep`。

## DRAM / 带宽（ncu，已测）

```bash
# 采集 GPU 计数器通常需要 sudo（否则可能 ERR_NVGPUCTRPERM）；ncu 用 2025.2.1
./06_tvm_cutlass_byoc/scripts/profile_ncu.sh
```

负载：**1** 次融合 `.so` 调用，形状 **256³**。产物：

- `06_tvm_cutlass_byoc/docs/profile_artifacts/stage6_ncu.ncu-rep`
- `06_tvm_cutlass_byoc/docs/profile_artifacts/stage6_ncu_summary.csv`

| Kernel | DRAM 读 | DRAM 写 | Sectors 读 | Sectors 写 | DRAM 峰值占比 | SM 峰值占比 | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RowSumKernel` | 69.0 KB | 0 | 2157 | 0 | 0.15% | 0.04% | 53.0 µs |
| `ColSumKernel` | 70.0 KB | 0 | 2189 | 0 | 0.53% | 0.18% | 14.6 µs |
| CUTLASS `Gemm`（INT8 + Bias + FastGELU + Q） | 169.2 KB | 0 | 5289 | 0 | 0.33% | 1.45% | 56.4 µs |
| **合计（本调用）** | **约 308 KB** | **0** | **9635** | **0** | — | — | **约 124 µs** |

如何读：

- 仍 **没有** 独立 DQ / GELU / Q kernel；量化在 Gemm 寄存器路径内完成。
- 256³ DRAM **读**约 308 KB。未融合 GPU 路径若把 DQ(A)/DQ(B)/matmul/gelu 的 f32 中间结果落显存，量级往往是数 MB（单块 256²×4 ≈ 256 KB，再乘多块）。
- 本规模 DRAM **写**为 0：INT8 输出（约 64 KB）和很小的 sum 缓冲可落在 V100 L2，**不是**「kernel 没写」。要看写流量可改脚本形状（例如 `1×4096×4096`）后重跑。

上表只描述 **融合调用本身** 的 GPU DRAM；端到端加速比用的是下一节的 CPU NumPy 基线。

## 端到端延迟（已测）

主机：Tesla V100；阶段五 SIMT `.so`。下表为已记录一次运行；当前 `bench_e2e.py` 默认 warmup **5** / 重复 **20**（表中数字来自更早的 warmup 3 / 重复 10，数量级应接近）。

```bash
python 06_tvm_cutlass_byoc/scripts/bench_e2e.py --m 256 --n 256 --k 256
python 06_tvm_cutlass_byoc/scripts/bench_e2e.py --m 1 --n 4096 --k 4096
```

| 形状 | 融合路径（ms） | 未融合 NumPy CPU（ms） | 加速比 |
| --- | ---: | ---: | ---: |
| 256³ | 0.528 | 2.413 | **4.57×** |
| 1×4096×4096 | 7.079 | 79.362 | **11.21×** |

相对文档约定的未融合基线，目标 **≥ 1.5×**：**已满足**。

读数时注意：

- 融合侧是 GPU `.so`（含 host `launch()` 的 H2D/D2H）；未融合侧是 **CPU NumPy**，跨设备，说明的是数量级，不是「同 GPU 未融合 vs 融合」。
- 脚本 **没有**走 Relax CUDA VM；M3 才覆盖那条路径。
- 与阶段四/五一样，同 GPU 的未融合 AOT 基线仍缺（需 NVPTX），见 [阶段五 aot_gpu_baseline.md](../../05_qdq_fusion/docs/aot_gpu_baseline.md)。

## Tile / 对齐

阶段六 **不改** 阶段五的 `ThreadblockShape` / `WarpShape`。256³ 上 Gemm SM 利用率约峰值 **1.5%**；occupancy / tile 实验请换更大形状并重新 profile。

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `python/relax_patterns.py` | 手写 QDQ Relax 图 + matcher |
| `python/relax_pipeline.py` | composite → packed；CUDA `build_vm` |
| `python/so_wrapper.py` | ctypes 加载 `.so`；`launch` / `launch_device` |
| `python/build_and_run.py` | CUDA VM e2e |
| `scripts/bench_e2e.py` | 融合 `.so` vs NumPy CPU |
| `scripts/profile_nsys.sh` / `profile_ncu.sh` | kernel 数量与 DRAM |
| [阶段五 runtime ABI](../../05_qdq_fusion/runtime/cutlass_qgemm_bias_gelu_c_api.cu) | `cutlass_qgemm_bias_gelu` |

# 阶段五验收与基准报告 — QDQ 整图融合 → INT8 CUTLASS 调用

## 范围

- **编译路径**（Linalg-on-Tensor，在 One-Shot Bufferize **之前**）：
  1. `--fuse-qdq-qgemm-bias-gelu` → 高阶 Op `cutlass.qgemm_bias_gelu`
  2. `--one-shot-bufferize` → 同一 Op 的 memref 形态
  3. `--lower-cutlass-qgemm-to-call` → `func.call @cutlass_qgemm_bias_gelu`
- **运行时 kernel**：SIMT INT8 GEMM，在 epilogue 内完成 bias、Fast-GELU、再量化；affine 零点修正在 C ABI 内部固定为 **RowSum + Gemm + ColSum**（与阶段六 nsys/ncu 看到的结构一致）。
- **硬件**：默认 SIMT 路径面向 Tesla V100（sm_70）；Sm90 visitor 为旁路构建（`sm_90a`），**不替换**默认 `.so`。
- **与阶段四的关系**：阶段四在 bufferize **之后**融合 FP32 GEMM+GELU；阶段五在 **tensor IR 上**收齐整条 QDQ 链，避免 OSB 为 DQ/GELU/Q 中间结果分配 f32 缓冲区。

## CMake：配置 vs 编译

| 命令 | 作用 |
| --- | --- |
| `cmake -B build -DCU_EPILOGUE_ENABLE_CUDA=ON` | **配置**构建树，开启 CUDA，以便生成 INT8 kernel / `.so` / 正确性可执行文件。 |
| `cmake --build build -j$(nproc) --target qdq-opt cutlass_qgemm_bias_gelu_runtime stage5_qgemm_correctness_test` | **编译** MLIR 工具、共享库与 GPU 正确性二进制。 |

```bash
cmake -B build -DCU_EPILOGUE_ENABLE_CUDA=ON
cmake --build build -j$(nproc) --target qdq-opt \
  cutlass_qgemm_bias_gelu_runtime stage5_qgemm_correctness_test
```

可选 Sm90 visitor（Hopper 上跑正确性；V100 上设备 correctness 预期 SKIP）：

```bash
cmake -B build -DCU_EPILOGUE_ENABLE_SM90_VISITOR=ON
cmake --build build -j$(nproc) --target stage5_visitor_correctness_test
```

## 正确性

### A. IR 流水线（不需要 GPU）

对 `test/qdq_pattern.mlir` 做三段检查：

```bash
cd build && ctest -R stage5_qdq_fusion_test --output-on-failure
# 或：
./05_qdq_fusion/test/run_tests.sh ./build/05_qdq_fusion/qdq-opt
```

| 步骤 | Pass | 期望 |
| --- | --- | --- |
| 1 | `--fuse-qdq-qgemm-bias-gelu` | 出现 `cutlass.qgemm_bias_gelu`；无 `linalg.matmul` |
| 2 | + `--one-shot-bufferize=...` | Op 落在 `memref` 上；融合路径上无多余的 `tensor.empty` |
| 3 | + `--lower-cutlass-qgemm-to-call` | 出现 `call @cutlass_qgemm_bias_gelu` |

第 3 步手工等价命令：

```bash
BUFFERIZE_OPTS='bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map'
./build/05_qdq_fusion/qdq-opt 05_qdq_fusion/test/qdq_pattern.mlir \
  --fuse-qdq-qgemm-bias-gelu \
  --one-shot-bufferize="$BUFFERIZE_OPTS" \
  --lower-cutlass-qgemm-to-call --canonicalize
```

### B. GPU 数值 vs CPU affine 参考

默认 SIMT kernel，与 CPU 参考（affine DQ → matmul → bias → Fast-GELU → Q）比对：

```bash
cd build && ctest -R stage5_qgemm_correctness_test --output-on-failure
# 或：
./build/05_qdq_fusion/stage5_qgemm_correctness_test
```

输入：host 侧由 f32 模式量化得到的 INT8 A/B、f32 bias、以及每张量的 scale / zero-point——均为 **运行时内存数据**，不是 StableHLO，也不经 `mlir-runner`。

### C. Sm90 visitor（可选）

```bash
./05_qdq_fusion/test/verify_sfu_visitor_sm90.sh
cd build && ctest -R stage5_visitor --output-on-failure
```

在 V100 上，visitor 的设备正确性预期为 **SKIP**；若已编出 `sm_90a` 目标，仍可做 PTX/SASS 的 SFU 静态检查。

## 延迟与 Profiling 说明

阶段五 **没有** 单独的 `benchmark_e2e.sh`。与阶段四对称的 **未融合 GPU AOT** 基线（多 kernel INT8 链、内嵌 PTX ELF）尚在规划中，见 [aot_gpu_baseline.md](aot_gpu_baseline.md)（依赖阶段四 AOT 工具链与 NVPTX MLIR）。

同一份 INT8 `.so` 的端到端融合 vs 未融合 NumPy 延迟，以及 nsys/ncu 分解，见阶段六：

- [06_tvm_cutlass_byoc/docs/benchmarking_report.md](../../06_tvm_cutlass_byoc/docs/benchmarking_report.md)
- 相关命令：`profile_nsys.sh`、`profile_ncu.sh`、`bench_e2e.py`

本阶段验收重点是：**IR 改写正确** + **GPU 结果与 CPU affine 参考一致**。

## 为何在 tensor 上融合（而不是 memref 上）

若 DQ/Q 一直保持为独立 linalg，直到 bufferize 之后再处理，One-Shot Bufferize 会为中间 f32 结果分配 memref。先收成 `cutlass.qgemm_bias_gelu` 后，OSB 只处理 Op 已有的 INT8/f32 操作数；零点相关计算留在 C ABI 内部（RowSum / ColSum），不暴露为 MLIR Op。

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `lib/FuseQdqPattern.cpp` | Tensor 级 QDQ 融合 |
| `lib/LowerToCall.cpp` | `cutlass.qgemm_bias_gelu` → `func.call` |
| `test/qdq_pattern.mlir` | IR 测试输入 |
| `kernels/qgemm_bias_gelu.cu` | 默认 SIMT INT8 融合 kernel |
| `runtime/cutlass_qgemm_bias_gelu_c_api.cu` | 共享库 ABI |
| `kernels/correctness_test.cu` | GPU vs CPU affine 参考 |

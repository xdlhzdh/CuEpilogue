# 阶段一验收与基准报告 — 手写 CUDA SGEMM 基线

## 范围

- **实现**：两套 FP32 行主序 SGEMM（`C = α·A·B + β·C`）
  - **Naive**：`sgemm_naive.cu` — 一线程一输出元素，K 维每步直读 global memory，无 tiling
  - **Shared-Memory Tiling**：`sgemm_tiled_smem.cu` — `32×32` tile，块内协作把 A/B 搬进 `__shared__` 再累加
- **目的**：为后续 CUTLASS / 融合路径建立 **正确性 + profile** 基线（非 Tensor Core）。
- **硬件（下列实测）**：Tesla V100（sm_70）。

## CMake：配置 vs 编译

| 命令 | 作用 |
| --- | --- |
| `cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70` | **配置**构建树（默认开启 CUDA）。 |
| `cmake --build build -j$(nproc) --target stage1_correctness_test` | **编译**阶段一正确性可执行文件。 |

```bash
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70
cmake --build build -j$(nproc) --target stage1_correctness_test
```

## 正确性

相对 `common/matrix_ref.hpp` 中的 CPU 参考（double 累加）：

```bash
cd build && ctest -R stage1_correctness_test --output-on-failure
# 或：
./build/01_baseline_cuda/stage1_correctness_test
```

| 用例 | 规模 | 说明 |
| --- | --- | --- |
| `sgemm_naive` | 256³ | Naive kernel |
| `sgemm_tiled_smem` | 256³ | Tiled kernel |
| `sgemm_tiled_smem_nonmultiple` | 200×130×77 | 非 32 对齐边界 |

输入：host 侧 `FillRandom` 生成的 f32 矩阵，拷到 GPU 后运行；**不是** MLIR / StableHLO。

## Profiling（已测）

```bash
./01_baseline_cuda/profile.sh build/01_baseline_cuda/stage1_correctness_test
```

脚本会尝试 `nsys` / `ncu`（`ncu` 常需 sudo，并自动挑选与驱动兼容的版本）。

256³ 下 **`SgemmTiledSmemKernel`** 代表性结果：

| 指标 | 约值 |
| --- | ---: |
| SM 吞吐（相对峰值） | **51.8%** |
| 显存吞吐（相对峰值） | **50.5%** |
| Achieved Occupancy | **49.8%** |

说明：同一规模下 naive 访存更重；大矩阵吞吐对比请看阶段二 `bench_throughput_large`（手写 tiled 约 3.05 TFLOPS @ 4096³）。

## 实现要点

| 版本 | 核心思路 |
| --- | --- |
| Naive | `for k: acc += A[row,k]*B[k,col]`，全程 global |
| Tiled | 沿 K 分 tile；`As`/`Bs` 进 shared；`__syncthreads` 后块内复用 |

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `sgemm_naive.cu` / `sgemm_tiled_smem.cu` | 两套 kernel |
| `sgemm_baseline.cuh` | host 启动声明 |
| `correctness_test.cu` | GPU vs CPU 参考 |
| `profile.sh` | nsys / ncu 采集 |

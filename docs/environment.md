# 环境说明

记录 CuEpilogue 当前宿主机的软硬件环境、关键版本选择的原因，以及历史沙盒背景。

各阶段测试的**目的 / 运行方法 / 环境依赖 / 结果**见 [README「验收结果总览」](../README.md#验收结果总览)，本文档只覆盖机器本身的环境状态。

## 1. 硬件与系统

| 项目 | 值 |
| --- | --- |
| OS | Linux Mint 22.3（zena），内核 6.17，**原生宿主机**（非 WSL、非沙盒） |
| GPU | 2× Tesla V100-SXM2（sm_70 / Volta）：16GB + 32GB |
| GPU 使用范围 | 项目**默认仅用 GPU 0**；代码中无多卡并行逻辑，第二块卡跑测试时空闲 |
| NVIDIA Driver | 535.309.01 |

## 2. 工具链版本

| 组件 | 版本 / 路径 | 备注 |
| --- | --- | --- |
| CUDA Toolkit | **12.9.86**，`/usr/local/cuda` 已通过 `update-alternatives` 指向此版本 | 必须是 12.x，见下方说明 |
| CMake | 4.3.4（`pipx install cmake`） | |
| Nsight Systems / Compute | `nsys` 2025.1.3；`ncu` 装了两个版本：`2025.2.1`（能用）和 `2026.2.1`（与 535 驱动不兼容，见下） | `ncu` 还需要 `sudo`，见下 |
| LLVM / MLIR | 23.0.0git，`mlir-opt` / `llvm-config` / `MLIRConfig.cmake` 均在 `/usr/local` | 阶段四、五依赖 |
| CUTLASS | v3.5.1，CMake `FetchContent` 自动拉取 | |

### 为什么编译必须用 CUDA 12.x

CUDA **13.0** 起移除了 Volta（sm_70）的离线编译支持。本机同时装有 CUDA 13.3，如果 `update-alternatives` 被切到 13.3，编译会报：

```
nvcc fatal : Unsupported gpu architecture 'sm_70'
```

排查/修复：

```bash
sudo update-alternatives --config cuda   # 选择 /usr/local/cuda-12.9
```

### `ncu` 需要兼容版本 + `sudo`

本机跑 `ncu` 需要同时处理两个问题：

1. **版本兼容**：默认 `ncu` 会选较新的 `2026.2.1`，但它与驱动 535.309.01 不兼容，会误报 `failed to connect to the CUDA driver (stub libcuda.so[.1] on path?)`。实际并不是 stub `libcuda.so` 路径问题，而是工具/驱动版本不匹配；需改用 `/opt/nvidia/nsight-compute/2025.2.1/ncu`。
2. **性能计数器权限**：驱动默认 `RmProfilingAdminOnly=1`，普通用户不能读 GPU 性能计数器，必须对 `ncu` 本身使用 `sudo`，否则报 `ERR_NVGPUCTRPERM`。

`01_baseline_cuda/profile.sh` 已经把“探测兼容版本 + sudo”自动化，直接跑脚本即可；细节见 README「`ncu` 排障记录」。

### 架构相关配置（Volta / V100）

spec 建议 Ampere 及以上架构，但本项目实际目标硬件是 Volta，因此：

- `CU_EPILOGUE_CUDA_ARCH` 默认 `70`（其他 GPU 可覆盖为 `80`/`86`/`89`/`90`）
- CUTLASS Tensor Core 形状使用 `InstructionShape<8,8,4>`，对应 PTX `mma.sync.aligned.m8n8k4`（而非 Ampere 的 `m16n8k8/k16`）
- 流水线深度 `kStages=2`（Volta 无 `cp.async`，不支持更深的多级软件流水线）

### Sm90 visitor 旁路（Hopper）

阶段三、五另有 CUTLASS 3.x visitor / EVT 路径（`CU_EPILOGUE_ENABLE_SM90_VISITOR`，默认 ON）：

- 这些目标单独以 **`CUDA_ARCHITECTURES=90a`** 编译（WGMMA 需要 `__CUDA_ARCH_FEAT_SM90_ALL`；plain `sm_90` 会编出仅 `printf` 的 stub）
- 不改变默认 `CU_EPILOGUE_CUDA_ARCH=70`；阶段四 runtime 仍走 Sm70 functor，阶段五默认 C ABI 仍走 SIMT INT8 kernel
- 本机 V100 上：可做编译 + `./03_fastgelu_epilogue/verify_sfu_visitor_sm90.sh` / `./05_qdq_fusion/test/verify_sfu_visitor_sm90.sh` 静态验收；`stage3_visitor_correctness_test` / `stage5_visitor_correctness_test` 在 cc &lt; 90 时 SKIP
- 关闭：`cmake -B build -DCU_EPILOGUE_ENABLE_SM90_VISITOR=OFF`

## 3. 历史：项目最初的 WSL2 沙盒环境

项目最初在**没有 GPU 访问权限**的 WSL2 沙盒中搭建（Ubuntu 26.04, CUDA 12.4.131, LLVM/MLIR 23.0.0git）。该环境下：

- 阶段一～三：仅完成编译 + 静态 PTX/SASS 指令验证，正确性测试和 Profile 因无 GPU 无法运行。
- 阶段四、五（MLIR，不需要 GPU 即可跑 IR 测试）：曾在沙盒内完整验证阶段四；阶段五同样只依赖 `qdq-opt`。

当前 Linux Mint 宿主机已具备完整的 GPU + MLIR 环境。阶段一～四的构建、正确性测试与静态验证均已在本机跑通；阶段五增加 QDQ tensor 融合与 INT8 kernel。详见 README「验收结果总览」。

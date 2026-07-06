# 环境说明与验证范围 (Environment & Verification Scope)

本项目最初是在一个 **没有 NVIDIA GPU 设备访问权限的 WSL2 沙盒**里搭建、编译和（部分）测试的。这份文档记录了沙盒的真实能力边界、已经验证过什么、以及在真机（2x Tesla V100）上还需要跑哪些步骤，避免后续误以为"编译通过 = 验收标准全部达成"。

## 1. 沙盒环境（编写本项目时的开发环境）

| 项目 | 值 |
| --- | --- |
| OS | Ubuntu 26.04 (WSL2, kernel 6.6.114.1-microsoft-standard-WSL2) |
| GPU 设备访问 | **不可用** - `nvidia-smi` 报 `GPU access blocked by the operating system`，即使用最高权限重试依然失败 |
| CUDA Toolkit | 12.4.131（通过 `apt install nvidia-cuda-toolkit` 安装，见 [scripts/setup_env.sh](../scripts/setup_env.sh)） |
| Host 编译器 | nvcc 自动选用 gcc-13.4（CUDA 12.4 官方支持的编译器版本，随 apt 包一起安装） |
| Nsight 工具 | `nsys` (2023.4.4)、`ncu` (2024.1.1) 均已安装，但**无法在沙盒内实际采集 Profile**（需要设备） |
| CMake | 4.2.3 |
| LLVM/MLIR | 23.0.0git（trunk），预先构建安装在 `/usr/local`，`mlir-opt`/`llvm-config`/`MLIRConfig.cmake` 等均可用 |
| CUTLASS | v3.5.1，通过 CMake `FetchContent` 从 GitHub 拉取（`cmake/FindOrFetchCutlass.cmake`） |

## 2. 真实目标硬件

项目的实际运行/验收硬件是：

```
2x Tesla V100-SXM2 (Volta, sm_70), Driver 582.53, CUDA Version 13.0 (TCC 模式)
```

这与 spec 文档"Ampere 或更高架构优先"的建议不同（V100 属于 **Volta** 架构），因此：

- 顶层 `CMakeLists.txt` 的 `CU_EPILOGUE_CUDA_ARCH` 默认值设为 **`70`**（可通过 `-DCU_EPILOGUE_CUDA_ARCH=80/86/89/90` 切换到 Ampere/Ada/Hopper）。
- 阶段二/三的 CUTLASS 模板使用 Volta 的 Tensor Core MMA 形状 `InstructionShape<8,8,4>`（对应 PTX `mma.sync.aligned.m8n8k4`），而非 Ampere 的 `m16n8k8/k16`。
- Volta 没有 `cp.async`，多级软件流水线（`NumStages > 2`）是 Ampere+ 特性，因此 GEMM 模板的 `kStages` 固定为 `2`。

## 3. 每个阶段：沙盒内验证了什么 / 真机上还需要验证什么

| 阶段 | 沙盒内已验证 | 需要真机（V100）验证 |
| --- | --- | --- |
| 阶段一 Native CUDA SGEMM | `nvcc` 编译通过（`stage1_correctness_test` 可执行文件成功链接） | 实际执行 `stage1_correctness_test` 做数值正确性比对；用 `01_baseline_cuda/profile.sh` 跑 `nsys`/`ncu`，记录 Occupancy、显存带宽 vs. 理论峰值 |
| 阶段二 CUTLASS 融合 GEMM | `nvcc` 编译通过；`02_cutlass_gemm/verify_mma_ptx.sh` 静态确认生成的 PTX 中出现 `mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32`，且未检测到 `ld.local`/`st.local`（无寄存器溢出迹象） | 实际执行 `stage2_correctness_test`；用 `ncu` 确认吞吐量显著优于阶段一 |
| 阶段三 Inline PTX Fast-GELU | `nvcc` 编译通过；`03_fastgelu_epilogue/verify_sfu_sass.sh` 静态确认 PTX 含 `ex2.approx.f32`/`rcp.approx.f32`，编译出的 cubin SASS 含 `MUFU.EX2`/`MUFU.RCP`，且无寄存器溢出 | 实际执行 `stage3_correctness_test`，检查数值误差是否在脚本设定的 epsilon 范围内（`max_abs_exact < 0.05`，`max_abs_sfu < 0.01`） |
| 阶段四 MLIR 编译器集成 | **完整验证**：`fused-opt` 工具构建成功；`04_compiler_integration/test/run_tests.sh` 端到端跑通 `--one-shot-bufferize` + `--fuse-gemm-gelu`，确认 `linalg.matmul`/`linalg.generic` 被正确替换为对 `@cutlass_fused_gemm_gelu` 的 `call`；`cutlass_fused_gemm_gelu_runtime.so` 编译成功 | 把 IR 进一步 Lower 到 LLVM（`use-bare-ptr-memref-call-conv=1`），链接 `cutlass_fused_gemm_gelu_runtime.so`，用 `mlir-cpu-runner`/自定义 runner 实际执行，对比"完全展开为标量指令"的端到端延迟，验证 spec 要求的 30% 以上下降 |

## 4. 在真机上复现的步骤

```bash
# 1. 安装 CUDA Toolkit（如果目标机器还没有 nvcc/nsys/ncu）
./scripts/setup_env.sh

# 2. 配置 + 编译（sm_70 对应 V100；换其他卡改 -DCU_EPILOGUE_CUDA_ARCH=）
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70
cmake --build build -j$(nproc)

# 3. 跑正确性测试（这一步在真机上才会真正执行 kernel）
cd build && ctest --output-on-failure

# 4. 跑 Profile（阶段一验收标准：nsys/ncu 报告）
cd .. && ./01_baseline_cuda/profile.sh build/01_baseline_cuda/stage1_correctness_test

# 5. 静态指令检查（真机/沙盒均可跑，只是在真机上"眼见为实"更有说服力）
./02_cutlass_gemm/verify_mma_ptx.sh
./03_fastgelu_epilogue/verify_sfu_sass.sh

# 6. 阶段四端到端测试（不需要 GPU，任何机器都能跑）
./04_compiler_integration/test/run_tests.sh build/04_compiler_integration/fused-opt
```

也可以直接跑 [scripts/build_all.sh](../scripts/build_all.sh) 一次性完成第 2-6 步。

## 5. 关于"编译通过"和"验收标准"的区别

沙盒内所有 4 个阶段都能**编译成功**，阶段二/三额外做了**静态 PTX/SASS 指令级验证**（不需要 GPU 就能确认关键指令确实生成了），阶段四做到了**完整的端到端功能验证**。但阶段一/二/三 spec 里要求的"输出 nsys/ncu Profile 报告"、"计算吞吐量显著优于阶段一"、"数值误差在 epsilon 范围内"这几条验收标准，**必须在真机上跑 `ctest` 和 `profile.sh` 才能拿到实际数据**，沙盒本身无法完成。

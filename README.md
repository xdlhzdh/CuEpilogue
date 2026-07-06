# CuEpilogue

Fused GEMM + Epilogue (Fast-GELU) 自定义算子开发与编译器集成 —— 完整需求见 [docs/spec.md](docs/spec.md)。

四个阶段依次递进：

1. **Native CUDA SGEMM baseline**（`01_baseline_cuda/`）：无 tiling 版本 + Shared Memory tiling 版本，作为性能/精度基准。
2. **CUTLASS 融合 GEMM**（`02_cutlass_gemm/`）：FP16 输入 / FP32 累加，针对目标 GPU 架构配置 Threadblock/Warp/Instruction Shape，触发 Tensor Core `mma.sync.aligned` 指令。
3. **Inline PTX Fast-GELU Epilogue**（`03_fastgelu_epilogue/`）：把 spec 给出的 `FastGeluPTX`（`ex2.approx` + `rcp.approx` 硬件近似指令）接入 CUTLASS Epilogue，GEMM 输出直接在寄存器里做激活，不再落一次中间显存。
4. **MLIR 编译器后端集成**（`04_compiler_integration/`）：`fused-opt` 工具在 Bufferization 之后识别 `linalg.matmul` + 逐元素激活 `linalg.generic` 的组合，替换成对外部 CUTLASS 算子 `.so` 的 `call`，而不是继续下降到标量 LLVM IR。

> **重要**：本项目最初是在一个没有 NVIDIA GPU 设备访问权限的沙盒里搭建和编译验证的。阶段一~三已完成编译 + 静态 PTX/SASS 指令验证，但需要在真机（本项目的目标硬件是 2x Tesla V100, sm_70）上跑 `ctest`/`profile.sh` 才能拿到 spec 要求的实际性能数据和数值误差报告。阶段四（MLIR）不需要 GPU，已在沙盒内完整验证。详见 [docs/environment.md](docs/environment.md)。

## 目录结构

```
CuEpilogue/
├── CMakeLists.txt                  # 顶层构建脚本
├── cmake/FindOrFetchCutlass.cmake  # CUTLASS FetchContent
├── common/                         # 共享 CUDA 工具 / CPU 参考实现
├── 01_baseline_cuda/                # 阶段一：Native CUDA SGEMM
├── 02_cutlass_gemm/                  # 阶段二：CUTLASS 融合 GEMM
├── 03_fastgelu_epilogue/              # 阶段三：Inline PTX Fast-GELU
├── 04_compiler_integration/            # 阶段四：MLIR 编译器集成
├── scripts/setup_env.sh              # 安装 CUDA Toolkit
├── scripts/build_all.sh              # 一键配置+编译+验证+测试
└── docs/
    ├── spec.md                      # 需求规格说明书
    └── environment.md               # 沙盒验证范围 / 真机复现步骤
```

## 快速开始

### 1. 安装依赖

```bash
./scripts/setup_env.sh   # apt 安装 CUDA Toolkit（nvcc, nsys, ncu）
```

阶段四（MLIR）需要预先装好 LLVM/MLIR（含 CMake 配置文件 `MLIRConfig.cmake`），本项目开发环境里已经装在 `/usr/local`。如果你的机器还没有，需要自行构建/安装一份支持 `find_package(MLIR CONFIG)` 的 LLVM/MLIR（推荐与本项目同源的 LLVM trunk 或最近的 release 分支）。

### 2. 配置与编译

```bash
# CU_EPILOGUE_CUDA_ARCH 默认 70 (Volta/V100)；换其他 GPU 请覆盖，例如 Ampere: 80
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70
cmake --build build -j$(nproc)
```

常用 CMake 选项：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `CU_EPILOGUE_CUDA_ARCH` | `70` | 目标 SM 架构（70/80/86/89/90...） |
| `CU_EPILOGUE_ENABLE_CUDA` | `ON` | 是否构建阶段一~三（需要 `nvcc`，找不到会自动关闭并给出警告） |
| `CU_EPILOGUE_ENABLE_MLIR` | `ON` | 是否构建阶段四（需要 `find_package(MLIR)` 成功） |
| `CU_EPILOGUE_BUILD_TESTS` | `ON` | 是否注册 `ctest` 测试 |

### 3. 运行测试

```bash
cd build && ctest --output-on-failure
```

- `stage1_correctness_test` / `stage2_correctness_test` / `stage3_correctness_test`：需要真实 GPU 才能通过（在无 GPU 环境下会报 `CUDA driver version is insufficient` 之类的运行时错误，这是预期行为——编译已经成功，只是没有设备可执行）。
- `stage4_fuse_gemm_gelu_test`：不需要 GPU，任何装了 LLVM/MLIR 的机器都能跑通。

### 4. 静态指令级验证（不需要 GPU，任何装了 nvcc 的机器都能跑）

```bash
./02_cutlass_gemm/verify_mma_ptx.sh     # 确认生成 PTX 含 mma.sync.aligned
./03_fastgelu_epilogue/verify_sfu_sass.sh  # 确认 PTX/SASS 含 ex2.approx / rcp.approx (MUFU.EX2/RCP)
```

### 5. 真机 Profile（阶段一验收标准，需要真实 GPU + nsys/ncu）

```bash
./01_baseline_cuda/profile.sh build/01_baseline_cuda/stage1_correctness_test
```

### 一键跑完 2-5 步

```bash
./scripts/build_all.sh
```

## 技术要点速览

- **数据类型/Layout**（阶段二）：A/B 为 `cutlass::half_t`（FP16）Row-Major，累加器/输出为 FP32 Row-Major，与阶段一的 CPU 参考实现共用同一套正确性比较工具（`common/matrix_ref.hpp`）。
- **Threadblock/Warp Tiling**（阶段二/三）：`ThreadblockShape<128,128,32>` / `WarpShape<64,64,32>` / `InstructionShape<8,8,4>`（Volta HMMA `mma.sync.aligned.m8n8k4`），`NumStages=2`（Volta 无 `cp.async`，不支持深层软件流水线）。
- **Fast-GELU Inline PTX**（阶段三）：`03_fastgelu_epilogue/fast_gelu_ptx.cuh` 中的 `FastGeluPTX` 与 spec 给出的代码完全一致；`fast_gelu_epilogue_op.cuh` 把它包装成与 `cutlass::epilogue::thread::LinearCombination` 接口兼容的 Epilogue functor，直接替换 CUTLASS `device::Gemm` 模板的 `EpilogueOutputOp` 参数。
- **MLIR Pattern Fusion**（阶段四）：`04_compiler_integration/lib/FuseGemmGeluPattern.cpp` 在 Bufferization 之后的 `func.func` 内扫描 "matmul 的输出唯一地被一个全 parallel、identity-map 的 `linalg.generic` 消费" 这一结构模式；`lib/EmitExternalCall.cpp` 负责校验操作数的 rank/元素类型/内存布局（对应 spec 风险点"IR 语义不对齐"），并把匹配到的两个算子替换成对 `@cutlass_fused_gemm_gelu` 的 `func.call`。`runtime/cutlass_fused_gemm_gelu_c_api.cu` 把阶段三的 kernel 包装成这个符号对应的 C ABI，编译进 `libcutlass_fused_gemm_gelu_runtime.so`。

## 关键风险与排查（对应 spec 第 4 节）

- **寄存器溢出**：`verify_mma_ptx.sh` / `verify_sfu_sass.sh` 会顺带统计生成 PTX 中 `ld.local`/`st.local` 的出现次数；当前配置下均为 0。
- **访存合并失败**：建议在真机上结合 `ncu` 的 `l1tex__data_bank_conflicts` 等指标，以及 SASS 里 `LDG.128`/`STG.128` 的向量化情况进行确认。
- **IR 语义不对齐**：`EmitExternalCall.cpp` 的 `validateOperand()` 在 emit 调用前严格检查 rank==2、元素类型为 f32、内存布局为 identity（连续 row-major），任何不匹配都会让 pass 报编译期错误而不是静默生成错误的调用。

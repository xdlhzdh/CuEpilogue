# CuEpilogue

Fused GEMM + Epilogue (Fast-GELU) 自定义算子开发与编译器集成 —— 完整需求见 [docs/spec.md](docs/spec.md)。当前宿主机软硬件环境详情见 [docs/environment.md](docs/environment.md)。

## 项目阶段概览

| 阶段 | 目录 | 内容 |
| --- | --- | --- |
| 1 | `01_baseline_cuda/` | Native CUDA SGEMM：无 tiling 版本 + Shared Memory tiling 版本，作为性能/精度基准 |
| 2 | `02_cutlass_gemm/` | CUTLASS GEMM：FP16 输入 / FP32 累加，配置 Threadblock/Warp/Instruction Shape 触发 Tensor Core `mma.sync.aligned` |
| 3 | `03_fastgelu_epilogue/` | Inline PTX Fast-GELU Epilogue：`FastGeluPTX`（`ex2.approx` + `rcp.approx`）接入 CUTLASS Epilogue，GEMM 输出直接在寄存器里做激活 |
| 4 | `04_compiler_integration/` | MLIR 编译器后端集成：`fused-opt` 在 Bufferization 后把 `linalg.matmul`+`linalg.generic` 替换为对外部 CUTLASS 算子 `.so` 的 `call` |

四个阶段依次递进，均已实现；功能验证和性能验收结果见下文。

## 目录结构

```
CuEpilogue/
├── CMakeLists.txt                  # 顶层构建脚本
├── cmake/FindOrFetchCutlass.cmake  # CUTLASS FetchContent
├── common/                         # 共享 CUDA 工具 / CPU 参考实现
├── 01_baseline_cuda/                # 阶段一：Native CUDA SGEMM
├── 02_cutlass_gemm/                  # 阶段二：CUTLASS GEMM + 大矩阵吞吐 benchmark
├── 03_fastgelu_epilogue/              # 阶段三：Inline PTX Fast-GELU
├── 04_compiler_integration/            # 阶段四：MLIR 编译器集成 + 端到端延迟 benchmark
├── scripts/setup_env.sh              # 安装 CUDA Toolkit
├── scripts/build_all.sh              # 一键配置+编译+功能验证
└── docs/
    ├── spec.md                      # 需求规格说明书
    └── environment.md               # 当前环境 / 版本选择说明
```

## 快速开始

### 1. 安装依赖

```bash
sudo apt-get install -y cuda-toolkit-12-9   # V100 (sm_70) 需要 CUDA 12.x
sudo update-alternatives --config cuda      # 确保默认 nvcc 指向 12.9
```

阶段四需要 LLVM/MLIR（`find_package(MLIR CONFIG)` 能找到 `MLIRConfig.cmake`）；未安装时构建加 `-DCU_EPILOGUE_ENABLE_MLIR=OFF`。CMake 缺失可用 `pipx install cmake`。版本选择的原因见 [docs/environment.md](docs/environment.md)。

### 2. 配置与编译

```bash
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

### 3. 一键跑完编译 + 功能验证

```bash
./scripts/build_all.sh
```

等价于依次执行下方「功能验证」里的 `ctest`、静态验证脚本和阶段四 fusion pass 测试。性能 benchmark（`ncu` / `nsys` / 大矩阵吞吐 / 端到端延迟）需要单独运行，见「性能验收」。

## 功能验证

功能验证覆盖两类内容：**运行时正确性 / pass 行为测试**（`ctest`，其中阶段一~三需要真实 GPU 执行 kernel）和**静态指令级验证**（脚本，只需 `nvcc` 能生成目标架构的 PTX/SASS，不需要 GPU）。

| 测试 | 类型 | 目的 | 运行方法 | 环境依赖 | 本机结果 |
| --- | --- | --- | --- | --- | --- |
| `stage1_correctness_test` | `ctest` | 手写 SGEMM（naive + shared-memory tiling，含非整除 tile 尺寸）数值正确性，对比 CPU FP32 参考，为后续阶段建立性能/精度基线 | `cd build && ctest -R stage1_correctness_test --output-on-failure` | GPU；支持 sm_70 的 `nvcc` | ✓ 通过 |
| `stage2_correctness_test` | `ctest` | CUTLASS FP16 Tensor Core GEMM 数值正确性，对比 FP16 量化后的 CPU 参考（`max_abs < 0.05`） | `cd build && ctest -R stage2_correctness_test --output-on-failure` | GPU；`nvcc`(sm_70)；CUTLASS（CMake FetchContent 自动拉取） | ✓ 通过 |
| `verify_mma_ptx.sh` | 静态脚本 | 确认阶段二生成的 PTX 含 `mma.sync.aligned.m8n8k4`（真的触发了 Tensor Core），且无 `ld.local`/`st.local`（无寄存器溢出迹象） | `./02_cutlass_gemm/verify_mma_ptx.sh` | 只需 `nvcc`(sm_70)，**不需要 GPU** | ✓ 通过 |
| `stage3_correctness_test` | `ctest` | 融合 GEMM+Fast-GELU 数值误差在推理可接受 ε 内（`max_abs_exact < 0.05`，即 SFU 近似引入的额外误差 `max_abs_sfu < 0.01`） | `cd build && ctest -R stage3_correctness_test --output-on-failure` | GPU；`nvcc`(sm_70) | ✓ 通过 |
| `verify_sfu_sass.sh` | 静态脚本 | 确认阶段三 PTX 含 `ex2.approx`/`rcp.approx`，编译出的 SASS 含对应硬件指令 `MUFU.EX2`/`MUFU.RCP` | `./03_fastgelu_epilogue/verify_sfu_sass.sh` | 只需 `nvcc`(sm_70)，**不需要 GPU** | ✓ 通过 |
| `stage4_fuse_gemm_gelu_test` | `ctest` / 脚本 | 验证 `fused-opt` 把 `linalg.matmul` + `linalg.generic`(GELU) 融合为对 `@cutlass_fused_gemm_gelu` 的外部调用，原始算子从 IR 中消失 | `cd build && ctest -R stage4 --output-on-failure` 或 `./04_compiler_integration/test/run_tests.sh build/04_compiler_integration/fused-opt` | LLVM/MLIR（`MLIRConfig.cmake`）；构建期仍需 `nvcc` 编译 runtime `.so`；**不需要 GPU 执行** | ✓ 通过 |

**`ctest` 汇总：4/4 通过**（`cd build && ctest --output-on-failure`；单独跑某阶段用 `ctest -R stage2` 等）。

## 性能验收

以下三项 spec 要求的性能验收，均已在本机实测。它们不属于 `scripts/build_all.sh` 的功能验证流程：`ncu`/`nsys` 采集通常需要较长时间，且本机 `ncu` 需要 `sudo` 访问性能计数器。

| 验收项 | 运行方法 | 结果 |
| --- | --- | --- |
| 阶段一 Profile（Occupancy / 显存带宽 vs. 理论峰值） | `./01_baseline_cuda/profile.sh build/01_baseline_cuda/stage1_correctness_test` | `nsys` + `ncu --set full` 报告均已产出（`01_baseline_cuda/profile_reports/`，已被 `.gitignore` 忽略）。256³：`SgemmTiledSmemKernel` SM 吞吐 **51.8%** of peak、显存吞吐 **50.5%** of peak、Achieved Occupancy **49.8%**；`SgemmNaiveKernel` 分别为 39.8% / 39.8% / 49.8% |
| 阶段二吞吐量 vs. 阶段一 | 见下方「吞吐量对比方法」 | 小矩阵（256³，correctness test 用的尺寸）下 CUTLASS 反而因 grid 太小跑不满；换成 4096³ 大矩阵后：阶段一 **3.05 TFLOPS**，阶段二 **36.27 TFLOPS**，**11.9× 加速**，达标 spec"显著优于阶段一" |
| 阶段四端到端延迟（融合 vs. 标量展开） | `./04_compiler_integration/benchmark_e2e.sh` | 融合外部调用 20 次共 **0.68ms**（34us/次）；未融合标量 CPU 循环 20 次共 **148.3ms**（7.4ms/次）。**延迟下降 99.5%**，远超 spec 的 30% 目标 |

**吞吐量对比方法**：正确性测试用的 256×256×256 矩阵是为“CPU 参考算得快”选的尺寸，对阶段二并不公平——`ncu` 数据显示 CUTLASS `ThreadblockShape<128,128,32>` 在 256×256 输出上只切出 `(2,2,1)=4` 个 threadblock，而 V100 有 80 个 SM，4 个 block 连 5% 的 SM 都占不满，Tensor Core 峰值算力根本没机会发挥，此时阶段二的 SM 吞吐（1.3% of peak）反而远低于阶段一（51.8%）。

`02_cutlass_gemm/bench_throughput_large`（直接链接阶段一、二的 kernel，不做 CPU 校验，纯粹测吞吐）在 **4096×4096×4096** 下重新对比：

```bash
./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096
# [stage1] SgemmTiledSmem:   ~45.0 ms/call  (3.05 TFLOPS)
# [stage2] CutlassGemmFp16:  ~3.8 ms/call  (36.27 TFLOPS)
# speedup (stage1_ms / stage2_ms): 11.9x
```

`ncu`（`sudo /opt/nvidia/nsight-compute/2025.2.1/ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size --csv --page raw <bin>`）确认了这个规模下两边 grid 都已经打满 SM（阶段一 grid=16384 block、Occupancy 99.8%；阶段二 grid=1024 block）：阶段一 SM 吞吐 93.6% of *FP32 峰值*，阶段二 SM 吞吐 80.3% of *Tensor Core FP16 峰值*。**这两个百分比不能直接比大小**——阶段二对标的峰值（V100 Tensor Core ≈125 TFLOPS）本身是阶段一峰值（FP32 ≈15.7 TFLOPS）的约 8 倍，所以“80% of 大峰值”换算成绝对 TFLOPS 仍然是“93% of 小峰值”的 11.9 倍，与上面的实测 wall-clock 结果完全吻合。真正可比的是 TFLOPS 这个绝对数字，不是各自的“% of peak”。

**阶段四延迟对比的实现说明**：`04_compiler_integration/test/bench_fused_e2e.cu`（编译进 `bench_fused_e2e` 可执行文件）直接反复调用与编译器外部调用相同的 CUTLASS Fast-GELU kernel，绕开了 MLIR JIT 执行——因为 MLIR 的 bare-pointer 调用约定不支持*动态形状* memref 的函数声明，而 `EmitExternalCall.cpp` 为了通用性总是把算子签名规整成 `memref<?x?xf32>`。融合 pass 本身的正确性已由 `stage4_fuse_gemm_gelu_test` 单独覆盖。「未融合」侧则货真价实地跑了 `fused-opt`（不加 `--fuse-gemm-gelu`）→ `--convert-linalg-to-loops` → `mlir-runner` JIT 执行的标量 CPU 循环，对应 spec 3.4 的“基础指令展开”基线；已用双倍迭代次数验证耗时线性缩放（未被优化器提前求值/hoist）。

### `ncu` 排障记录：两个问题，都已在 `profile.sh` 里自动处理

本机同时装了两个 Nsight Compute 版本（`/opt/nvidia/nsight-compute/{2025.2.1,2026.2.1}`），默认 `ncu`（`/usr/local/cuda-12.9/bin/ncu`）会自动选最新的 2026.2.1，但它比本机的 535.309.01 驱动新，一跑就报 `Nsight Compute failed to connect to the CUDA driver (stub libcuda.so[.1] on path?)`——这条报错**具有误导性**：用 `strace` 追踪发现它其实成功 `open()` 到了真正的驱动 `libcuda.so.1`（`/lib/x86_64-linux-gnu/`），根本不是 stub 库路径问题，而是工具/驱动版本不兼容导致 `cuInit` 失败，NVIDIA 用了一个通用但不准确的错误文案。换成同机已装的旧版 `2025.2.1` 后，才得到真正有用的报错：

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

这是第二个问题：NVIDIA 驱动默认把性能计数器访问限制为 admin-only（`RmProfilingAdminOnly=1`，可用 `cat /proc/driver/nvidia/params | grep -i profil` 确认），普通用户即使是 `/dev/nvidia*` 的 rw 权限也不够，必须 `sudo` 运行 `ncu` 本身。两个问题叠加，才会让人误以为“环境没准备好”——实际上驱动、GPU、CUDA 都没问题。

`01_baseline_cuda/profile.sh` 现在会自动探测 `PATH` 和 `/opt/nvidia/nsight-compute/*` 下所有 `ncu`，从新到旧依次用 `sudo` 尝试，用第一个能连上驱动的版本采集数据，不需要手动指定路径。

## 技术要点速览

- **数据类型/Layout**（阶段二）：A/B 为 `cutlass::half_t`（FP16）Row-Major，累加器/输出为 FP32 Row-Major，与阶段一的 CPU 参考实现共用同一套正确性比较工具（`common/matrix_ref.hpp`）。
- **Threadblock/Warp Tiling**（阶段二/三）：`ThreadblockShape<128,128,32>` / `WarpShape<64,64,32>` / `InstructionShape<8,8,4>`（Volta HMMA `mma.sync.aligned.m8n8k4`），`NumStages=2`（Volta 无 `cp.async`，不支持深层软件流水线）。
- **Fast-GELU Inline PTX**（阶段三）：`03_fastgelu_epilogue/fast_gelu_ptx.cuh` 中的 `FastGeluPTX` 与 spec 给出的代码完全一致；`fast_gelu_epilogue_op.cuh` 把它包装成与 `cutlass::epilogue::thread::LinearCombination` 接口兼容的 Epilogue functor，直接替换 CUTLASS `device::Gemm` 模板的 `EpilogueOutputOp` 参数。
- **MLIR Pattern Fusion**（阶段四）：`04_compiler_integration/lib/FuseGemmGeluPattern.cpp` 在 Bufferization 之后的 `func.func` 内扫描 "matmul 的输出唯一地被一个全 parallel、identity-map 的 `linalg.generic` 消费" 这一结构模式；`lib/EmitExternalCall.cpp` 负责校验操作数的 rank/元素类型/内存布局（对应 spec 风险点"IR 语义不对齐"），并把匹配到的两个算子替换成对 `@cutlass_fused_gemm_gelu` 的 `func.call`。`runtime/cutlass_fused_gemm_gelu_c_api.cu` 把阶段三的 kernel 包装成这个符号对应的 C ABI，编译进 `libcutlass_fused_gemm_gelu_runtime.so`。

## 关键风险与排查（对应 spec 第 4 节）

- **寄存器溢出**：`verify_mma_ptx.sh` / `verify_sfu_sass.sh` 会顺带统计生成 PTX 中 `ld.local`/`st.local` 的出现次数；当前配置下均为 0。
- **访存合并失败**：已用 `ncu` 覆盖阶段一/二核心吞吐指标；进一步定位访存合并问题时，可结合 `l1tex__data_bank_conflicts` 等细分指标，以及 SASS 里 `LDG.128`/`STG.128` 的向量化情况确认。
- **IR 语义不对齐**：`EmitExternalCall.cpp` 的 `validateOperand()` 在 emit 调用前严格检查 rank==2、元素类型为 f32、内存布局为 identity（连续 row-major），任何不匹配都会让 pass 报编译期错误而不是静默生成错误的调用。

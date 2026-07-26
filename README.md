# CuEpilogue

Fused GEMM + Epilogue (Fast-GELU) 自定义算子开发与编译器集成 —— 完整需求见 [docs/spec.md](docs/spec.md)。当前宿主机软硬件环境详情见 [docs/environment.md](docs/environment.md)。

## 项目主线

本项目按四个阶段递进：先用手写 CUDA SGEMM 建立基线，再用 CUTLASS Tensor Core GEMM 提升吞吐，随后把 Fast-GELU 注入 CUTLASS Epilogue，最后用 MLIR pass 把 `linalg.matmul + linalg.generic` 改写成对外部 CUTLASS fused kernel 的调用。

| 阶段 | 目录 | 内容 |
| --- | --- | --- |
| 1 | `01_baseline_cuda/` | Native CUDA SGEMM：无 tiling 版本 + Shared Memory tiling 版本，作为性能/精度基准 |
| 2 | `02_cutlass_gemm/` | CUTLASS GEMM：FP16 输入 / FP32 累加，配置 Threadblock/Warp/Instruction Shape 触发 Tensor Core `mma.sync.aligned` |
| 3 | `03_fastgelu_epilogue/` | Inline PTX Fast-GELU：functor（Sm70 `device::Gemm`）与 Sm90 visitor（`CollectiveBuilder` + `LinCombEltAct`）并存；阶段四默认走 functor |
| 4 | `04_compiler_integration/` | MLIR 编译器后端集成：`fused-opt` 在 Bufferization 后把 `linalg.matmul`+`linalg.generic` 替换为对外部 CUTLASS 算子 `.so` 的 `call` |

四个阶段依次递进，均已实现；每个阶段的目的、运行方法和验收结果见下文「验收结果总览」。

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
| `CU_EPILOGUE_CUDA_ARCH` | `70` | 目标 SM 架构（70/80/86/89/90...）；作用于阶段一～三 functor 路径 |
| `CU_EPILOGUE_ENABLE_CUDA` | `ON` | 是否构建阶段一~三（需要 `nvcc`，找不到会自动关闭并给出警告） |
| `CU_EPILOGUE_ENABLE_SM90_VISITOR` | `ON` | 是否构建阶段三 Sm90 visitor 旁路（单独编 `sm_90a`；设备 correctness 需 Hopper） |
| `CU_EPILOGUE_ENABLE_MLIR` | `ON` | 是否构建阶段四（需要 `find_package(MLIR)` 成功） |
| `CU_EPILOGUE_BUILD_TESTS` | `ON` | 是否注册 `ctest` 测试 |

### 3. 一键跑完编译 + 功能验证

```bash
./scripts/build_all.sh
```

等价于依次执行下方「验收结果总览」里的 `ctest`、静态验证脚本和阶段四 fusion pass 测试。`ncu` / `nsys` profile、大矩阵吞吐和端到端延迟 benchmark 耗时更长，需要单独运行。

## 验收结果总览

下面按阶段列出验收项。阶段一、二、四侧重正确性 + 性能；阶段三侧重**把 Fast-GELU 融进 CUTLASS 矩阵乘、确认走了 SFU 硬件指令、数值误差在 spec 允许范围内**——它不是「再跑一遍吞吐 benchmark」，融合带来的延迟收益在阶段四体现。

| 阶段 | 阶段目的 | 验收方法 | 当前结果 |
| --- | --- | --- | --- |
| 1. Native CUDA SGEMM baseline | 用 naive / shared-memory tiled SGEMM 建立正确性、性能和 profile 基线 | `cd build && ctest -R stage1_correctness_test --output-on-failure`；`./01_baseline_cuda/profile.sh build/01_baseline_cuda/stage1_correctness_test` | `ctest` 通过；`nsys` + `ncu --set full` 报告已产出。256³ 下 `SgemmTiledSmemKernel` SM 吞吐 **51.8%** of peak、显存吞吐 **50.5%**、Achieved Occupancy **49.8%** |
| 2. CUTLASS Tensor Core GEMM | 用 CUTLASS FP16 Tensor Core GEMM 替代阶段一手写 FP32 kernel，并验证吞吐显著提升 | `cd build && ctest -R stage2_correctness_test --output-on-failure`；`./02_cutlass_gemm/verify_mma_ptx.sh`；`./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096` | `ctest` 通过；PTX 含 `mma.sync.aligned.m8n8k4`；4096³ 下阶段一 **3.05 TFLOPS**、阶段二 **36.27 TFLOPS**，**11.9× 加速** |
| 3. Fast-GELU Epilogue | 把 Fast-GELU 融合进 CUTLASS Epilogue，GEMM 输出在寄存器路径中完成激活，避免中间张量落显存 | Functor：`ctest -R stage3_correctness_test`；`./03_fastgelu_epilogue/verify_sfu_sass.sh`。Visitor：`./03_fastgelu_epilogue/verify_sfu_visitor_sm90.sh`；`ctest -R stage3_visitor`（cc&lt;90 时 SKIP） | Functor：`ctest` 通过；PTX/SASS 含 SFU。Visitor：`sm_90a` 下 PTX 含 `ex2.approx`/`rcp.approx`，SASS 含 `MUFU.EX2`/`MUFU.RCP`；V100 上 correctness SKIP |
| 4. MLIR compiler integration | 在 MLIR 后端识别 `linalg.matmul + linalg.generic`，改写为对外部 fused CUTLASS kernel 的调用 | `cd build && ctest -R stage4 --output-on-failure`；`./04_compiler_integration/benchmark_e2e.sh` | fusion pass 测试通过；20 次 fused 调用 **0.68ms**，未融合标量 CPU 循环 **148.3ms**，端到端延迟下降 **99.5%** |

`ctest` 汇总：`cd build && ctest --output-on-failure` 当前 4/4 通过。

### 阶段二：为什么 `ctest` 里阶段二看起来反而更慢？

一句话：**256×256 对 CUTLASS 来说太小了，GPU 大部分时间在空转，不能代表真实吞吐。**

`stage1/2_correctness_test` 用 256³，是因为 CPU 参考实现也得在几秒内算完。但这个尺寸对阶段二不公平：

- CUTLASS 每个 threadblock 一次处理 128×128 的 tile → 256×256 的输出矩阵只需要 **4 个 block**
- V100 有 **80 个 SM**，4 个 block 远远不够把 GPU 喂饱
- 所以 `ncu` 里阶段二 SM 利用率只有 **1.3%**，反而低于阶段一（**51.8%**）——这不是 CUTLASS 不如手写 SGEMM，而是**矩阵太小，Tensor Core 没机会发力**

要看阶段二的真实表现，跑大矩阵吞吐 benchmark（不做 CPU 校验，只计时）：

```bash
./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096
# [stage1] SgemmTiledSmem:   ~45.0 ms/次  (3.05 TFLOPS)
# [stage2] CutlassGemmFp16:  ~3.8 ms/次   (36.27 TFLOPS)
# 加速比: 11.9×
```

4096³ 下两边都能占满 GPU（阶段一约 16384 个 block、Occupancy 99.8%；阶段二约 1024 个 block）。读 `ncu` 时**不要直接比两边的 `% of peak`**——阶段一算 FP32 峰值，阶段二算 Tensor Core FP16 峰值，分母不同。**比实际耗时和 TFLOPS 就行。**

### 阶段三：为什么不单独做吞吐 benchmark？

阶段三要证明的不是「矩阵乘再快一倍」，而是两件事：

1. **GELU 融进了 GEMM**——在寄存器里做完 `alpha·acc + beta·C` 后直接算 GELU，不必先把 GEMM 结果写回显存再读一遍
2. **确实走了 SFU 硬件路径**（`ex2.approx` / `rcp.approx`），且数值误差在推理可接受范围内

所以验收靠 `stage3_correctness_test`（functor 数值）和 `verify_sfu_sass.sh`（PTX/SASS），外加旁路的 `verify_sfu_visitor_sm90.sh` / `stage3_visitor_correctness_test`（Hopper 或 SKIP）。和「分开做 GEMM + GELU」比快慢，放到阶段四；阶段四默认仍调用 functor 路径的 fused kernel。

### 阶段四：0.68 ms 和 148 ms 分别测了什么？

| 对比项 | 实际在测什么 |
| --- | --- |
| **融合路径（0.68 ms / 20 次）** | 直接反复调用阶段三的 CUTLASS Fast-GELU kernel——和编译器融合后要调用的 CUDA 代码一模一样 |
| **未融合路径（148 ms / 20 次）** | MLIR 把 `matmul + gelu` 展开成标量循环，在 CPU 上 JIT 执行——对应 spec 的「基础指令展开」基线，故意代表「没融合时的慢路径」 |

融合侧为什么没走 `mlir-runner`？MLIR 的 bare-pointer 调用约定不支持动态形状 memref 的函数声明，而融合 pass 为了适配不同矩阵尺寸，签名写成了 `memref<?x?xf32>`。融合 pass 改 IR 对不对，已由 `stage4_fuse_gemm_gelu_test` 单独验证；这个 benchmark 只回答一个问题：**融合后的 GPU kernel 比展开循环快多少**。

未融合侧已用双倍迭代次数确认耗时线性增长，排除「循环被编译器提前算完」的假象。

### `ncu` 排障记录

本机同时装了两个 Nsight Compute 版本（`/opt/nvidia/nsight-compute/{2025.2.1,2026.2.1}`），默认 `ncu`（`/usr/local/cuda-12.9/bin/ncu`）会自动选最新的 2026.2.1，但它比本机的 535.309.01 驱动新，一跑就报 `Nsight Compute failed to connect to the CUDA driver (stub libcuda.so[.1] on path?)`——这条报错**具有误导性**：用 `strace` 追踪发现它其实成功 `open()` 到了真正的驱动 `libcuda.so.1`（`/lib/x86_64-linux-gnu/`），根本不是 stub 库路径问题，而是工具/驱动版本不兼容导致 `cuInit` 失败，NVIDIA 用了一个通用但不准确的错误文案。换成同机已装的旧版 `2025.2.1` 后，才得到真正有用的报错：

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

这是第二个问题：NVIDIA 驱动默认把性能计数器访问限制为 admin-only（`RmProfilingAdminOnly=1`，可用 `cat /proc/driver/nvidia/params | grep -i profil` 确认），普通用户即使是 `/dev/nvidia*` 的 rw 权限也不够，必须 `sudo` 运行 `ncu` 本身。两个问题叠加，才会让人误以为“环境没准备好”——实际上驱动、GPU、CUDA 都没问题。

`01_baseline_cuda/profile.sh` 现在会自动探测 `PATH` 和 `/opt/nvidia/nsight-compute/*` 下所有 `ncu`，从新到旧依次用 `sudo` 尝试，用第一个能连上驱动的版本采集数据，不需要手动指定路径。

## 技术要点速览

- **阶段一：手写 CUDA 矩阵乘**（`01_baseline_cuda/`）：`sgemm_naive.cu` 是最直白的写法——一个线程算结果矩阵里的一个数，K 维每走一步就从显存读一次 A、B，重复读很多。`sgemm_tiled_smem.cu` 改进为先把一小块 A/B 搬进 shared memory，块内线程复用这块数据，少读显存。算得对不对，用 `common/matrix_ref.hpp` 里的 CPU 参考实现对照（内部用 double 累加，避免参考本身有误差）；后面阶段沿用同一套比对工具。
- **阶段二：换成 CUTLASS + Tensor Core**（`02_cutlass_gemm/cutlass_gemm.cu`）：不再手写乘加循环，改用 NVIDIA 的 CUTLASS 库，让 V100 的 Tensor Core 干矩阵乘（编译后能看到 `mma.sync.aligned.m8n8k4` 指令）。输入矩阵用 FP16 省带宽，累加和输出仍用 FP32 保精度。一次处理 128×128 的子块（`ThreadblockShape<128,128,32>`）；Volta 没有 `cp.async`，流水线只能做 2 级，没法像新卡那样叠很多层预取。阶段三在**同一套矩阵乘配置**上，只改「乘完之后干什么」。
- **阶段三：乘完立刻算 GELU，不落显存**（`03_fastgelu_epilogue/`）：普通做法是 GEMM 先把结果写回显存，再另起一个 kernel 做 GELU。这里把 GELU 塞进 CUTLASS 的 epilogue——矩阵乘累加完成后，在寄存器里直接做 `α·acc + β·C`，紧接着用 `FastGeluPTX`（`fast_gelu_ptx.cuh`）算 GELU，全程不经过中间张量。GELU 走 GPU 的 SFU 近似指令（`ex2.approx` / `rcp.approx`），`fast_gelu_epilogue_op.cuh` 把它包装成 CUTLASS 能识别的 epilogue 接口，替换原来的线性组合输出。
- **阶段四：编译器认出「矩阵乘 + GELU」并合成一次调用**（`04_compiler_integration/`）：`FuseGemmGeluPattern.cpp` 在 MLIR 里找这样的模式——`linalg.matmul` 的输出**只**被一个逐元素激活（GELU）消费，且两者已经 bufferize 成内存上的张量。匹配成功就把两个算子删掉，改成调用外部函数 `@cutlass_fused_gemm_gelu`。`EmitExternalCall.cpp` 在生成调用前检查张量是不是 2D、是不是 f32、布局对不对，不对就直接报错，避免生成错误的 C 接口。`runtime/cutlass_fused_gemm_gelu_c_api.cu` 把阶段三的 kernel 包成这个外部函数，打进 `libcutlass_fused_gemm_gelu_runtime.so`，供 MLIR 生成的代码链接调用。

## 关键风险与当前状态（对应 spec 第 4 节）

本节列的是 spec 要求关注的三类技术风险，以及本项目当前的排查结果。**“风险”指需要警惕的问题类型，不等于当前一定出了 bug。**

| 风险 | 风险是什么 | 当前状态 | 怎么查 / 已有防护 |
| --- | --- | --- | --- |
| 寄存器溢出 | CUTLASS tile 或 epilogue 过大时，线程寄存器不够用，编译器会把数据溢出到 local memory，kernel 变慢、occupancy 下降 | **已验证，当前无溢出** | `verify_mma_ptx.sh` / `verify_sfu_sass.sh` 统计 PTX 中 `ld.local` / `st.local`；当前配置下均为 **0**。若以后改 tile 配置，需重跑这两个脚本 |
| 访存合并失败 | Epilogue 写回 global memory 时未形成高效向量化访问，带宽利用变差 | **核心指标已验证，细分项可继续深入** | 阶段一/二已有 `ncu` 核心吞吐数据；若继续定位，可看 `l1tex__data_bank_conflicts`、global load/store transaction，以及 SASS 中 `LDG.128` / `STG.128` |
| IR 语义不对齐 | MLIR 侧 memref 的 rank、dtype、stride/layout 与底层 CUTLASS C API 不一致，可能生成错误外部调用 | **已有编译期防护** | `EmitExternalCall.cpp::validateOperand()` 要求 rank==2、f32、identity layout；不满足则 pass 直接报错，不会静默生成错误 ABI |
| 动态形状 memref JIT 调用 | MLIR bare-pointer 约定下，动态形状 memref 难以直接 JIT 调用外部 fused kernel | **已在 benchmark 设计中规避** | 阶段四 fused 性能侧用 `bench_fused_e2e` 直接调 C++ kernel；fusion pass 行为由 `stage4_fuse_gemm_gelu_test` 单独覆盖 |

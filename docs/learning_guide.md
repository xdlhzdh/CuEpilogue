# CuEpilogue 学习手册

这份手册面向想读懂并复现实验的人。`README.md` 负责快速说明工程状态；本文件负责解释为什么这样设计、重点代码在哪里、应该按什么顺序学习。

## 1. 这个工程在解决什么问题

AI 推理里常见的计算形态是：

```text
D = GELU(A @ B)
```

如果普通 lowering 把 GEMM 和 GELU 拆成两个 kernel，中间矩阵会先写回显存，再被 GELU kernel 读回来。这个项目的目标是把 GEMM 和 Fast-GELU 融合成一个 CUDA/CUTLASS kernel，并让 MLIR 认出 `linalg.matmul + linalg.generic` 后直接调用这个 kernel。

整体路线：

```text
手写 CUDA SGEMM
  -> CUTLASS Tensor Core GEMM
  -> 把 Fast-GELU 塞进 GEMM 的 epilogue
  -> MLIR 在 memref 上把 FP32 GEMM+GELU 收成一次外部调用（阶段四）
  -> MLIR 在 tensor 上把量化图收成一次 INT8 外部调用（阶段五）
```

## 2. 先看懂目录


| 路径                         | 重点                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `common/`                  | CUDA error check、CPU reference GEMM、随机输入、误差统计                                             |
| `01_baseline_cuda/`        | 手写 naive SGEMM 和 shared-memory tiled SGEMM；建立 profile 基线                                  |
| `02_cutlass_gemm/`         | CUTLASS Tensor Core **纯 GEMM** 基座（尚未融合 GELU）；`mma.sync` 验证与大矩阵吞吐 benchmark                |
| `03_fastgelu_epilogue/`    | Fast-GELU inline PTX；functor epilogue（Sm70）与 Sm90 visitor（`LinCombEltAct` / EVT）；SFU 指令验证 |
| `04_compiler_integration/` | 阶段四：在 memref 上匹配 GEMM+GELU，改成调用 `.so` |
| `05_qdq_fusion/`           | 阶段五：在 tensor 上匹配量化图，bufferize 后再调用 INT8 kernel |
| `scripts/`                 | 一键构建和功能验证                                                                                 |
| `docs/`                    | spec、环境说明和本学习手册                                                                           |




## 3. 阶段一：CUDA SGEMM Baseline

阶段一不是为了写最快的 GEMM，而是为了建立可解释的基线：先看最直接的 global-memory 版本，再看 shared-memory tiling 如何减少重复读。

重点文件：

- `01_baseline_cuda/sgemm_naive.cu`
- `01_baseline_cuda/sgemm_tiled_smem.cu`
- `01_baseline_cuda/correctness_test.cu`
- `01_baseline_cuda/profile.sh`
- `common/matrix_ref.hpp`

naive 版本中，一个线程计算一个输出元素，每次 K 循环都直接读 global memory：

```cpp
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
  acc += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = alpha * acc + beta * C[row * N + col];
```

tiled 版本把 A/B 的 tile 放入 shared memory，线程块内复用：

```cpp
__shared__ float As[kBlockSize][kBlockSize];
__shared__ float Bs[kBlockSize][kBlockSize];

As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
__syncthreads();

#pragma unroll
for (int k = 0; k < kBlockSize; ++k) {
  acc += As[ty][k] * Bs[k][tx];
}
```

正确性由 CPU reference 负责。`common/matrix_ref.hpp` 用 double 内部累加，避免 CPU reference 本身成为误差来源：

```cpp
double acc = 0.0;
for (int k = 0; k < K; ++k) {
  acc += static_cast<double>(A[m * K + k]) *
         static_cast<double>(B[k * N + n]);
}
C[m * N + n] = static_cast<float>(alpha * acc + beta * prev);
```

建议运行：

```bash
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70
cmake --build build -j$(nproc)
cd build && ctest -R stage1_correctness_test --output-on-failure
cd .. && ./01_baseline_cuda/profile.sh build/01_baseline_cuda/stage1_correctness_test
```

学习重点：

- 为什么 naive kernel 的 global memory 读会成为瓶颈。
- shared memory tiling 如何把 A/B 的重复读取摊薄。
- `nsys` 看时间线，`ncu` 看 SM throughput、memory throughput、occupancy。



## 4. 阶段二：CUTLASS Tensor Core GEMM



### 4.1 这个目录到底在做什么

`02_cutlass_gemm/` **只做一件事**：把手写 FP32 SGEMM（阶段一）换成 CUTLASS 的 device-level Tensor Core GEMM，算出

```text
D = alpha * (A @ B) + beta * C
```

没有 GELU，也没有第二个激活 kernel。目录里的工作可以拆成四块：


| 文件                          | 作用                                                                |
| --------------------------- | ----------------------------------------------------------------- |
| `cutlass_gemm.cu` / `.cuh`  | 配置 `device::Gemm`（Sm70、FP16 输入、FP32 累加），封装 `CutlassGemmFp16` 启动入口 |
| `correctness_test.cu`       | 小矩阵上对比 CPU reference（A/B 先量化到 FP16）                               |
| `verify_mma_ptx.sh`         | 离线编译 PTX，确认出现 `mma.sync.aligned.m8n8k4`（不依赖 GPU 运行）               |
| `bench_throughput_large.cu` | 大矩阵计时，对比阶段一 tiled SGEMM 的 TFLOPS                                  |


`CutlassGemmFp16` 这个函数的调用流程分三步：

1. **对外接口仍是 FP32**：`CutlassGemmFp16(M, N, K, alpha, A_fp32, B_fp32, beta, C_fp32, ...)` 的入参和阶段一的 SGEMM 函数长得一样，全是 `float`*。
2. **内部先做一次 FP32 → FP16 的类型转换**：函数一开始会在 GPU 上额外跑一个小 kernel（`ConvertF32ToF16Kernel`），把 A、B 从 FP32 转成 FP16，存进两块新分配的临时显存里。真正喂给 CUTLASS Tensor Core 的是这两块 FP16 数据，C 全程仍是 FP32（Tensor Core 用 FP16 相乘、FP32 累加）。
3. **调用 CUTLASS**：用转换后的 FP16 A/B 和原来的 FP32 C 构造 `Arguments`，依次调 `can_implement`（检查这个问题规模/类型合不合法）→ `initialize`（分配 workspace、绑定参数）→ `operator()`（真正跑 kernel），结果写回同一块 FP32 C。

为什么对外接口还留着 FP32，而不是直接要求调用者传 FP16 数据？因为这样一来，`correctness_test.cu` 里生成随机数、`cudaMemcpy`、算误差的那套代码，可以跟阶段一测试 SGEMM 时用的完全是同一份（`common/matrix_ref.hpp` 里的 `FillRandom`、`MaxErrors` 等），不用为 FP16 单独写一套测试逻辑，方便阶段一和阶段二对比。但这也意味着每次调用都要多付一次「转换 kernel」的开销；真实产品部署时，模型权重会提前一次性转换成 FP16 存好（即数据本来就「常驻」FP16，不用每次现场转），直接把 FP16 指针传给 GEMM，省掉这一步。

### 4.2 它算不算「算子融合」？

**不算。** 这里的 GEMM 只算出 `D = alpha*(A@B) + beta*C`，没有任何激活函数。项目真正要消除的中间张量，是「GEMM 算完先写回显存 → GELU kernel 再读一遍」这一步，那一步要到**阶段三**才处理。

容易搞混的地方在于：`cutlass_gemm.cu` 的注释里写着 "fused Device-level GEMM"，CUTLASS 官方文档也常说这类 GEMM 是「fused」的。但这里的「fused」指的是 CUTLASS **库内部**的实现方式——一个 GEMM kernel 把 mainloop（做矩阵乘法的 MMA 指令）和 epilogue（默认是 `LinearCombination`，也就是 `alpha*acc+beta*C`）一起编译进同一个 kernel，中间不需要再启动第二个 kernel。这是 CUTLASS 库自身的结构，跟本项目讲的「把 GELU 也融合进去」是两件不同的事，读到「fused」这个词时不要把两者混为一谈。

用一张示意图区分两者：

```text
阶段二：搭好「会跑 Tensor Core 的 GEMM 机架」，epilogue 仍是 LinearCombination（只做线性组合，没有激活函数）
阶段三：把机架上的 epilogue 换成 FastGeluLinearCombination（或 Sm90 visitor），才是本项目说的「算子融合」
```

所以学阶段二的真正目的，是为阶段三准备一个**可以直接换零件的底座**：阶段三的 `fused_gemm_gelu.cu` 几乎原样照抄这里的全部模板参数，只替换其中第 10 个参数（epilogue op）。如果跳过阶段二直接读阶段三，很容易看不出「阶段三到底改了哪一处」。

`docs/spec.md` 里阶段二的标题写了「算子融合」，更准确的说法是：**为算子融合打基础的 CUTLASS 核心重构**；「融合」本身要到阶段三才真正验收。

### 4.3 核心配置（必读）

```cpp
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;

using CutlassGemmSm70 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages>;
```

正确性细节：GPU Tensor Core 吃 FP16，CPU reference 也要先把 A/B 量化到 FP16（`ReferenceGemmFp16InputsRowMajor` / `RoundToFp16`），否则比的是「未量化输入」和「已量化输入」两套不同问题。

吞吐要用大矩阵。256³ 的 correctness 尺寸只有 4 个 CUTLASS threadblock，喂不饱 V100 的 80 个 SM；4096³ 才能体现 Tensor Core：

```bash
./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096
# stage1: 3.05 TFLOPS
# stage2: 36.27 TFLOPS
# speedup: 11.9x
```



### 4.4 建议学习顺序

1. **先建立定位**：明确本目录输出是纯 GEMM，不是 fused GELU；对照打开 `03_fastgelu_epilogue/fused_gemm_gelu.cu`，看哪些 typedef 一字不差、哪里只换成了 `FastGeluEpilogueOp`。
2. **读** `cutlass_gemm.cu`：搞清 `ElementA/B=half_t`、`ElementAccumulator=float`、`OpClassTensorOp`、`Sm70`、`kStages=2`（Volta 无 `cp.async`）各自约束什么。
3. **认 epilogue 插槽**：盯住模板参数里的 `LinearCombination`——阶段三要替换的就是它；此时它只做线性组合。
4. **跑正确性**：`cd build && ctest -R stage2_correctness_test --output-on-failure`，理解为何 reference 要 FP16 预量化、为何 tolerance 比阶段一松。
5. **静态指令验收**：`./02_cutlass_gemm/verify_mma_ptx.sh`，确认 PTX 里有 `mma.sync.aligned.m8n8k4`（不要求 GPU 运行）。
6. **大矩阵吞吐**：`./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096`；读 `ncu` 时分清 FP32 peak 与 Tensor Core FP16 peak 分母不同。
7. **再进阶段三**：带着「epilogue 是可替换的最后一个模板参数」这一印象去读 Fast-GELU 注入。

学习重点自检：

- 能否用一句话说清：阶段二不是融合，阶段三才是。
- `ThreadblockShape` / `WarpShape` / `InstructionShape` 如何落到 `mma.sync.aligned.m8n8k4`。
- 为什么 correctness 用小矩阵、performance 必须用大矩阵。



## 5. 阶段三：Fast-GELU Epilogue

阶段三把激活函数融合进 CUTLASS Epilogue。目标不是再启动一个 GELU kernel，而是在 GEMM accumulator 输出时直接完成：

```text
D = FastGELU(alpha * accumulator + beta * source)
```

同一套 Inline PTX Fast-GELU（`FastGeluPTX`）有两条并存注入路径：


| 路径                     | API                                                     | 编译目标架构                                 | 是否默认路径                                                               | 阶段四会用它吗 |
| ---------------------- | ------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------- | ------- |
| **Functor**（2.x 风格）    | `device::Gemm` + `FastGeluLinearCombination`            | Sm70（跟随 `CU_EPILOGUE_CUDA_ARCH`，默认 70） | 是                                                                    | 会       |
| **Visitor / EVT**（3.x） | `CollectiveBuilder` + `fusion::LinCombEltAct<FastGelu>` | Hopper，编译时需用 `sm_90a`                  | 否，是旁路（`CU_EPILOGUE_ENABLE_SM90_VISITOR` 默认开着，但只表示「默认会被编译」，不代表它是主线路径） | 不会      |


说明：仓库通过 CMake 拉取的是 CUTLASS **v3.5.1**，但 functor 路径用的是其中兼容 2.x 的 `cutlass::gemm::device::Gemm` API，并不是 3.x 的新接口。

真正的 3.x visitor（`FusionCallbacks` / `Sm90EVT`）目前只有 Hopper 架构才完整支持。编译 visitor 目标时，nvcc 的目标架构必须写成 `sm_90a`（注意末尾的 `a`），不能只写 `sm_90`——否则编译器只会生成一个 `printf` 报错的空壳 kernel，不会真正执行任何 WGMMA 计算。

### 5.1 Functor 路径（默认）

重点文件：

- `03_fastgelu_epilogue/fast_gelu_ptx.cuh`
- `03_fastgelu_epilogue/fast_gelu_epilogue_op.cuh`
- `03_fastgelu_epilogue/fused_gemm_gelu.cu`
- `03_fastgelu_epilogue/correctness_test.cu`
- `03_fastgelu_epilogue/verify_sfu_sass.sh`

Fast-GELU 近似公式：

```text
GELU(x) ~= x / (1 + 2^(-2.455492 * x))
```

Inline PTX 重点是 `ex2.approx.f32` 和 `rcp.approx.f32`：

```cpp
asm volatile(
    "{ \n\t"
    " .reg .f32 t, r, e, p; \n\t"
    " mul.f32 t, %1, %2; \n\t"
    " ex2.approx.f32 e, t; \n\t"
    " add.f32 r, e, 1.0; \n\t"
    " rcp.approx.f32 p, r; \n\t"
    " mul.f32 %0, %1, p; \n\t"
    "} \n\t"
    : "=f"(res)
    : "f"(x), "f"(constant));
```

`FastGeluLinearCombination` 的关键是复刻 CUTLASS epilogue output op 的接口，让它能替换 `LinearCombination`：

```cpp
ElementCompute linear = alpha_ * converted_accumulator[i] +
                        beta_ * converted_source[i];
result[i] = static_cast<ElementCompute>(gelu_(static_cast<float>(linear)));
```

建议运行：

```bash
cd build && ctest -R stage3_correctness_test --output-on-failure
cd .. && ./03_fastgelu_epilogue/verify_sfu_sass.sh
```



### 5.2 Visitor 路径（Sm90 EVT 旁路）

重点文件：

- `03_fastgelu_epilogue/fast_gelu_activation.cuh` — CUTLASS 风格 `FastGelu` activation（供 `LinCombEltAct`）
- `03_fastgelu_epilogue/fused_gemm_gelu_visitor_sm90.cu`
- `03_fastgelu_epilogue/correctness_test_visitor.cu` — 如果当前 GPU 的 compute capability 低于 9.0（不是 Hopper），测试会直接 SKIP（exit 0），不算失败
- `03_fastgelu_epilogue/verify_sfu_visitor_sm90.sh` — 只做静态检查：用 `nvcc -arch=sm_90a` 离线生成 PTX/cubin，再用 `grep` / `cuobjdump` 确认里面有没有 SFU 指令；全程不需要真的启动 GPU 或算一次矩阵乘

核心配置示意：

```cpp
using FusionOperation =
    cutlass::epilogue::fusion::LinCombEltAct<FastGelu, ElementD, ElementCompute,
                                             ElementC, ElementScalar>;
using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ...
        EpilogueSchedule, FusionOperation>::CollectiveOp;
```

`LinCombEltAct` 由 CollectiveBuilder 展开为 `Sm90EVT` / `FusionCallbacks` 树：先做 `alpha * acc + beta * C`，再对每个元素调用 `FastGelu`（内部仍是同一套 SFU PTX）。

建议运行：

```bash
cmake -B build -DCU_EPILOGUE_ENABLE_SM90_VISITOR=ON
cmake --build build -j$(nproc) --target fused_gemm_gelu_visitor_sm90
./03_fastgelu_epilogue/verify_sfu_visitor_sm90.sh
cd build && ctest -R stage3_visitor_correctness_test --output-on-failure
```

在 V100 上会发生什么：编译能通过，`verify_sfu_visitor_sm90.sh` 的静态检查也能通过；但 `stage3_visitor_correctness_test` 会打印 SKIP 并直接退出——因为它要求 GPU 的 compute capability ≥ 9.0（即 Hopper 及更新架构），而 V100 只有 7.0，达不到要求。

学习重点：

- 为什么 GELU 放在 epilogue 里可以避免中间张量显存读写。
- 为什么要用 SFU 近似指令，而不是普通 `expf` 展开。
- 数值验收为什么看绝对误差和 SFU 额外误差，而不是只看相对误差。
- Functor（替换 `LinearCombination`）与 Visitor（`LinCombEltAct` → EVT 树）的接口差异，以及为何 3.x visitor 绑定 Hopper `sm_90a`。



## 6. 阶段四：MLIR 编译器集成

阶段四验证这个 fused kernel 能不能作为编译器后端的高性能执行体。输入 IR 中是 `linalg.matmul` 和 elementwise `linalg.generic`，pass 匹配后替换成外部函数调用：

```text
linalg.matmul + linalg.generic
  -> func.call @cutlass_fused_gemm_gelu(...)
```

重点文件：

- `04_compiler_integration/lib/FuseGemmGeluPattern.cpp`
- `04_compiler_integration/lib/EmitExternalCall.cpp`
- `04_compiler_integration/runtime/cutlass_fused_gemm_gelu_c_api.cu`
- `04_compiler_integration/test/run_tests.sh`
- `04_compiler_integration/benchmark_e2e.sh`

Pattern matching 的关键约束：

```cpp
if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
  return false;
if (generic.getInputs()[0] != expectedInput) return false;
if (generic.getNumResults() != 0) return false;

for (utils::IteratorType iter : generic.getIteratorTypesArray())
  if (iter != utils::IteratorType::parallel) return false;

for (AffineMap map : generic.getIndexingMapsArray())
  if (!map.isIdentity()) return false;
```

External call emission 的关键防护是 ABI 校验：只接受 rank-2、f32、identity layout 的 memref。

```cpp
if (type.getRank() != 2)
  return op->emitError() << role << " must be rank-2";
if (!type.getElementType().isF32())
  return op->emitError() << role << " must have f32 elements";
if (!type.getLayout().isIdentity())
  return op->emitError() << role << " must be contiguous row-major";
```

建议运行：

```bash
cd build && ctest -R stage4 --output-on-failure
cd .. && ./04_compiler_integration/benchmark_e2e.sh
```

学习重点：

- 为什么 pass 要等 tensor 变成 memref 再匹配：后面要调的是 C 函数，入参是指针。
- 为什么 matmul 的输出只能被那一个 GELU 用：否则删掉 matmul 会改掉别的观察者看到的值。
- 为什么生成 `func.call` 前要检查 rank / 类型 / 布局：对不上 CUTLASS C ABI 就会静默算错。
- 端到端 benchmark 为什么 fused 侧直接调 C++：MLIR JIT 不好喂动态形状 memref；IR 对不对由 `run_tests.sh` 单独测。



## 7. 阶段五：量化图在 Tensor 上整段融合

几个词：

- **QDQ**：INT8 和 FP32 互转。`x_f32 = (x_i8 - zp) * scale`，写回时再除 scale、加 zp、截断到 INT8。
- **DPS**：结果写进 `outs` 参数，像 C 的输出指针，而不是 `return` 一块新内存。
- **OSB**：One-Shot Bufferize。把 `tensor`（抽象数组）换成 `memref`（内存指针）。

阶段四是在已经变成 memref 之后才匹配 GEMM+GELU，中间 FP32 矩阵这时已经被分配出来了。量化图如果也走这条路，INT8 会先被展开成 FP32 再分配，融合就没意义。所以阶段五在还是 tensor 时就把整条链收成一条 op：

```text
DQ(A)、DQ(B) -> matmul -> +bias -> Fast-GELU -> Q
  -> cutlass.qgemm_bias_gelu
  -> OSB（tensor 换成 memref，只给输出 D 分配）
  -> call @cutlass_qgemm_bias_gelu
```

重点文件：

- `CutlassQGemmOps.td`：声明这条指令长什么样
- `FuseQdqPattern.cpp`：在 tensor IR 上匹配，插入这条指令
- `BufferizableOpInterfaceImpl.cpp`：告诉 OSB 怎么把它换成 memref
- `LowerToCall.cpp`：再换成 `func.call`
- `kernels/qgemm_bias_gelu.cu`、`runtime/cutlass_qgemm_bias_gelu_c_api.cu`：INT8 kernel 和 C 接口
- `test/qdq_pattern.mlir`、`test/run_tests.sh`

### `.td` 和 `.inc`

`.td` 只描述「有一条叫 `cutlass.qgemm_bias_gelu` 的指令」：输入、输出 `D`、怎么打印。匹配和 bufferize 都不写在这里。

CMake 用 `mlir-tblgen` 生成 `build/05_qdq_fusion/*.inc`（不要手改），展开成 C++ 类 `QgemmBiasGeluOp`。fuse / bufferize / lower 都用这个类去 `create` 或读写操作数。

### OSB：对每个 op 调用 `bufferize()`

测试命令：

```bash
qdq-opt test/qdq_pattern.mlir \
  --fuse-qdq-qgemm-bias-gelu \
  --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
  --lower-cutlass-qgemm-to-call \
  --canonicalize
```

OSB 会走遍函数里的每个 op，调用各自的 `bufferize()`。两个选项只控制 **`func.func` 那一步**的行为：

- `bufferize-function-boundaries=1`：默认 OSB 只改函数体内部，函数参数和返回值保持 tensor 不变。开了这个选项，函数签名也会被 bufferize，参数/返回从 tensor 改成 memref。
- `function-boundary-type-conversion=identity-layout-map`：签名里的 memref 用连续布局（`memref<64x64xi8>`），而不是带任意 stride 的版本。后面 `LowerToCall` 校验 identity layout，所以必须这样设。

这两个选项**不影响** `cutlass.qgemm_bias_gelu` 的 bufferize；那条 op 的 `bufferize()` 是我们自己在 `BufferizableOpInterfaceImpl.cpp` 里写的。MLIR 不会给自定义 op 自动生成这个函数，所以 `qdq-opt` 启动时要先挂上：

```cpp
registry.insert<CutlassDialect>();
registerCutlassBufferizationExternalModels(registry);
```

**`--fuse-qdq-qgemm-bias-gelu` 之后，函数体内 IR 是这个样子**（下一步 OSB 的输入）：

```mlir
func.func @f(%A: tensor<64x64xi8>, ...) -> tensor<64x64xi8> {
  %0 = tensor.empty() : tensor<64x64xi8>        // 为输出 D 占位的空 tensor
  %1 = cutlass.qgemm_bias_gelu ins(%A, ...) outs(%0) -> tensor<64x64xi8>
  return %1
}
```

注意：tensor 是值语义，不能原地改。`outs(%0)` 声明 `%0` 是写入目标（DPS），但仍然要有返回值 `%1` 让下游引用。

**OSB 之后**，每个 op 都完成了自己的 `bufferize()`，结果变成：

```mlir
func.func @f(%A: memref<64x64xi8>, ...) -> memref<64x64xi8> {
  %alloc = memref.alloc() : memref<64x64xi8>    // tensor.empty 的 bufferize 结果
  cutlass.qgemm_bias_gelu ins(%A, ...) outs(%alloc)   // 无返回值，直接写进 %alloc
  return %alloc
}
```

三件事分别发生：
1. `func.func` 的 `bufferize()`：参数 `%A` 和返回值从 tensor 改成 memref（受那两个选项控制）
2. `tensor.empty` 的 `bufferize()`：变成 `memref.alloc`
3. `cutlass.qgemm_bias_gelu` 的 `bufferize()`（即 `BufferizableOpInterfaceImpl.cpp`）：`getBuffer` 查出 `%A`、`%0` 各自对应哪块 memref，`create` 一条没有返回值的同名 op，再让原来用 `%1` 的地方改用 `%alloc`

三步各管各的，互不依赖。

zero-point 的行/列和只在 C ABI 里算，不进 MLIR。Volta 没有 INT8 Tensor Core，kernel 用 CUTLASS SIMT INT8 GEMM，再跑 bias + Fast-GELU + 量化。

建议运行：

```bash
cd build && ctest -R stage5 --output-on-failure
./05_qdq_fusion/test/run_tests.sh build/05_qdq_fusion/qdq-opt
```

学习重点：

- 量化图要在变成 memref 之前融合，否则中间 FP32 会被分配出来。
- OSB 对每个 op 调 `bufferize()`；自定义 op 必须自己挂上。
- 函数签名、`tensor.empty`、`cutlass.qgemm_bias_gelu` 各走各的 `bufferize()`；我们写的那份是把返回值收成输出指针。
- memref 形态不要标 `Pure`，否则 canonicalize 会把这条写内存的指令删掉。
- 阶段四是 FP32、memref 上融合；阶段五是 INT8、tensor 上融合。



## 8. 验证与排障清单


| 目标                      | 命令                                                                             | 看什么                                             |
| ----------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------- |
| 全量功能验证                  | `./scripts/build_all.sh`                                                       | 构建成功、静态脚本通过、`ctest`                             |
| 只跑 CUDA correctness     | `cd build && ctest -R stage[1-3] --output-on-failure`                          | 数值误差和 CUDA runtime 是否正常                         |
| 只跑 MLIR pass test       | `cd build && ctest -R 'stage4|stage5_qdq' --output-on-failure` | 阶段四/五的 IR 是否按预期改写 |
| 阶段五 INT8 kernel         | `cd build && ctest -R stage5_qgemm --output-on-failure`                        | GPU 结果是否和 CPU 参考一致 |
| 验证 Tensor Core 指令       | `./02_cutlass_gemm/verify_mma_ptx.sh`                                          | `mma.sync.aligned.m8n8k4`、无 `ld.local/st.local` |
| 验证 SFU 指令（functor）      | `./03_fastgelu_epilogue/verify_sfu_sass.sh`                                    | `ex2.approx`、`rcp.approx`、`MUFU.EX2`、`MUFU.RCP` |
| 验证 SFU 指令（Sm90 visitor） | `./03_fastgelu_epilogue/verify_sfu_visitor_sm90.sh`                            | 同上，且需 `sm_90a` 才有真实 WGMMA 主体                    |
| 阶段一 profile             | `./01_baseline_cuda/profile.sh build/01_baseline_cuda/stage1_correctness_test` | `nsys` timeline、`ncu` SM/memory/occupancy       |
| 阶段二吞吐                   | `./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096`                | TFLOPS 和 speedup                                |
| 阶段四延迟                   | `./04_compiler_integration/benchmark_e2e.sh`                                   | fused/unfused 总耗时和 latency change               |


常见问题：

- CUDA 13.x 不再适合 Volta sm_70 离线编译；V100 建议使用 CUDA 12.9。
- `ncu` 报 stub `libcuda.so` 不一定真是 stub 路径问题，也可能是 Nsight Compute 与驱动版本不兼容。
- `ERR_NVGPUCTRPERM` 表示性能计数器权限不足；本机需要 `sudo ncu`。
- 小矩阵 correctness 结果不能代表 Tensor Core 吞吐；做性能对比要用大矩阵。



## 9. 推荐学习路线与时间安排



### 7 天路线


| 时间    | 任务                                                                                                         | 产出                                            |
| ----- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| 第 1 天 | 阅读 `README.md`、`docs/spec.md`、`docs/environment.md`，完成构建                                                   | 能解释五个阶段和当前环境约束                                |
| 第 2 天 | 学阶段一，读 `sgemm_naive.cu` / `sgemm_tiled_smem.cu` / `matrix_ref.hpp`，跑 stage1 correctness                    | 能解释 naive 与 tiled 的访存差异                       |
| 第 3 天 | 跑 `profile.sh`，看 `nsys`/`ncu` 指标                                                                           | 能读懂 SM throughput、memory throughput、occupancy |
| 第 4 天 | 学阶段二：先弄清「纯 GEMM、不是融合」，对照 `cutlass_gemm.cu` 与 `fused_gemm_gelu.cu` 的差异；跑 `verify_mma_ptx.sh` 和大矩阵 benchmark | 能指出 epilogue 插槽，并解释 Tensor Core 与小/大矩阵差异      |
| 第 5 天 | 学阶段三 functor 路径，读 `FastGeluPTX` 和 epilogue op，跑 SFU 验证；再对照 Sm90 visitor 文件与 `verify_sfu_visitor_sm90.sh`   | 能解释 functor vs visitor 与 SFU 误差验收             |
| 第 6 天 | 学阶段四，读 pattern matching 和 external call emission，跑 stage4 test                                             | 能解释 MLIR IR 如何变成外部 fused call                 |
| 第 7 天 | 学阶段五：对照 `qdq_pattern.mlir` 和三段 `qdq-opt` dump；跑 `ctest -R stage5` | 能说出为什么量化图要在变成 memref 之前融合 |




### 进阶练习

- 把阶段二 benchmark 的矩阵从 2048³、4096³、8192³ 分别跑一遍，观察 TFLOPS 和 `ncu` 指标变化。
- 改 `ThreadblockShape` 或 `WarpShape`，重新跑 `verify_mma_ptx.sh` 和大矩阵 benchmark，比较吞吐变化。
- 在阶段三替换 Fast-GELU 常数或公式，观察 correctness 和 SFU 指令验证是否仍通过。
- 给阶段四增加一个负例 MLIR 测试：非 identity layout 或 rank 不是 2 时，确认 pass 报错而不是生成 external call。
- 给阶段五增加 zp=0（省略 `subf`/`addf`）的 DQ/Q 变体，确认 matcher 仍能抽出 scale 并插入 zp=0 常量。



## 10. 阅读顺序建议

先读 `common/matrix_ref.hpp`，理解 correctness 怎么比。然后按 `01 -> 02 -> 03 -> 04 -> 05` 读。不要先从 MLIR pass 开始：你会看懂 IR 怎么改，却不知道最后调用的 CUDA kernel 为什么对、为什么快。阶段五要在弄清阶段四「已经变成 memref 再融合」的局限之后再读。
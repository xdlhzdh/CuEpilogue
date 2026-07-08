# CuEpilogue 学习手册

这份手册面向想读懂并复现实验的人。`README.md` 负责快速说明工程状态；本文件负责解释为什么这样设计、重点代码在哪里、应该按什么顺序学习。

## 1. 这个工程在解决什么问题

AI 推理里常见的计算形态是：

```text
D = GELU(A @ B)
```

如果普通 lowering 把 GEMM 和 GELU 拆成两个 kernel，中间矩阵会先写回显存，再被 GELU kernel 读回来。这个项目的目标是把 GEMM 和 Fast-GELU 融合成一个高性能 CUDA/CUTLASS kernel，并让 MLIR 后端在识别到 `linalg.matmul + linalg.generic` 时直接发出外部 fused kernel 调用。

整体路线：

```text
Native CUDA baseline
  -> CUTLASS Tensor Core GEMM
  -> CUTLASS Epilogue 注入 Fast-GELU
  -> MLIR pass 发出 external call
```

## 2. 先看懂目录

| 路径 | 重点 |
| --- | --- |
| `common/` | CUDA error check、CPU reference GEMM、随机输入、误差统计 |
| `01_baseline_cuda/` | 手写 naive SGEMM 和 shared-memory tiled SGEMM；建立 profile 基线 |
| `02_cutlass_gemm/` | CUTLASS FP16 Tensor Core GEMM；PTX `mma.sync` 验证；大矩阵吞吐 benchmark |
| `03_fastgelu_epilogue/` | Fast-GELU inline PTX；CUTLASS Epilogue output op；SFU 指令验证 |
| `04_compiler_integration/` | MLIR pass、external call emission、runtime C ABI、端到端 benchmark |
| `scripts/` | 一键构建和功能验证 |
| `docs/` | spec、环境说明和本学习手册 |

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

阶段二把手写 FP32 SGEMM 换成 CUTLASS device-level GEMM，目标是触发 Volta Tensor Core HMMA 指令。

重点文件：

- `02_cutlass_gemm/cutlass_gemm.cu`
- `02_cutlass_gemm/correctness_test.cu`
- `02_cutlass_gemm/verify_mma_ptx.sh`
- `02_cutlass_gemm/bench_throughput_large.cu`

核心 CUTLASS 配置：

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

这里有一个容易忽略的正确性细节：GPU 侧 Tensor Core 使用 FP16 输入，因此 CPU reference 也要先把 A/B 量化到 FP16 再比较：

```cpp
inline float RoundToFp16(float value) {
  return __half2float(__float2half(value));
}
```

吞吐对比要用大矩阵。256³ 的 correctness size 只产生 4 个 CUTLASS threadblock，远远不够填满 V100 的 80 个 SM；4096³ 才能体现 Tensor Core 的优势：

```bash
./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096
# stage1: 3.05 TFLOPS
# stage2: 36.27 TFLOPS
# speedup: 11.9x
```

学习重点：

- `ElementA/B = cutlass::half_t` 与 `ElementAccumulator = float` 的含义。
- `ThreadblockShape`、`WarpShape`、`InstructionShape` 如何对应底层 `mma.sync.aligned.m8n8k4`。
- 为什么 correctness 用小矩阵，performance 要用大矩阵。
- `ncu` 的 `% of peak` 必须看清楚分母：FP32 peak 和 Tensor Core FP16 peak 不是同一个峰值。

## 5. 阶段三：Fast-GELU Epilogue

阶段三把激活函数融合进 CUTLASS Epilogue。目标不是再启动一个 GELU kernel，而是在 GEMM accumulator 输出时直接完成：

```text
D = FastGELU(alpha * accumulator + beta * source)
```

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

学习重点：

- 为什么 GELU 放在 epilogue 里可以避免中间张量显存读写。
- 为什么要用 SFU 近似指令，而不是普通 `expf` 展开。
- 数值验收为什么看绝对误差和 SFU 额外误差，而不是只看相对误差。

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

- 为什么 pass 要在 bufferization 后匹配 memref-based `linalg.matmul`。
- 为什么 matmul 输出必须只有一个 activation consumer。
- 为什么 external call 前必须做 rank/type/layout 校验。
- 为什么当前 benchmark fused 侧直接调用 C++ 可执行文件，而 fusion 行为由单独的 pass test 覆盖。

## 7. 验证与排障清单

| 目标 | 命令 | 看什么 |
| --- | --- | --- |
| 全量功能验证 | `./scripts/build_all.sh` | 构建成功、静态脚本通过、`ctest` 4/4 |
| 只跑 CUDA correctness | `cd build && ctest -R stage[1-3] --output-on-failure` | 数值误差和 CUDA runtime 是否正常 |
| 只跑 MLIR pass test | `cd build && ctest -R stage4 --output-on-failure` | IR 中 matmul/generic 是否被替换 |
| 验证 Tensor Core 指令 | `./02_cutlass_gemm/verify_mma_ptx.sh` | `mma.sync.aligned.m8n8k4`、无 `ld.local/st.local` |
| 验证 SFU 指令 | `./03_fastgelu_epilogue/verify_sfu_sass.sh` | `ex2.approx`、`rcp.approx`、`MUFU.EX2`、`MUFU.RCP` |
| 阶段一 profile | `./01_baseline_cuda/profile.sh build/01_baseline_cuda/stage1_correctness_test` | `nsys` timeline、`ncu` SM/memory/occupancy |
| 阶段二吞吐 | `./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096` | TFLOPS 和 speedup |
| 阶段四延迟 | `./04_compiler_integration/benchmark_e2e.sh` | fused/unfused 总耗时和 latency change |

常见问题：

- CUDA 13.x 不再适合 Volta sm_70 离线编译；V100 建议使用 CUDA 12.9。
- `ncu` 报 stub `libcuda.so` 不一定真是 stub 路径问题，也可能是 Nsight Compute 与驱动版本不兼容。
- `ERR_NVGPUCTRPERM` 表示性能计数器权限不足；本机需要 `sudo ncu`。
- 小矩阵 correctness 结果不能代表 Tensor Core 吞吐；做性能对比要用大矩阵。

## 8. 推荐学习路线与时间安排

### 7 天路线

| 时间 | 任务 | 产出 |
| --- | --- | --- |
| 第 1 天 | 阅读 `README.md`、`docs/spec.md`、`docs/environment.md`，完成构建 | 能解释四个阶段和当前环境约束 |
| 第 2 天 | 学阶段一，读 `sgemm_naive.cu` / `sgemm_tiled_smem.cu` / `matrix_ref.hpp`，跑 stage1 correctness | 能解释 naive 与 tiled 的访存差异 |
| 第 3 天 | 跑 `profile.sh`，看 `nsys`/`ncu` 指标 | 能读懂 SM throughput、memory throughput、occupancy |
| 第 4 天 | 学阶段二，读 CUTLASS 模板配置，跑 `verify_mma_ptx.sh` 和大矩阵 benchmark | 能解释 Tensor Core 指令和小/大矩阵差异 |
| 第 5 天 | 学阶段三，读 `FastGeluPTX` 和 epilogue op，跑 SFU 验证 | 能解释 inline PTX 和误差验收 |
| 第 6 天 | 学阶段四，读 pattern matching 和 external call emission，跑 stage4 test | 能解释 MLIR IR 如何变成外部 fused call |
| 第 7 天 | 跑端到端 benchmark，整理自己的实验笔记 | 能复述完整 pipeline 和每个验收数据的含义 |

### 进阶练习

- 把阶段二 benchmark 的矩阵从 2048³、4096³、8192³ 分别跑一遍，观察 TFLOPS 和 `ncu` 指标变化。
- 改 `ThreadblockShape` 或 `WarpShape`，重新跑 `verify_mma_ptx.sh` 和大矩阵 benchmark，比较吞吐变化。
- 在阶段三替换 Fast-GELU 常数或公式，观察 correctness 和 SFU 指令验证是否仍通过。
- 给阶段四增加一个负例 MLIR 测试：非 identity layout 或 rank 不是 2 时，确认 pass 报错而不是生成 external call。

## 9. 阅读顺序建议

先读 `common/matrix_ref.hpp`，理解 correctness 的地基；再按 `01 -> 02 -> 03 -> 04` 目录推进。不要先从 MLIR pass 开始，否则很容易看懂了 IR rewrite，却不清楚它最终调用的 CUDA kernel 为什么可靠、为什么快。

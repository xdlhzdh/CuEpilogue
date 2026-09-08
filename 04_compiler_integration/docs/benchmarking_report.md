# 阶段四验收与基准报告 — MLIR 融合改写为外部 CUTLASS 调用

## 范围

- **编译路径**：`linalg.matmul` + Fast-GELU 的 `linalg.generic` → `--one-shot-bufferize` + `--fuse-gemm-gelu` → `func.call @cutlass_fused_gemm_gelu`（FP32 GEMM + GELU）。
- **运行时 kernel**：复用阶段三 CUTLASS Fast-GELU 融合 kernel（`libcutlass_fused_gemm_gelu_runtime.so`，与 `FusedGemmGeluFp16` 同一实现）。
- **实测硬件**：Tesla V100（sm_70）。
- **本阶段不做的事**：不会把融合后的 IR 再经 `mlir-runner` 去调 `.so`（bare-pointer 约定不支持动态形状 `memref<?x?xf32>`）。Pass 正确性只验 IR；延迟侧用手写 host 直接调用 **Pass 本应发出的同一 C/CUDA 符号**。

## CMake：配置 vs 编译

这是 CMake 流程里 **两个不同步骤**：

| 命令 | 作用 |
| --- | --- |
| `cmake -B build -DCU_EPILOGUE_ENABLE_CUDA=ON` | **配置**：在 `build/` 生成构建系统并写入选项（开启 CUDA 相关阶段及阶段四依赖 CUDA 的目标）。**本身不编译源码**。 |
| `cmake --build build -j$(nproc) --target fused-opt bench_fused_e2e` | **编译**：只构建指定目标。`fused-opt` 为 MLIR 工具；`bench_fused_e2e` 为调用 CUTLASS kernel 的计时可执行文件。 |

仓库根目录典型流程：

```bash
cmake -B build -DCU_EPILOGUE_ENABLE_CUDA=ON
cmake --build build -j$(nproc) --target fused-opt bench_fused_e2e
# 如需编齐其它目标：
cmake --build build -j$(nproc)
```

未开 `CU_EPILOGUE_ENABLE_CUDA=ON` 时，`fused-opt` 仍可只靠 MLIR 编过，但 `bench_fused_e2e` 与 runtime `.so` 会跳过。

## 正确性（仅 IR）

只测融合 Pass：matmul + GELU 是否收成一次外部调用。

```bash
cd build && ctest -R stage4_fuse_gemm_gelu_test --output-on-failure
# 或：
./04_compiler_integration/test/run_tests.sh ./build/04_compiler_integration/fused-opt
```

期望：出现 `call @cutlass_fused_gemm_gelu`，融合用例中不再残留对应的 `linalg.matmul` / GELU `linalg.generic`。

## 端到端延迟

### 三档基线（推荐理解方式）

| 档 | 含义 | 设备 / 产物 | 状态 |
| --- | --- | --- | --- |
| **A. CPU 标量 JIT** | spec「基础指令展开」：**不加** `--fuse-gemm-gelu`，`--convert-linalg-to-loops` | `mlir-runner` **CPU** JIT | 已测 |
| **B. GPU AOT 未融合** | 同上 IR，走 Linalg → `gpu.launch` → **内嵌 PTX 的 host ELF** | AOT 可执行文件（非 JIT、非 CUTLASS） | 脚本已提供；本机需 **NVPTX** MLIR（见下） |
| **C. CUTLASS 融合** | `--fuse-gemm-gelu` 应对应的外部 kernel | `bench_fused_e2e` 手写 GPU 调用 | 已测 |

A 与 C 跨执行域（CPU vs GPU），适合说明「融合 + 专用 kernel 相对 CPU 展开」的数量级差距。  
**B 与 C 同在 GPU**，更适合说明「编译器融合改写相对通用 MLIR GPU lowering」的收益。详见 [aot_gpu_baseline.md](aot_gpu_baseline.md)。

规模均为 **128³**、**20** 次迭代；输入为 memref f32（A/B 近似全 1），**不是** StableHLO。

### 已测数据（A vs C，V100）

| 档 | 总耗时（20 次） |
| --- | ---: |
| C. 融合 GPU（CUTLASS） | **约 0.68 ms** |
| A. 未融合 CPU 标量（mlir-runner） | **约 148.3 ms** |
| A→C 延迟下降 | **约 99.5%**（spec ≥ 30%） |

### B 档：GPU AOT 未融合（构建与运行）

前置检查（本机 LLVM 23 仅 `X86;RISCV`，**无 NVPTX**，B 档在此环境会失败）：

```bash
./04_compiler_integration/scripts/check_aot_gpu_prereqs.sh
./04_compiler_integration/scripts/build_aot_gpu_baseline.sh
# 成功则：
./04_compiler_integration/docs/work_dirs/aot_gpu_baseline/bench_aot_gpu_unfused
```

`benchmark_e2e.sh` 在检测到上述二进制时会额外打印 B 档耗时。

### 一键脚本（推荐）

```bash
# 需已编好 fused-opt、bench_fused_e2e；PATH 中有 mlir-runner 及 runner .so
./04_compiler_integration/benchmark_e2e.sh
# 或指定 build 目录：
./04_compiler_integration/benchmark_e2e.sh build
```

### 等价手工命令

```bash
# --- 快侧 ---
./build/04_compiler_integration/bench_fused_e2e
# 打印 warmup 后 20 次 GPU 调用的总毫秒数

# --- 慢侧 ---
MLIR_BIN_DIR="$(dirname "$(command -v mlir-runner)")"
./build/04_compiler_integration/fused-opt \
  04_compiler_integration/test/bench_unfused.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-math-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts > /tmp/stage4_unfused.mlir

mlir-runner /tmp/stage4_unfused.mlir -e main --entry-point-result=void \
  --shared-libs="$MLIR_BIN_DIR/../lib/libmlir_runner_utils.so,$MLIR_BIN_DIR/../lib/libmlir_c_runner_utils.so"
# 打印秒数；benchmark_e2e.sh 会换算成 ms 并算下降比例
```

### 如何理解这些对比

- **不是**「融合后的 MLIR JIT」对「未融合的 MLIR JIT」。
- **A vs C**：融合 CUTLASS kernel 相对 CPU 标量展开快多少（验收 spec ≥ 30%）。
- **B vs C**：同一未融合计算图若由 MLIR **通用 GPU lowering AOT** 执行，相对融合 CUTLASS 慢多少（更公平的编译器视角）。
- 细节不完全相同（A/B 每轮改写 `A[0,0]`；C 走 FP16 Tensor Core；B 为朴素 1×1 block GPU kernel）。当作 **延迟验收**，不作逐元素比特级对照。

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `lib/FuseGemmGeluPattern.cpp` / `EmitExternalCall.cpp` | 匹配并生成 `func.call` |
| `test/fuse_pattern.mlir` | IR 正确性输入 |
| `test/bench_unfused.mlir` | 延迟基线输入 |
| `test/bench_fused_e2e.cu` | 快侧 host 驱动 |
| `benchmark_e2e.sh` | 串联 A/C（及可选 B）并计算下降比例 |
| `docs/aot_gpu_baseline.md` | GPU AOT 未融合基线设计与流水线 |
| `scripts/build_aot_gpu_baseline.sh` | 构建 B 档 AOT 可执行文件 |
| `runtime/cutlass_fused_gemm_gelu_c_api.cu` | 封装阶段三 kernel 的 `.so` C ABI |

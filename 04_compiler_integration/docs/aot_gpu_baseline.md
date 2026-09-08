# 阶段四 GPU AOT 基线 — Linalg 展开 → 嵌入 PTX 的 Host ELF

## 动机

阶段四当前端到端对比是 **两档**：

| 档 | 路径 | 设备 |
| --- | --- | --- |
| 快侧 | `--fuse-gemm-gelu` → 外部 `func.call` → CUTLASS kernel | GPU（手写 host 驱动） |
| 慢侧 | 未融合 `linalg` → `--convert-linalg-to-loops` → 标量循环 | **CPU**（`mlir-runner` JIT） |

慢侧在 CPU 上跑标量循环，与快侧 GPU Tensor Core 不在同一执行域，只能说明「融合 + 手写 kernel 远快于 CPU 标量」，**不能**回答：

> 若编译器老老实实把未融合的 `linalg.matmul` + GELU **降到 GPU 并 AOT 成 ELF**（host 代码 + 内嵌 PTX/CUBIN），融合改写还能快多少？

因此需要第三档基线：**Linalg → GPU kernel outlining → NVVM/PTX → 嵌入 host LLVM IR → `clang` 链接为 ELF（AOT）**。

这与阶段四「融合路径走外部 CUTLASS `.so`」正交：第三档衡量的是 **MLIR 通用 GPU lowering 的未融合代价**，而非 CUTLASS 手写 kernel。

## 目标流水线

输入与慢侧相同：`test/bench_unfused.mlir`（128³、`gemm_gelu_unfused`、20 次迭代、`@rtclock` 计时）。

```text
bench_unfused.mlir
  │
  ├─ --convert-linalg-to-parallel-loops
  ├─ --gpu-map-parallel-loops
  ├─ --convert-parallel-loops-to-gpu      ← 不可省略；否则不会出现 gpu.launch
  ├─ --gpu-kernel-outlining
  │
  ├─ --gpu-lower-to-nvvm-pipeline=        ← 需要 LLVM 注册 NVPTX target
  │     "cubin-chip=sm_70 cubin-triple=nvptx64-nvidia-cuda"
  │   （device 侧 PTX/CUBIN 以 blob 属性挂在 module 上）
  │
  ├─ host 侧：--convert-gpu-to-cf / --convert-scf-to-cf / … → LLVM dialect
  ├─ mlir-translate --mlir-to-llvmir → host.ll
  └─ clang++ host.ll -o bench_aot_gpu_unfused
         -lcudart -lcuda
         -lmlir_runner_utils -lmlir_c_runner_utils   # @rtclock / @printF64
```

产物为 **单个 host ELF**：`main` 在 CPU 上分配 memref、发起 `gpu.launch_func`；device 代码以 **内嵌二进制** 形式随 ELF 携带，运行时经 CUDA Driver API 加载，**不是** `mlir-runner` JIT，也**不是**阶段三 CUTLASS 手写 kernel。

### 与阶段四其它路径的区别

| | CPU JIT 慢侧 | GPU AOT 慢侧（本档） | CUTLASS 融合快侧 |
| --- | --- | --- | --- |
| 融合 Pass | 无 | 无 | `--fuse-gemm-gelu` |
| 代码生成 | MLIR → LLVM → **CPU** ExecutionEngine | MLIR → GPU kernels + **host AOT ELF** | 外部 `func.call` + 预编 `.so` |
| 典型 kernel | 标量 `scf.for` | 两个朴素 `gpu.launch`（matmul + GELU） | CUTLASS Tensor Core + Fast-GELU epilogue |
| 公平性 | 验收「相对 CPU 展开」 | 验收「相对编译器生成的未融合 GPU」 | 项目目标性能 |

## 本机探测结果（MLIR 23.0.0git，`/usr/local`）

中间段（至 `gpu.launch_func`）**已成功**：

```bash
mlir-opt test/bench_unfused.mlir \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  -o docs/work_dirs/aot_gpu_baseline/gpu_launch.mlir
```

`gemm_gelu_unfused` 被拆成 **两个** device kernel（matmul、GELU），`main` 内 `linalg.fill` 亦各自 outline 为小型 launch（共十余个 kernel 调用，属通用 lowering 噪声）。

**阻塞点**：`--gpu-lower-to-nvvm-pipeline` 失败：

```text
Failed to lookup target for triple 'nvptx64-nvidia-cuda'
Unable to find target for this triple (no targets are registered)
```

根因：`llvm-config --targets-built` 仅为 `RISCV X86`，**未编入 NVPTX**。`mlir-opt` / `mlir-translate` 无法为 device triple 生成目标代码，整条 AOT 链在此步中断。

修复方向（任选其一）：

1. 自研 LLVM/MLIR，CMake 打开 `LLVM_TARGETS_TO_BUILD="X86;NVPTX"`，并确保 MLIR 链到带 NVPTX 的 LLVM。
2. 使用官方或第三方已启用 GPU target 的 MLIR 预编译包。
3. 手工分段：自行 `ptxas` / `nvcc -cubin` 后将 blob 喂给 `gpu-module-to-binary`（维护成本高，仅作兜底）。

环境要求已写入 [docs/environment.md](../../docs/environment.md)「MLIR GPU AOT 基线」小节。

## 使用方式

### 1. 检查前置条件

```bash
./04_compiler_integration/scripts/check_aot_gpu_prereqs.sh
```

### 2. 构建 AOT 可执行文件

```bash
./04_compiler_integration/scripts/build_aot_gpu_baseline.sh
# 产物默认：04_compiler_integration/docs/work_dirs/aot_gpu_baseline/bench_aot_gpu_unfused
```

### 3. 跑延迟（构建成功后）

```bash
./04_compiler_integration/docs/work_dirs/aot_gpu_baseline/bench_aot_gpu_unfused
# 或纳入三档对比：
./04_compiler_integration/benchmark_e2e.sh build   # 若检测到 AOT 二进制则打印第三列
```

## 预期性能特征（定性）

- 通用 `gpu-map-parallel-loops` 在 128³ 上常产出 **1×1 thread block** 的朴素 kernel（见 `gpu_launch.mlir` 中 `known_block_size = [1,1,1]`），GPU 利用率极低。
- 即便如此，相对 CPU 标量 JIT 仍可能快一个数量级以上；相对 CUTLASS 融合 kernel 仍应慢很多。
- 三档对比的意义：**融合改写节省的是「未融合 GPU 多 kernel + 中间写回」与「单 kernel CUTLASS」之间的差距**，而不是与 CPU 标量的差距。

## 阶段五延伸（规划）

阶段五未融合基线若走同一 GPU AOT 路径，需先把 QDQ tensor 链 bufferize 为 memref，再对 DQ / matmul / bias / GELU / Q **分别**做 linalg→GPU lowering（多 kernel、多中间缓冲），与单 `func.call @cutlass_qgemm_bias_gelu` 对比。INT8 / affine 零点语义在通用 `linalg.generic` GPU lowering 上更易踩坑，**建议阶段四 FP32 路径先跑通 AOT 脚本与三档表**，再在阶段五补 `bench_unfused_qdq` 变体或文档级说明。详见 [05_qdq_fusion/docs/aot_gpu_baseline.md](../../05_qdq_fusion/docs/aot_gpu_baseline.md)。

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `test/bench_unfused.mlir` | 与 CPU JIT 慢侧相同的 IR 输入 |
| `scripts/check_aot_gpu_prereqs.sh` | NVPTX / CUDA / MLIR 工具检查 |
| `scripts/build_aot_gpu_baseline.sh` | 完整 AOT 构建 |
| `docs/work_dirs/aot_gpu_baseline/` | 中间 MLIR / 构建日志（本地生成） |

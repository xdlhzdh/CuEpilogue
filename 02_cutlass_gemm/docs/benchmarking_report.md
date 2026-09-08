# 阶段二验收与基准报告 — CUTLASS Tensor Core GEMM

## 范围

- **实现**：CUTLASS FP16 输入 / FP32 累加的 Tensor Core GEMM（`cutlass_gemm.cu`），配置 Threadblock / Warp / Instruction Shape，触发 `mma.sync`。
- **对比基线**：阶段一手写 FP32 tiled SGEMM（同规模吞吐）。
- **硬件（下列实测）**：Tesla V100（sm_70）。
- **注意**：`ctest` 里常用 **256³** 做正确性（CPU 参考可算完）；该规模对 CUTLASS **喂不饱** GPU，SM 利用率会看起来很低——**吞吐要以大矩阵 bench 为准**。

## CMake：配置 vs 编译

| 命令 | 作用 |
| --- | --- |
| `cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70` | **配置**。 |
| `cmake --build build -j$(nproc) --target stage2_correctness_test bench_throughput_large` | **编译**正确性与大矩阵吞吐可执行文件。 |

```bash
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70
cmake --build build -j$(nproc) --target stage2_correctness_test bench_throughput_large
```

## 正确性

```bash
cd build && ctest -R stage2_correctness_test --output-on-failure
# 或：
./build/02_cutlass_gemm/stage2_correctness_test
```

与 CPU / 阶段一参考路径比对数值（允许 FP16 累加相关误差）。

## PTX 静态检查（mma）

确认生成 Tensor Core 指令、无异常 local spill：

```bash
./02_cutlass_gemm/verify_mma_ptx.sh
```

期望 PTX 含类似 `mma.sync.aligned.m8n8k4`；`ld.local` / `st.local` 计数为 0（当前配置无寄存器溢出到 local）。

## 大矩阵吞吐（已测）

正确性用的 256³ **不能**代表真实吞吐。请跑：

```bash
./build/02_cutlass_gemm/bench_throughput_large 4096 4096 4096
```

4096³ 代表性结果（V100）：

| 路径 | 约耗时 / 次 | 吞吐 |
| --- | ---: | ---: |
| 阶段一 `SgemmTiledSmem` | ~45.0 ms | **约 3.05 TFLOPS** |
| 阶段二 CUTLASS FP16 TC | ~3.8 ms | **约 36.27 TFLOPS** |
| 加速比 | — | **约 11.9×** |

读 `ncu` 时不要直接比两边的「% of peak」分母：阶段一按 FP32 峰值，阶段二按 Tensor Core FP16 峰值；**比耗时和 TFLOPS** 即可。

## 为何 256³ 上 CUTLASS 看起来更「慢 / 闲」

- Threadblock 一次处理约 128×128 → 256×256 输出只需很少 block
- V100 有大量 SM，小矩阵无法占满
- 故 correctness 场景下 CUTLASS SM 利用率可以远低于阶段一 tiled——这是 **规模问题**，不是库更慢

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `cutlass_gemm.cu` | CUTLASS GEMM 封装 |
| `correctness_test.cu` | 小规模数值正确性 |
| `bench_throughput_large.cu` | 大矩阵吞吐计时 |
| `verify_mma_ptx.sh` | PTX `mma.sync` / spill 检查 |

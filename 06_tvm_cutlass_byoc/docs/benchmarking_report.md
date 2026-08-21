# Stage 6 Benchmarking Report — TVM Relax + CUTLASS QGEMM BYOC

## Scope

- **Kernel**: reuse stage-5 `libcutlass_qgemm_bias_gelu_runtime.so` (SIMT INT8 + SFU FastGELU, affine zp).
- **Compiler path**: TVM Relax hand-built QDQ graph → pattern match → packed `cu_epilogue.qgemm_bias_gelu` → **one `.so` API call** per composite op.
- **Hardware**: Tesla V100 (sm_70). INT8 Tensor Core is not available; SIMT path is the acceptance path.
- **TVM**: apache-tvm 0.26.0 (project venv via `06_tvm_cutlass_byoc/scripts/setup_tvm.sh`).

> One `.so` call ≠ one CUDA kernel. Stage 5’s ABI internally launches **RowSum + Gemm + ColSum** for affine zero-point correction; see Kernel count below.

## Correctness

三项都已通过。各自在验证什么：

**M1 — 融合 kernel 算得对不对**（`tests/test_m1_so_correctness.py`）

直接调用阶段五的 `.so`，把 GPU 输出和 NumPy 手算结果逐元素比较。INT8 差最多允许 1。通过。

**M2 — TVM 能不能认出该融合的图、且不会误融合**（`tests/test_m2_pattern.py`）

给 matcher 两张图：

1. 完整的「反量化 → 矩阵乘 → 加偏置 → FastGELU → 再量化」。matcher 必须认出来，并打上 `cutlass.qgemm_bias_gelu` 标记。
2. 只有「矩阵乘 → FastGELU」、**没有加偏置**。这条链不是我们要融合的算子，matcher 必须放过，不能打标记。

两张图都按预期。通过。

**M3 — 从图到 GPU 整条路径通不通**（`tests/test_m3_e2e.py`）

TVM 认图、改成外部调用、再跑 `.so`。输出仍与 NumPy 差 ≤ 1。同时计数器显示这次测试只进了 **一次** `cutlass_qgemm_bias_gelu` C 接口（没有重复调用）。

注意：这里的「一次」是 **C 函数调用次数**，不是 Nsight 里看到的 CUDA kernel 个数。那个 C 函数内部仍会起 RowSum、Gemm、ColSum 三个 kernel，见下一节。

```bash
./06_tvm_cutlass_byoc/scripts/setup_tvm.sh
export CU_EPILOGUE_BUILD_DIR=$PWD/build
source 06_tvm_cutlass_byoc/.venv/bin/activate
export PYTHONPATH=06_tvm_cutlass_byoc/python:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/05_qdq_fusion:$LD_LIBRARY_PATH
pytest 06_tvm_cutlass_byoc/tests -q
# or: ctest --test-dir build -R stage6 --output-on-failure
```

## Kernel count (nsys) — measured

```bash
./06_tvm_cutlass_byoc/scripts/profile_nsys.sh
```

Workload: shape **256³**, **3** fused `.so` launches → **9** GPU compute kernels (3 per call).

| Time % | Instances | Kernel |
| --- | --- | --- |
| ~50% | 3 | `RowSumKernel` (affine zp prep) |
| ~42% | 3 | CUTLASS `Gemm<...>` (INT8 MMA + Bias + FastGELU + Q epilogue) |
| ~8% | 3 | `ColSumKernel` (affine zp prep) |

Per composite call this is the same **3-kernel** structure as stage 5. There are **no** separate DQ / Bias / FastGELU / Q compute kernels; those live in the Gemm epilogue. Intermediate f32 tensors are not written to DRAM. Host↔device memcpy around the call is glue, not a fusion failure.

Artifact: `06_tvm_cutlass_byoc/docs/profile_artifacts/stage6_nsys.nsys-rep` (from repo root).

## DRAM / bandwidth (ncu) — measured

```bash
# from repo root; needs sudo for GPU counters (ERR_NVGPUCTRPERM otherwise)
./06_tvm_cutlass_byoc/scripts/profile_ncu.sh
```

Workload: **one** fused `.so` call, shape **256³**, ncu **2025.2.1**.
Artifacts (from repo root):
`06_tvm_cutlass_byoc/docs/profile_artifacts/stage6_ncu.ncu-rep`,
`06_tvm_cutlass_byoc/docs/profile_artifacts/stage6_ncu_summary.csv`.

| Kernel | DRAM read | DRAM write | Sectors read | Sectors write | DRAM % peak | SM % peak | Duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RowSumKernel` | 69.0 KB | 0 | 2157 | 0 | 0.15% | 0.04% | 53.0 µs |
| `ColSumKernel` | 70.0 KB | 0 | 2189 | 0 | 0.53% | 0.18% | 14.6 µs |
| CUTLASS `Gemm` (INT8 + Bias + FastGELU + Q) | 169.2 KB | 0 | 5289 | 0 | 0.33% | 1.45% | 56.4 µs |
| **Total (this call)** | **~308 KB** | **0** | **9635** | **0** | — | — | **~124 µs** |

How to read this:

- There is still **no** separate DQ / GELU / Q kernel in the DRAM timeline — only zp prep + one Gemm that finishes quantization in-register.
- Total DRAM **read** ≈ 308 KB for 256³. A naive unfused GPU path that materializes f32 DQ(A), DQ(B), matmul, and gelu would touch on the order of several MB of intermediate f32 (256²×4 ≈ 256 KB per matrix, several of them). The fused path avoids those intermediate writes.
- Measured DRAM **write** is 0 on this small shape: the INT8 output (~64 KB) and tiny sum buffers fit in V100 L2 and are not flushed as DRAM write sectors in this capture. That is a profiling artifact of size, not “kernel wrote nothing”. For larger shapes (e.g. 1×4096×4096), re-run `profile_ncu.sh` after changing the launch shape in the script if you need write traffic numbers.

**Baseline note**: end-to-end speedup vs NumPy still uses the stage-4-style CPU unfused baseline (see next section). The ncu table above is the GPU-side DRAM view of the fused call only.

## End-to-end latency (measured)

Host: Tesla V100, stage-5 SIMT `.so`, warmup 3 / repeats 10.

```bash
python 06_tvm_cutlass_byoc/scripts/bench_e2e.py --m 256 --n 256 --k 256
python 06_tvm_cutlass_byoc/scripts/bench_e2e.py --m 1 --n 4096 --k 4096
```

| Shape | Fused (ms) | Unfused NumPy CPU (ms) | Speedup |
| --- | --- | --- | --- |
| 256³ | 0.528 | 2.413 | **4.57×** |
| 1×4096×4096 | 7.079 | 79.362 | **11.21×** |

Target **≥ 1.5×** vs the documented unfused baseline: **met**.

## Tile / alignment notes

Stage 6 does **not** change stage-5 `ThreadblockShape` / `WarpShape` by default (preserves stage-5 ctest). The 256³ ncu capture above is SM-light (~1.5% peak on Gemm); for occupancy/tile experiments use larger shapes and re-profile.

## Integration note

“CUTLASS codegen” here means **binding the prebuilt CUTLASS `.so`** from CMake, not re-invoking nvcc from TVM. Each composite op becomes **one** packed / `.so` launch; that launch’s GPU timeline is the stage-5 **RowSum + Gemm + ColSum** sequence above.

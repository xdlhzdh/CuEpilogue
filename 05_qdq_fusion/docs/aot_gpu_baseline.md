# 阶段五 GPU AOT 未融合基线（规划）

## 为何需要

阶段五快侧与阶段四类似：融合后 `func.call @cutlass_qgemm_bias_gelu`，运行时走预编 INT8 CUTLASS `.so`，**不是** MLIR GPU codegen。

若只对比「融合 `.so`」与「CPU affine 参考」（`stage5_qgemm_correctness_test`），缺少与阶段四对称的 **编译器生成未融合 GPU** 档位，无法单独量化「QDQ 整图融合」相对「多 kernel GPU AOT」的收益。

## 目标形态（与阶段四同构）

```text
qdq 未融合 memref IR（DQ + matmul + bias + GELU + Q 仍为独立 linalg/generic）
  → linalg/parallel → gpu-map → convert-parallel-loops-to-gpu
  → gpu-kernel-outlining（预期 5+ 个 device kernel + 中间写回）
  → gpu-lower-to-nvvm-pipeline（sm_70）
  → host AOT ELF + 内嵌 PTX
```

计时方式可与 `bench_unfused.mlir` 一样用 `@rtclock`，或单独 C++ driver 分配 device buffer。

## 当前状态

| 项 | 状态 |
| --- | --- |
| 融合路径 IR + `.so` | 已验收（见 [benchmarking_report.md](benchmarking_report.md)） |
| 端到端延迟（融合 vs 未融合 GPU） | 在阶段六 `bench_e2e.py` / nsys 中覆盖 |
| **未融合 GPU AOT 脚本** | **未实现**；依赖阶段四 `build_aot_gpu_baseline.sh` 与 NVPTX 工具链先跑通 |
| INT8 affine / clamp 在通用 GPU lowering | 需单独验证数值与 bufferization；可能比 FP32 更易失败 |

## 建议实施顺序

1. 在本机或 CI 镜像中提供 **带 NVPTX 的 MLIR**（见 [environment.md](../../docs/environment.md)）。
2. 复用阶段四 AOT 脚本模式，新增 `test/bench_unfused_qdq.mlir`（memref 上显式 DQ/matmul/bias/gelu/Q，**不**走 `--fuse-qdq-qgemm-bias-gelu`）。
3. 将三档表写入阶段五报告：CPU affine 参考（已有）| GPU AOT 未融合 | CUTLASS 融合 `.so`。
4. 与阶段六 nsys 结论交叉验证：多 kernel + 中间 INT8/f32 缓冲是否与 AOT 未融合路径一致。

## 相关

- 阶段四 AOT 设计：[04_compiler_integration/docs/aot_gpu_baseline.md](../../04_compiler_integration/docs/aot_gpu_baseline.md)
- 检查脚本：`04_compiler_integration/scripts/check_aot_gpu_prereqs.sh`

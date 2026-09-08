# 阶段三验收与基准报告 — Fast-GELU Epilogue 融合

## 范围

- **目标**：把 Fast-GELU 融进 CUTLASS Epilogue，在寄存器路径完成激活，避免「先写回 GEMM 结果再起独立 GELU kernel」。
- **默认路径（functor）**：Sm70 `device::Gemm` + Inline PTX（`ex2.approx` / `rcp.approx`）→ `FusedGemmGeluFp16`；阶段四默认调用此路径。
- **旁路（Sm90 visitor）**：`CollectiveBuilder` + `LinCombEltAct`，单独编 `sm_90a`，**不替换**默认 functor `.so` / 库。
- **本阶段不测什么**：不做「再跑一遍大矩阵吞吐」；融合相对未融合展开的 **端到端延迟收益在阶段四** 用 `benchmark_e2e.sh` 验收。

## CMake：配置 vs 编译

| 命令 | 作用 |
| --- | --- |
| `cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70` | **配置** functor 路径目标架构。 |
| `cmake --build build -j$(nproc) --target stage3_correctness_test fused_gemm_gelu` | **编译** functor 正确性与库。 |
| `-DCU_EPILOGUE_ENABLE_SM90_VISITOR=ON` | 额外编 visitor / `stage3_visitor_*`（`sm_90a`）。 |

```bash
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70
cmake --build build -j$(nproc) --target stage3_correctness_test
# 可选 visitor：
cmake -B build -DCU_EPILOGUE_ENABLE_SM90_VISITOR=ON
cmake --build build -j$(nproc) --target stage3_visitor_correctness_test
```

## 正确性（functor）

相对 CPU Fast-GELU 参考：

```bash
cd build && ctest -R stage3_correctness_test --output-on-failure
# 或：
./build/03_fastgelu_epilogue/stage3_correctness_test
```

输入为运行时生成的矩阵数据（与阶段一/二类似的 host→device 流程）。

## SFU 指令验证（functor）

确认 epilogue 走了近似 SFU，而非纯软件慢路径：

```bash
./03_fastgelu_epilogue/verify_sfu_sass.sh
```

期望：PTX 含 `ex2.approx` / `rcp.approx`（或对应 SASS `MUFU.*`）。

## Sm90 visitor（可选）

```bash
./03_fastgelu_epilogue/verify_sfu_visitor_sm90.sh
cd build && ctest -R stage3_visitor --output-on-failure
```

| 环境 | 预期 |
| --- | --- |
| 已编 `sm_90a` | PTX/SASS 可含 SFU 指令 |
| V100 上跑 visitor 设备正确性 | **SKIP**（算力不够 Hopper） |

## 与阶段四延迟数字的关系

阶段三证明的是：**融合进 epilogue + 走 SFU + 数值可接受**。  
「融合 GPU kernel vs 未融合 CPU 标量展开」的 0.68 ms / 148.3 ms / **99.5%** 见：

- [04_compiler_integration/docs/benchmarking_report.md](../../04_compiler_integration/docs/benchmarking_report.md)
- `./04_compiler_integration/benchmark_e2e.sh`

快侧调用的正是本阶段 functor 路径的同一套 CUTLASS Fast-GELU kernel。

## 相关文件

| 文件 | 作用 |
| --- | --- |
| `fused_gemm_gelu.cu` / `*.cuh` | Functor 融合 GEMM+GELU |
| `fused_gemm_gelu_visitor_sm90.cu` | Sm90 visitor 旁路 |
| `correctness_test.cu` | Functor 数值 |
| `verify_sfu_sass.sh` | Functor SFU 检查 |
| `verify_sfu_visitor_sm90.sh` | Visitor SFU 检查 |

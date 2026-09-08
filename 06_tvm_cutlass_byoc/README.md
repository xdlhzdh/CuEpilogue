# 阶段六 — TVM Relax + CUTLASS QGEMM BYOC

不重写 kernel。在 Relax 上认出 DQ→MatMul→Bias→FastGELU→Q，改写成 **一次** 对阶段五 `.so` 的 packed 调用。

```text
手写 Relax QDQ 图
  → pattern 匹配
  → call_dps_packed("cu_epilogue.qgemm_bias_gelu")
  → libcutlass_qgemm_bias_gelu_runtime.so
      内部 3 个 CUDA kernel：RowSum + Gemm + ColSum
```

测得的数字见 [docs/benchmarking_report.md](docs/benchmarking_report.md)。

---

## 环境准备

仓库根目录。先编阶段 1–5（要有 `.so`），再开 TVM：

```bash
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70 -DCU_EPILOGUE_ENABLE_TVM=ON
cmake --build build -j$(nproc)

./06_tvm_cutlass_byoc/scripts/setup_tvm.sh
source 06_tvm_cutlass_byoc/.venv/bin/activate
export CU_EPILOGUE_BUILD_DIR=$PWD/build
export PYTHONPATH=$PWD/06_tvm_cutlass_byoc/python:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/05_qdq_fusion:$LD_LIBRARY_PATH
```

---

## 两条执行路径

后面所有命令都落在其中一条，不要混着读数字。

| | CUDA VM（M3 / `build_and_run.py`） | Host `launch()`（M1、bench、nsys、ncu） |
| --- | --- | --- |
| 入口 | Relax VM，`tvm.runtime.tensor(..., cuda)` | numpy → `so_wrapper.launch()` |
| 进 `.so` 前 | 张量已在 GPU，`data_ptr()` 直传 | 内部 `cudaMalloc` + H2D / D2H |
| 经过 Relax？ | 是 | 否 |

无 GPU 时 CUDA VM 不可用，退到 host `launch()`（`build_and_run.py` 会打印 `using direct packed launch`）。

---

## 正确性（pytest M1–M3）

```bash
pytest 06_tvm_cutlass_byoc/tests -q
```

| 项 | 文件 | 测什么 | 通过标准 |
| --- | --- | --- | --- |
| **M1** | `tests/test_m1_so_correctness.py` | `.so` 数值（不经过 TVM） | vs NumPy，INT8 差 ≤ 1 |
| **M2** | `tests/test_m2_pattern.py` | Relax 图匹配；缺 bias 不误匹配 | 正例有 `Composite`；负例无 |
| **M3** | `tests/test_m3_e2e.py` | Relax → packed → `.so` 整条路径 | 差 ≤ 1；`launch_count() == 1` |

M1 = runtime 对不对；M2 = 匹配对不对（不需要 GPU）；M3 = TVM 接上 `.so` 后仍对。

手动改形状（与 M3 同一 `run_e2e()`）：

```bash
python 06_tvm_cutlass_byoc/python/build_and_run.py --m 64 --n 64 --k 64
# 期望：max_abs≤1  launches=1  OK
```

---

## CTest 怎么挂的

CMake 在 **`CU_EPILOGUE_ENABLE_TVM=ON`** 且 **`CU_EPILOGUE_BUILD_TESTS=ON`** 时，由 `06_tvm_cutlass_byoc/CMakeLists.txt` 注册一个名为 **`stage6_tvm_tests`** 的 test：

```cmake
add_test(
  NAME stage6_tvm_tests
  COMMAND bash scripts/run_pytest.sh ${CMAKE_BINARY_DIR} ${STAGE6_DIR}
)
set_tests_properties(stage6_tvm_tests PROPERTIES
  WORKING_DIRECTORY ${STAGE6_DIR}
  ENVIRONMENT "CU_EPILOGUE_BUILD_DIR=${CMAKE_BINARY_DIR};PYTHONPATH=..."
)
```

`run_pytest.sh` 做三件事：

1. 检查阶段五 `.so` 是否存在 → 缺则 **exit 0（SKIP）**
2. 检查 `import tvm; from tvm import relax` → 缺则 **exit 0（SKIP）**
3. 都有 → `exec python -m pytest tests/ -q`（即 **M1 + M2 + M3**）

CTest 跑法：

```bash
ctest --test-dir build -R stage6 --output-on-failure
```

> 测试注册在 **build 目录**。仓库根裸跑 `ctest` 会 `No tests were found!!!`。

CTest **不跑** bench / nsys / ncu。

---

## Bench（延迟对比）

```bash
python 06_tvm_cutlass_byoc/scripts/bench_e2e.py --m 1 --n 4096 --k 4096
```

两边都 **不经过 Relax / JIT**：

| 侧 | 实际跑什么 |
| --- | --- |
| **fused** | `so_wrapper.launch()` → GPU `.so`（含 H2D/D2H） |
| **unfused** | `qdq_reference.py` 纯 NumPy：DQ → matmul → FastGELU → Q，CPU |

加速比是「GPU 融合 kernel vs CPU 参考算法」，不是同设备未融合 GPU 对比。

---

## nsys — 看有哪些 kernel、耗时分布

### 怎么跑

一键脚本：

```bash
./06_tvm_cutlass_byoc/scripts/profile_nsys.sh
```

脚本内部实际执行的命令（等价手跑）：

```bash
# 采集：对 256³ 调 3 次 launch()，记录 CUDA/NVTX/OS runtime 事件
nsys profile \
  -o docs/profile_artifacts/stage6_nsys \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  python3 -c "
from so_wrapper import launch, reset_launch_count
from qdq_reference import make_test_inputs
A, B, bias, sa, za, sb, zb, sd, zd = make_test_inputs(256, 256, 256)
reset_launch_count()
for _ in range(3):
    launch(A, B, bias, sa, za, sb, zb, sd, zd)
"

# 查看：打印 GPU kernel 汇总表
nsys stats --report cuda_gpu_kern_sum docs/profile_artifacts/stage6_nsys.nsys-rep
```

### 产物

| 文件 | 说明 |
| --- | --- |
| `docs/profile_artifacts/stage6_nsys.nsys-rep` | 二进制报告，可用 Nsight Systems GUI 打开看时间线 |

### 怎么看

`nsys stats` 终端输出示例：

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Kernel Name
 --------  ---------------  ---------  --------  -----------
    60.2          xxxxx          3      xxxxx     cutlass::Kernel<Gemm_...>
    25.1          xxxxx          3      xxxxx     RowSum_kernel
    14.7          xxxxx          3      xxxxx     ColSum_kernel
```

确认要点：

- 计算 kernel **只有 RowSum / Gemm / ColSum**，没有独立 DQ / GELU / Q kernel → 融合成功
- 时间线上的 `cudaMemcpy` 来自 host `launch()` 的 H2D/D2H，不是融合失败
- 3 次 `launch()` → 9 个 GPU kernel（每次 3 个）
- 也可以 GUI 打开 `.nsys-rep`，看时间线上 kernel 之间有无不必要的空隙

---

## ncu — 看每个 kernel 的 DRAM / SM 利用率

### 怎么跑

一键脚本（采 GPU 硬件计数器通常要 sudo）：

```bash
./06_tvm_cutlass_byoc/scripts/profile_ncu.sh
```

脚本内部实际执行的命令（等价手跑）：

```bash
# 采集：对 256³ 一次 launch()，采指定硬件计数器
sudo ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --metrics launch__grid_size,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum,\
gpu__time_duration.sum \
  -o docs/profile_artifacts/stage6_ncu \
  --force-overwrite \
  python3 docs/profile_artifacts/_ncu_launch.py

# 导出 CSV（可直接看的文本表）
sudo ncu --import docs/profile_artifacts/stage6_ncu.ncu-rep --csv --page raw \
  > docs/profile_artifacts/stage6_ncu_summary.csv
```

> 建议 ncu **2025.2.1**；2026.x 在 driver 535 上可能不兼容，脚本会自动跳过。

### 产物

| 文件 | 说明 |
| --- | --- |
| `docs/profile_artifacts/stage6_ncu.ncu-rep` | 二进制报告，Nsight Compute GUI 可打开 |
| `docs/profile_artifacts/stage6_ncu_summary.csv` | 文本表，每行一个 kernel，列出各项指标 |

### 怎么看

打开 CSV，关键列：

| 列名 | 含义 |
| --- | --- |
| `Kernel Name` | kernel 函数名（RowSum / Gemm / ColSum） |
| `dram__bytes_read.sum` | 该 kernel 从 DRAM 读的总字节 |
| `dram__bytes_write.sum` | 该 kernel 写 DRAM 的总字节 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 计算利用率（%） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽利用率（%） |
| `gpu__time_duration.sum` | kernel 执行时间 |

确认要点：

- 只有 **RowSum / Gemm / ColSum** 三个 kernel
- 融合路径中间 f32 不落 DRAM（中间结果走寄存器/shared memory）
- 256³ 规模 `dram__bytes_write` 可能为 0（小输出停在 L2），不代表没写回；换大形状重跑会出现

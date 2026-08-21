# Stage 6 — TVM Relax + CUTLASS QGEMM BYOC

Reuses stage-5 `libcutlass_qgemm_bias_gelu_runtime.so` (affine zp + SFU FastGELU SIMT).
Relax matches the QDQ chain and rewrites it to a packed call.

## Setup

```bash
# From repo root. Build stages 1–5 so the .so exists, and register stage-6 ctest.
cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70 -DCU_EPILOGUE_ENABLE_TVM=ON
cmake --build build -j$(nproc)

# Python deps (project-local venv; do not use system pip)
./06_tvm_cutlass_byoc/scripts/setup_tvm.sh
source 06_tvm_cutlass_byoc/.venv/bin/activate
export CU_EPILOGUE_BUILD_DIR=$PWD/build
export PYTHONPATH=$PWD/06_tvm_cutlass_byoc/python:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/05_qdq_fusion:$LD_LIBRARY_PATH
```

## Run tests

**Preferred (no ctest path confusion):**

```bash
# still in the venv from Setup
pytest 06_tvm_cutlass_byoc/tests -q
```

**Via CTest** — tests are registered under the **build tree**, not the repo root:

```bash
# either:
cd build && ctest -R stage6 --output-on-failure

# or from repo root:
ctest --test-dir build -R stage6 --output-on-failure
```

If you run plain `ctest` in the repo root, you will see `No tests were found!!!`
because there is no `CTestTestfile.cmake` there.

Behavior of `stage6_tvm_tests`:

| Condition | Result |
| --- | --- |
| `CU_EPILOGUE_ENABLE_TVM=OFF` (default) | test target not registered |
| TVM venv missing / no stage-5 `.so` | script **exits 0 (SKIP)** with a message |
| TVM + `.so` OK | runs pytest |

## Bench & profile

```bash
python 06_tvm_cutlass_byoc/scripts/bench_e2e.py --m 1 --n 4096 --k 4096
./06_tvm_cutlass_byoc/scripts/profile_nsys.sh
./06_tvm_cutlass_byoc/scripts/profile_ncu.sh   # needs sudo; uses ncu 2025.2.1
# writes 06_tvm_cutlass_byoc/docs/profile_artifacts/stage6_ncu.ncu-rep
```

Report: [docs/benchmarking_report.md](docs/benchmarking_report.md).

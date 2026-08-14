#!/usr/bin/env bash
# Configures and builds the entire project (all 5 stages), then runs every
# static verification script plus `ctest`. On a machine without a live GPU
# (see docs/environment.md) the CUDA ctest entries are *expected* to
# fail at runtime with a CUDA driver/device error - the build and the
# static PTX/SASS checks are what matter there. On a real GPU machine CUDA
# tests plus stage-4/5 IR tests should pass.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_ARCH="${CU_EPILOGUE_CUDA_ARCH:-70}"

echo "== Configuring (CUDA arch = sm_${CUDA_ARCH}) =="
cmake -B build -DCU_EPILOGUE_CUDA_ARCH="$CUDA_ARCH" || exit 1

echo "== Building all targets =="
cmake --build build -j"$(nproc)" || exit 1

echo
echo "== Stage 2: static PTX mma.sync.aligned check =="
./02_cutlass_gemm/verify_mma_ptx.sh || echo "(non-fatal: see output above)"

echo
echo "== Stage 3: static PTX/SASS SFU instruction check =="
./03_fastgelu_epilogue/verify_sfu_sass.sh || echo "(non-fatal: see output above)"

echo
echo "== Stage 4: fusion pass test (runs fully in any environment, no GPU needed) =="
./04_compiler_integration/test/run_tests.sh build/04_compiler_integration/fused-opt

echo
echo "== Stage 5: QDQ tensor fusion (IR only, no GPU needed) =="
./05_qdq_fusion/test/run_tests.sh build/05_qdq_fusion/qdq-opt

echo
echo "== ctest (stage 1-3 and stage5 kernel tests need a real GPU to pass; see docs/environment.md) =="
(cd build && ctest --output-on-failure)

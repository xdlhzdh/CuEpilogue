#!/usr/bin/env bash
# Profile fused stage-6 path with Nsight Systems; expect one compute kernel
# (excluding memcpy / sync).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE6="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CU_EPILOGUE_BUILD_DIR:-${ROOT}/build}"
PY="${PYTHON:-python3}"
OUT_DIR="${STAGE6}/docs/profile_artifacts"
mkdir -p "${OUT_DIR}"

export CU_EPILOGUE_BUILD_DIR="${BUILD_DIR}"
export PYTHONPATH="${STAGE6}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${BUILD_DIR}/05_qdq_fusion:${LD_LIBRARY_PATH:-}"

REPORT="${OUT_DIR}/stage6_nsys"
rm -f "${REPORT}.nsys-rep" "${REPORT}.sqlite" || true

nsys profile -o "${REPORT}" --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  $PY - <<'PY'
from so_wrapper import launch, reset_launch_count
from qdq_reference import make_test_inputs
A, B, bias, sa, za, sb, zb, sd, zd = make_test_inputs(256, 256, 256)
reset_launch_count()
# Warmup + measured launches
for _ in range(3):
    launch(A, B, bias, sa, za, sb, zb, sd, zd)
print("done")
PY

echo "[nsys] report: ${REPORT}.nsys-rep"
if command -v nsys >/dev/null; then
  nsys stats --report cuda_gpu_kern_sum "${REPORT}.nsys-rep" 2>/dev/null | head -40 || true
fi

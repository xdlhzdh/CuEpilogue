#!/usr/bin/env bash
# CTest wrapper: SKIP (exit 0) when TVM or the stage-5 .so is unavailable.
set -euo pipefail

BUILD_DIR="${1:-${CU_EPILOGUE_BUILD_DIR:-}}"
STAGE6_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV="${CU_EPILOGUE_TVM_VENV:-${STAGE6_DIR}/.venv}"

if [[ -x "${VENV}/bin/python" ]]; then
  PY="${VENV}/bin/python"
else
  PY="${PYTHON:-python3}"
fi

if [[ -z "${BUILD_DIR}" ]]; then
  echo "[stage6] SKIP: BUILD_DIR not set"
  exit 0
fi

SO="${BUILD_DIR}/05_qdq_fusion/libcutlass_qgemm_bias_gelu_runtime.so"
if [[ ! -f "${SO}" ]]; then
  echo "[stage6] SKIP: missing ${SO} (build stage 5 with CUDA+MLIR first)"
  exit 0
fi

if ! $PY -c 'import tvm; from tvm import relax' 2>/dev/null; then
  echo "[stage6] SKIP: TVM+Relax not installed (run 06_tvm_cutlass_byoc/scripts/setup_tvm.sh)"
  exit 0
fi

export CU_EPILOGUE_BUILD_DIR="${BUILD_DIR}"
export PYTHONPATH="${STAGE6_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${BUILD_DIR}/05_qdq_fusion:${LD_LIBRARY_PATH:-}"

cd "${STAGE6_DIR}"
exec $PY -m pytest tests/ -q --tb=short

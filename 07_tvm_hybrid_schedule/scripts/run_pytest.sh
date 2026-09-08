#!/usr/bin/env bash
# CTest wrapper for stage 7. SKIP (exit 0) when TVM is unavailable.
# M1 always runs when TVM is present; M3/M4 may skip inside pytest without GPU.
set -euo pipefail

BUILD_DIR="${1:-${CU_EPILOGUE_BUILD_DIR:-}}"
STAGE7_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
STAGE6_DIR="${3:-$(cd "${STAGE7_DIR}/../06_tvm_cutlass_byoc" && pwd)}"
VENV="${CU_EPILOGUE_TVM_VENV:-${STAGE6_DIR}/.venv}"

if [[ -x "${VENV}/bin/python" ]]; then
  PY="${VENV}/bin/python"
else
  PY="${PYTHON:-python3}"
fi

if ! $PY -c 'import tvm; from tvm import relax; from tvm.s_tir import dlight' 2>/dev/null; then
  echo "[stage7] SKIP: TVM+Relax+DLight not installed (run 06_tvm_cutlass_byoc/scripts/setup_tvm.sh)"
  exit 0
fi

export CU_EPILOGUE_BUILD_DIR="${BUILD_DIR:-${STAGE7_DIR}/../build}"
export PYTHONPATH="${STAGE7_DIR}/python:${STAGE6_DIR}/python:${PYTHONPATH:-}"
if [[ -n "${BUILD_DIR}" && -d "${BUILD_DIR}/05_qdq_fusion" ]]; then
  export LD_LIBRARY_PATH="${BUILD_DIR}/05_qdq_fusion:${LD_LIBRARY_PATH:-}"
fi

cd "${STAGE7_DIR}"
exec $PY -m pytest tests/ -q --tb=short

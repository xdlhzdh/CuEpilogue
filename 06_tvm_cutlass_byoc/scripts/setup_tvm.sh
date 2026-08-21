#!/usr/bin/env bash
# Create a project-local venv and install Apache TVM + pytest for stage 6.
set -euo pipefail

STAGE6="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${CU_EPILOGUE_TVM_VENV:-${STAGE6}/.venv}"
PY_SYS="${PYTHON:-python3}"

if [[ ! -d "${VENV}" ]]; then
  echo "[setup_tvm] creating venv at ${VENV}"
  "${PY_SYS}" -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

echo "[setup_tvm] using interpreter: ${PY}"

if "${PY}" -c 'import tvm; from tvm import relax; print(tvm.__version__)' 2>/dev/null; then
  echo "[setup_tvm] TVM+Relax already available"
  exit 0
fi

echo "[setup_tvm] installing apache-tvm / numpy / pytest..."
"${PIP}" install -U pip
"${PIP}" install 'apache-tvm' 'numpy' 'pytest'

if ! "${PY}" -c 'import tvm; from tvm import relax; print("ok", tvm.__version__)'; then
  echo "[setup_tvm] ERROR: apache-tvm installed but Relax import failed."
  echo "  Try building from source: https://tvm.apache.org/docs/install/from_source.html"
  exit 1
fi

echo "[setup_tvm] done. Activate with: source ${VENV}/bin/activate"

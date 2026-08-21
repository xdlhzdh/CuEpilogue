#!/usr/bin/env bash
# Profile stage-6 fused .so with Nsight Compute (DRAM / SM metrics).
# Same ncu discovery pattern as 01_baseline_cuda/profile.sh; counters need sudo.
#
# Usage (from repo root):
#   ./06_tvm_cutlass_byoc/scripts/profile_ncu.sh
#
# Writes:
#   docs/profile_artifacts/stage6_ncu.ncu-rep
#   docs/profile_artifacts/stage6_ncu_summary.csv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE6="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CU_EPILOGUE_BUILD_DIR:-${ROOT}/build}"
OUT_DIR="${STAGE6}/docs/profile_artifacts"
mkdir -p "${OUT_DIR}"

if [[ -x "${STAGE6}/.venv/bin/python" ]]; then
  PY="${STAGE6}/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

# Absolute paths so sudo -E env still finds modules / .so.
export CU_EPILOGUE_BUILD_DIR="${BUILD_DIR}"
PY_PATH="${STAGE6}/python"
SO_DIR="${BUILD_DIR}/05_qdq_fusion"
SO="${SO_DIR}/libcutlass_qgemm_bias_gelu_runtime.so"
if [[ ! -f "${SO}" ]]; then
  echo "error: missing ${SO}" >&2
  echo "       build stage 5 first: cmake -B build && cmake --build build -j" >&2
  exit 1
fi

find_working_ncu() {
  local cand
  local -a tried=() cands=()
  [[ -n "${NCU_BIN:-}" ]] && cands+=("${NCU_BIN}")
  [[ -x /opt/nvidia/nsight-compute/2025.2.1/ncu ]] && cands+=(/opt/nvidia/nsight-compute/2025.2.1/ncu)
  for cand in /opt/nvidia/nsight-compute/*/ncu; do
    [[ -x "$cand" ]] && cands+=("$cand")
  done
  command -v ncu >/dev/null && cands+=("$(command -v ncu)")

  for cand in "${cands[@]}"; do
    [[ -x "$cand" ]] || continue
    [[ " ${tried[*]-} " == *" $cand "* ]] && continue
    tried+=("$cand")
    echo "-- trying ncu at $cand --" >&2
    # Prefer a binary that starts without driver/version mismatch.
    if sudo -n "$cand" --version >/tmp/ncu_ver_$$.txt 2>&1 \
       || sudo "$cand" --version >/tmp/ncu_ver_$$.txt 2>&1; then
      if grep -qiE 'stub libcuda|unsupported driver|incompatible' /tmp/ncu_ver_$$.txt; then
        rm -f /tmp/ncu_ver_$$.txt
        continue
      fi
      rm -f /tmp/ncu_ver_$$.txt
      # Skip known-bad 2026.x on driver 535.
      if [[ "$cand" == *"/2026."* ]]; then
        echo "skipping $cand (incompatible with driver 535)" >&2
        continue
      fi
      echo "$cand"
      return 0
    fi
    rm -f /tmp/ncu_ver_$$.txt
  done
  return 1
}

NCU_BIN="$(find_working_ncu)" || {
  echo "error: no compatible ncu found. Prefer /opt/nvidia/nsight-compute/2025.2.1/ncu" >&2
  exit 1
}
echo "[ncu] using ${NCU_BIN}"

TMP_PY="${OUT_DIR}/_ncu_launch.py"
cat >"${TMP_PY}" <<'PY'
"""One fused launch for ncu (256^3, same scale as nsys report)."""
from so_wrapper import launch
from qdq_reference import make_test_inputs

A, B, bias, sa, za, sb, zb, sd, zd = make_test_inputs(256, 256, 256)
launch(A, B, bias, sa, za, sb, zb, sd, zd)
print("ncu-launch-done", flush=True)
PY

REPORT="${OUT_DIR}/stage6_ncu"
CSV="${OUT_DIR}/stage6_ncu_summary.csv"

METRICS=(
  launch__grid_size
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes_read.sum
  dram__bytes_write.sum
  dram__sectors_read.sum
  dram__sectors_write.sum
  gpu__time_duration.sum
)
METRIC_CSV="$(IFS=,; echo "${METRICS[*]}")"

echo "[ncu] profiling → ${REPORT}.ncu-rep"
# sudo drops PYTHONPATH by default; pass env explicitly with -E env.
sudo -E env \
  "PATH=${PATH}" \
  "CU_EPILOGUE_BUILD_DIR=${BUILD_DIR}" \
  "PYTHONPATH=${PY_PATH}" \
  "LD_LIBRARY_PATH=${SO_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${NCU_BIN}" \
  --target-processes all \
  --kernel-name-base demangled \
  --metrics "${METRIC_CSV}" \
  -o "${REPORT}" \
  --force-overwrite \
  "${PY}" "${TMP_PY}"

echo "[ncu] exporting CSV → ${CSV}"
sudo "${NCU_BIN}" --import "${REPORT}.ncu-rep" --csv --page raw >"${CSV}" 2>/dev/null \
  || sudo "${NCU_BIN}" --import "${REPORT}.ncu-rep" --csv >"${CSV}"

sudo chown "$(id -u):$(id -g)" "${REPORT}.ncu-rep" "${CSV}" "${TMP_PY}" 2>/dev/null || true

echo "[ncu] done"
ls -lh "${REPORT}.ncu-rep" "${CSV}"

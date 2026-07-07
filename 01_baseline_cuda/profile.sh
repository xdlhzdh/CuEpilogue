#!/usr/bin/env bash
# Stage 1 profiling script - run this on a machine with an actual NVIDIA GPU
# (e.g. the Tesla V100 host), NOT inside the compile-only sandbox this
# project may have been scaffolded in.
#
# Usage:
#   ./01_baseline_cuda/profile.sh /path/to/build/01_baseline_cuda/correctness_test
#
# Produces:
#   sgemm_baseline.nsys-rep   - timeline / occupancy overview (Nsight Systems)
#   sgemm_baseline.ncu-rep    - detailed kernel metrics (Nsight Compute)
#
# `ncu` needs two things this repo's dev host didn't have out of the box
# (see docs/environment.md):
#   1. Root/CAP_SYS_ADMIN: the NVIDIA driver restricts GPU performance
#      counters to admin users by default (`RmProfilingAdminOnly=1`). We run
#      `ncu` under `sudo`; without it you'll see `ERR_NVGPUCTRPERM`.
#   2. A `ncu` build that's actually compatible with the installed driver.
#      If multiple Nsight Compute versions are installed (e.g. one bundled
#      with a newer CUDA toolkit than the driver supports), the too-new one
#      fails with a *misleading* "failed to connect to the CUDA driver
#      (stub libcuda.so[.1] on path?)" error even though libcuda.so resolves
#      fine - the real cause is a driver/tool version mismatch. This script
#      probes every installed `ncu` (PATH + /opt/nvidia/nsight-compute/*)
#      newest-first and uses the first one that actually connects.
set -uo pipefail

BIN="${1:?usage: profile.sh <path-to-correctness_test-binary>}"
OUT_DIR="$(dirname "$0")/profile_reports"
mkdir -p "$OUT_DIR"

echo "== nsys: timeline, occupancy, achieved SM/memory throughput =="
nsys profile \
  --stats=true \
  --trace=cuda,osrt \
  -o "$OUT_DIR/sgemm_baseline" \
  --force-overwrite=true \
  "$BIN" || exit 1

find_working_ncu() {
  local candidates=()
  command -v ncu >/dev/null && candidates+=("$(command -v ncu)")
  while IFS= read -r -d '' p; do candidates+=("$p"); done < \
    <(find /opt/nvidia/nsight-compute -maxdepth 2 -name ncu -type f -print0 2>/dev/null | sort -zrV)

  local tried=() cand
  for cand in "${candidates[@]}"; do
    [[ " ${tried[*]-} " == *" $cand "* ]] && continue
    tried+=("$cand")
    echo "-- trying ncu at $cand --" >&2
    if sudo "$cand" --metrics launch__grid_size "$BIN" >/tmp/ncu_probe.$$ 2>&1; then
      rm -f /tmp/ncu_probe.$$
      echo "$cand"
      return 0
    fi
    if grep -q "ERR_NVGPUCTRPERM" /tmp/ncu_probe.$$; then
      echo "found a working ncu ($cand) but permission check failed even under sudo - see ERR_NVGPUCTRPERM link in its output" >&2
      cat /tmp/ncu_probe.$$ >&2
      rm -f /tmp/ncu_probe.$$
      return 1
    fi
    rm -f /tmp/ncu_probe.$$
  done
  return 1
}

echo
echo "== ncu: per-kernel occupancy + memory bandwidth vs. theoretical peak =="
NCU_BIN="$(find_working_ncu)" || {
  echo "error: no installed ncu could connect to the CUDA driver (tried PATH + /opt/nvidia/nsight-compute/*)." >&2
  echo "       nsys data above is still valid; see docs/environment.md for ncu troubleshooting." >&2
  exit 1
}
echo "using $NCU_BIN"
sudo "$NCU_BIN" --set full \
  -o "$OUT_DIR/sgemm_baseline" \
  --force-overwrite \
  "$BIN"

echo
echo "Reports written to $OUT_DIR/"
echo "Open sgemm_baseline.nsys-rep / .ncu-rep with 'nsys-ui' / 'ncu-ui', or inspect"
echo "textually with: nsys stats <file>.nsys-rep  /  $NCU_BIN --import <file>.ncu-rep"

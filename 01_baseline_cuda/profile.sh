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
set -euo pipefail

BIN="${1:?usage: profile.sh <path-to-correctness_test-binary>}"
OUT_DIR="$(dirname "$0")/profile_reports"
mkdir -p "$OUT_DIR"

echo "== nsys: timeline, occupancy, achieved SM/memory throughput =="
nsys profile \
  --stats=true \
  --trace=cuda,osrt \
  -o "$OUT_DIR/sgemm_baseline" \
  "$BIN"

echo "== ncu: per-kernel occupancy + memory bandwidth vs. theoretical peak =="
ncu --set full \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  -o "$OUT_DIR/sgemm_baseline" \
  "$BIN"

echo "Reports written to $OUT_DIR/"
echo "Open sgemm_baseline.nsys-rep / .ncu-rep with 'nsys-ui' / 'ncu-ui', or inspect"
echo "textually with: nsys stats <file>.nsys-rep  /  ncu --import <file>.ncu-rep"

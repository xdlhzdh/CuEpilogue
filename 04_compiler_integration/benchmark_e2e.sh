#!/usr/bin/env bash
# Stage 4 end-to-end latency benchmark (spec 3.4 acceptance).
#
# Tiers:
#   A - Unfused: fused-opt (no --fuse-gemm-gelu) -> scalar CPU loops -> mlir-runner JIT
#   B - Unfused GPU AOT: linalg -> gpu.launch -> embedded PTX host ELF (optional; see
#       docs/aot_gpu_baseline.md and scripts/build_aot_gpu_baseline.sh)
#   C - Fused: bench_fused_e2e -> CUTLASS Fast-GELU kernel (external-call target)
#
# Spec ">=30% faster than 基础指令展开" is checked as A vs C. B vs C is the fairer
# compiler baseline when NVPTX-enabled MLIR is available.
#
# Usage: ./04_compiler_integration/benchmark_e2e.sh [path/to/build/dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build}"

FUSED_OPT="$BUILD_DIR/04_compiler_integration/fused-opt"
BENCH_FUSED_E2E="$BUILD_DIR/04_compiler_integration/bench_fused_e2e"

command -v mlir-runner >/dev/null || {
  echo "error: mlir-runner not found in PATH (needs LLVM/MLIR installed)." >&2
  exit 1
}
[[ -x "$FUSED_OPT" ]] || {
  echo "error: $FUSED_OPT not built. Run cmake --build $BUILD_DIR first." >&2
  exit 1
}
[[ -x "$BENCH_FUSED_E2E" ]] || {
  echo "error: $BENCH_FUSED_E2E not built (needs CUDA stages enabled, i.e." >&2
  echo "-DCU_EPILOGUE_ENABLE_CUDA=ON, the default)." >&2
  exit 1
}

MLIR_BIN_DIR="$(dirname "$(command -v mlir-runner)")"
RUNNER_UTILS="$MLIR_BIN_DIR/../lib/libmlir_runner_utils.so"
C_RUNNER_UTILS="$MLIR_BIN_DIR/../lib/libmlir_c_runner_utils.so"
[[ -f "$RUNNER_UTILS" && -f "$C_RUNNER_UTILS" ]] || {
  echo "error: libmlir_runner_utils.so / libmlir_c_runner_utils.so not found next to mlir-runner." >&2
  exit 1
}

echo "== Fused path: direct call into the CUTLASS Fast-GELU kernel (same one the compiler's external call targets) =="
FUSED_MS="$("$BENCH_FUSED_E2E")"
echo "20 iterations, total: ${FUSED_MS} ms"

echo
echo "== Unfused baseline: fused-opt (no --fuse-gemm-gelu) -> scalar CPU loops -> mlir-runner =="
LOWERED="$(mktemp)"
"$FUSED_OPT" "$SCRIPT_DIR/test/bench_unfused.mlir" \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-math-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts >"$LOWERED"

UNFUSED_SEC="$(mlir-runner "$LOWERED" -e main --entry-point-result=void \
  --shared-libs="$RUNNER_UTILS,$C_RUNNER_UTILS")"
rm -f "$LOWERED"
UNFUSED_MS="$(awk -v s="$UNFUSED_SEC" 'BEGIN { printf "%.6f", s * 1000 }')"
echo "20 iterations, total: ${UNFUSED_MS} ms"

AOT_GPU_BIN="$SCRIPT_DIR/docs/work_dirs/aot_gpu_baseline/bench_aot_gpu_unfused"
AOT_GPU_MS=""
if [[ -x "$AOT_GPU_BIN" ]]; then
  echo
  echo "== GPU AOT unfused baseline: linalg -> embedded PTX host ELF (see docs/aot_gpu_baseline.md) =="
  AOT_OUT="$("$AOT_GPU_BIN")"
  # bench_unfused.mlir prints seconds via @printF64 (same as mlir-runner).
  AOT_GPU_MS="$(echo "$AOT_OUT" | awk '/^[0-9]/ { printf "%.6f", $1 * 1000; exit }')"
  echo "20 iterations, total: ${AOT_GPU_MS} ms"
else
  echo
  echo "== GPU AOT unfused baseline: skipped (not built) =="
  echo "Build: ./04_compiler_integration/scripts/build_aot_gpu_baseline.sh"
  echo "       (requires MLIR/LLVM with NVPTX; run check_aot_gpu_prereqs.sh)"
fi

echo
echo "== Comparison =="
if [[ -n "$AOT_GPU_MS" ]]; then
  awk -v fused="$FUSED_MS" -v cpu="$UNFUSED_MS" -v gpu="$AOT_GPU_MS" 'BEGIN {
    drop_cpu = (cpu - fused) / cpu * 100.0;
    drop_gpu = (gpu - fused) / gpu * 100.0;
    printf "C fused CUTLASS:     %.4f ms (20 iters)\n", fused;
    printf "A unfused CPU JIT:   %.4f ms (20 iters)  -> vs C: %.1f%% faster (spec >= 30%%)\n", cpu, drop_cpu;
    printf "B unfused GPU AOT:   %.4f ms (20 iters)  -> vs C: %.1f%% faster\n", gpu, drop_gpu;
  }'
else
  awk -v fused="$FUSED_MS" -v unfused="$UNFUSED_MS" 'BEGIN {
    drop = (unfused - fused) / unfused * 100.0;
    printf "C fused CUTLASS:   %.4f ms (20 iters)\nA unfused CPU JIT: %.4f ms (20 iters)\nlatency change (A vs C): %.1f%% (positive = fused is faster; spec 3.4 target: >= 30%%)\n", fused, unfused, drop;
  }'
fi

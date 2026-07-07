#!/usr/bin/env bash
# Stage 4 end-to-end latency benchmark (spec 3.4 acceptance: the fused
# external-call path should be >=30% faster than "基础指令展开" - the
# plain scalar loop nest MLIR would produce WITHOUT the --fuse-gemm-gelu
# pass).
#
# "Fused" side: bench_fused_e2e (built by CMake) calls directly into the
# same CUTLASS kernel that libcutlass_fused_gemm_gelu_runtime.so wraps
# behind the C symbol the compiler pass emits a call to (see
# test/bench_fused_e2e.cu for why this bypasses fused-opt/mlir-runner).
#
# "Unfused" side: test/bench_unfused.mlir, lowered by fused-opt WITHOUT
# --fuse-gemm-gelu (plain linalg.matmul + linalg.generic ->
# --convert-linalg-to-loops -> scalar CPU loops), executed via mlir-runner.
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

echo
echo "== Comparison =="
awk -v fused="$FUSED_MS" -v unfused="$UNFUSED_MS" 'BEGIN {
  drop = (unfused - fused) / unfused * 100.0;
  printf "fused:   %.4f ms (20 iters)\nunfused: %.4f ms (20 iters)\nlatency change: %.1f%% (positive = fused is faster; spec 3.4 target: >= 30%%)\n", fused, unfused, drop;
}'

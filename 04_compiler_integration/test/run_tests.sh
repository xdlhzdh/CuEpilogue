#!/usr/bin/env bash
# Stage 4 test runner. No FileCheck/llvm-lit is available in this
# environment (see docs/environment.md), so assertions are done with
# plain grep against `fused-opt`'s output - equivalent in spirit to a
# FileCheck test, just less concise.
#
# Usage: ./04_compiler_integration/test/run_tests.sh [path/to/fused-opt]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

FUSED_OPT="${1:-}"
if [[ -z "$FUSED_OPT" ]]; then
  for candidate in \
      "$ROOT_DIR"/build*/04_compiler_integration/fused-opt \
      "$ROOT_DIR"/build*/04_compiler_integration/tools/fused-opt; do
    if [[ -x "$candidate" ]]; then FUSED_OPT="$candidate"; break; fi
  done
fi
if [[ -z "$FUSED_OPT" || ! -x "$FUSED_OPT" ]]; then
  echo "error: could not locate the fused-opt binary. Build it first with:" >&2
  echo "  cmake -B build && cmake --build build --target fused-opt" >&2
  echo "or pass its path as arg 1." >&2
  exit 1
fi

BUFFERIZE_OPTS='bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map'
OUT="$(mktemp)"
fail=0

check_present() {
  if grep -q -- "$1" "$OUT"; then
    echo "PASS: found expected pattern: $1"
  else
    echo "FAIL: expected pattern not found: $1" >&2
    fail=1
  fi
}

check_absent() {
  if grep -q -- "$1" "$OUT"; then
    echo "FAIL: unexpected pattern still present (should have been fused away): $1" >&2
    fail=1
  else
    echo "PASS: pattern correctly absent: $1"
  fi
}

run_fuse_case() {
  local label="$1"
  local input="$2"

  echo "== Running fused-opt ($label): one-shot-bufferize + fuse-gemm-gelu =="
  "$FUSED_OPT" "$input" \
    --one-shot-bufferize="$BUFFERIZE_OPTS" \
    --fuse-gemm-gelu \
    --canonicalize \
    > "$OUT"

  echo "--- fused-opt output ($label) ---"
  cat "$OUT"
  echo "---------------------------------"
}

run_fuse_case "default GELU" "$SCRIPT_DIR/fuse_pattern.mlir"
check_present "call @cutlass_fused_gemm_gelu"
check_present "func.func private @cutlass_fused_gemm_gelu"
check_absent "linalg.matmul"
check_absent "linalg.generic"
check_present "1.000000e+00"
check_present "0.000000e+00"

run_fuse_case "alpha/beta GELU" "$SCRIPT_DIR/fuse_pattern_alpha_beta.mlir"
check_present "call @cutlass_fused_gemm_gelu"
check_present "func.func private @cutlass_fused_gemm_gelu"
check_absent "linalg.matmul"
check_absent "linalg.generic"
check_present "2.000000e+00"
check_present "5.000000e-01"

rm -f "$OUT"

if [[ "$fail" == "0" ]]; then
  echo "ALL STAGE-4 TESTS PASSED"
else
  echo "SOME STAGE-4 TESTS FAILED" >&2
  exit 1
fi

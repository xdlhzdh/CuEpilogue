#!/usr/bin/env bash
# Stage 5: tensor fuse -> OSB -> func.call. No GPU required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

QDQ_OPT="${1:-}"
if [[ -z "$QDQ_OPT" ]]; then
  for candidate in "$ROOT_DIR"/build*/05_qdq_fusion/qdq-opt; do
    if [[ -x "$candidate" ]]; then QDQ_OPT="$candidate"; break; fi
  done
fi
if [[ -z "$QDQ_OPT" || ! -x "$QDQ_OPT" ]]; then
  echo "error: could not locate qdq-opt. Build with:" >&2
  echo "  cmake -B build && cmake --build build --target qdq-opt" >&2
  exit 1
fi

INPUT="$SCRIPT_DIR/qdq_pattern.mlir"
BUFFERIZE_OPTS='bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map'
fail=0
OUT="$(mktemp)"

check_present() {
  if grep -q -- "$1" "$OUT"; then
    echo "PASS: found: $1"
  else
    echo "FAIL: missing: $1" >&2
    fail=1
  fi
}
check_absent() {
  if grep -q -- "$1" "$OUT"; then
    echo "FAIL: still present: $1" >&2
    fail=1
  else
    echo "PASS: absent: $1"
  fi
}

echo "== stage1: fuse-qdq-qgemm-bias-gelu =="
"$QDQ_OPT" "$INPUT" --fuse-qdq-qgemm-bias-gelu --canonicalize > "$OUT"
cat "$OUT"
check_present "cutlass.qgemm_bias_gelu"
check_absent "linalg.matmul"
check_absent "linalg.generic"

echo "== stage2: one-shot-bufferize =="
"$QDQ_OPT" "$INPUT" --fuse-qdq-qgemm-bias-gelu \
  --one-shot-bufferize="$BUFFERIZE_OPTS" --canonicalize > "$OUT"
cat "$OUT"
check_present "cutlass.qgemm_bias_gelu"
check_present "memref<"
check_absent "linalg.matmul"
check_absent "tensor.empty"

echo "== stage3: lower-cutlass-qgemm-to-call =="
"$QDQ_OPT" "$INPUT" --fuse-qdq-qgemm-bias-gelu \
  --one-shot-bufferize="$BUFFERIZE_OPTS" \
  --lower-cutlass-qgemm-to-call --canonicalize > "$OUT"
cat "$OUT"
check_present "call @cutlass_qgemm_bias_gelu"
check_present "func.func private @cutlass_qgemm_bias_gelu"
check_absent "cutlass.qgemm_bias_gelu ins"
check_absent "linalg.matmul"

rm -f "$OUT"
if [[ "$fail" == "0" ]]; then
  echo "ALL STAGE-5 TESTS PASSED"
else
  echo "SOME STAGE-5 TESTS FAILED" >&2
  exit 1
fi

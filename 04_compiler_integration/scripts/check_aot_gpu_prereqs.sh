#!/usr/bin/env bash
# Prerequisites for stage-4 GPU AOT baseline (linalg -> embedded PTX in host ELF).
set -euo pipefail

ok=0
fail() { echo "FAIL: $*" >&2; ok=1; }
pass() { echo "OK:   $*"; }

for tool in mlir-opt mlir-translate llvm-config clang++; do
  if command -v "$tool" >/dev/null 2>&1; then
    pass "$tool -> $(command -v "$tool")"
  else
    fail "$tool not found in PATH"
  fi
done

if command -v llvm-config >/dev/null 2>&1; then
  targets="$(llvm-config --targets-built 2>/dev/null || true)"
  echo "     llvm targets-built: ${targets:-<unknown>}"
  if [[ "$targets" != *NVPTX* ]]; then
    fail "LLVM was not built with NVPTX (need LLVM_TARGETS_TO_BUILD including NVPTX)"
    echo "       gpu-lower-to-nvvm-pipeline / gpu-module-to-binary will not work." >&2
  else
    pass "NVPTX target registered"
  fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  cuda_root="$CUDA_HOME"
elif [[ -d /usr/local/cuda ]]; then
  cuda_root=/usr/local/cuda
else
  cuda_root=""
fi

if [[ -n "$cuda_root" && -f "$cuda_root/include/cuda.h" ]]; then
  pass "CUDA toolkit at $cuda_root"
else
  fail "CUDA toolkit not found (set CUDA_HOME or install under /usr/local/cuda)"
fi

MLIR_BIN_DIR=""
if command -v mlir-opt >/dev/null 2>&1; then
  MLIR_BIN_DIR="$(dirname "$(command -v mlir-opt)")"
  for lib in libmlir_runner_utils.so libmlir_c_runner_utils.so; do
    if [[ -f "$MLIR_BIN_DIR/../lib/$lib" ]]; then
      pass "$lib"
    else
      fail "$MLIR_BIN_DIR/../lib/$lib missing (needed for @rtclock in bench_unfused.mlir)"
    fi
  done
fi

if [[ "$ok" -ne 0 ]]; then
  echo >&2
  echo "Fix the items above, then run:" >&2
  echo "  ./04_compiler_integration/scripts/build_aot_gpu_baseline.sh" >&2
  exit 1
fi

echo
echo "All prerequisites satisfied for GPU AOT baseline build."
exit 0

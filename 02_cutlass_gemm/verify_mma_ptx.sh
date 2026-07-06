#!/usr/bin/env bash
# Stage 2 acceptance check (spec 3.2): statically confirm the CUTLASS GEMM
# lowers to aligned Tensor Core MMA instructions. This is a *static*
# compilation check - it only needs nvcc/ptxas, not a live GPU - so it can
# run inside a compile-only sandbox.
#
# Usage: ./02_cutlass_gemm/verify_mma_ptx.sh [path/to/cutlass/include/root]
# If no path is given, this script looks for a CMake FetchContent checkout
# under ../build/_deps/cutlass-src.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCH="${CU_EPILOGUE_CUDA_ARCH:-70}"

CUTLASS_DIR="${1:-}"
if [[ -z "$CUTLASS_DIR" ]]; then
  for candidate in "$ROOT_DIR"/build*/_deps/cutlass-src; do
    if [[ -d "$candidate" ]]; then CUTLASS_DIR="$candidate"; break; fi
  done
fi
if [[ -z "$CUTLASS_DIR" || ! -d "$CUTLASS_DIR" ]]; then
  echo "error: could not locate CUTLASS checkout. Pass its root as arg 1," >&2
  echo "or run 'cmake -B build' first so FetchContent populates build/_deps/cutlass-src." >&2
  exit 1
fi

PTX_OUT="$(mktemp -d)/cutlass_gemm.ptx"
echo "Compiling $SCRIPT_DIR/cutlass_gemm.cu -> PTX for sm_${ARCH} ..."
nvcc -arch="sm_${ARCH}" \
  --expt-relaxed-constexpr --expt-extended-lambda -std=c++17 \
  -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
  -I"$ROOT_DIR/common" \
  -ptx "$SCRIPT_DIR/cutlass_gemm.cu" -o "$PTX_OUT"

echo "== Searching for aligned MMA instructions in $PTX_OUT =="
if grep -n "mma\.sync\.aligned" "$PTX_OUT"; then
  echo "PASS: found mma.sync.aligned instruction(s) in generated PTX."
else
  echo "FAIL: no mma.sync.aligned instruction found - check ThreadblockShape/WarpShape/arch." >&2
  exit 1
fi

echo
echo "== Register-spill risk hint: ld.local / st.local occurrences =="
grep -c "ld\.local\|st\.local" "$PTX_OUT" || echo "0 (no local-memory spills detected in PTX)"

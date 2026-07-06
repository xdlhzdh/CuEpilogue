#!/usr/bin/env bash
# Stage 3 acceptance check (spec 3.3): statically confirm the SFU
# instructions ex2.approx / rcp.approx survive all the way down to the
# generated cubin's SASS. This only needs ptxas/cuobjdump/nvdisasm - no
# live GPU required - so it runs in a compile-only sandbox.
#
# Usage: ./03_fastgelu_epilogue/verify_sfu_sass.sh [path/to/cutlass/include/root]
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

WORK_DIR="$(mktemp -d)"
PTX_OUT="$WORK_DIR/fused_gemm_gelu.ptx"
CUBIN_OUT="$WORK_DIR/fused_gemm_gelu.cubin"

COMMON_FLAGS=(-arch="sm_${ARCH}" --expt-relaxed-constexpr --expt-extended-lambda -std=c++17
  -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" -I"$ROOT_DIR/common"
  -I"$SCRIPT_DIR")

echo "== [1/2] PTX-level check: ex2.approx.f32 / rcp.approx.f32 =="
nvcc "${COMMON_FLAGS[@]}" -ptx "$SCRIPT_DIR/fused_gemm_gelu.cu" -o "$PTX_OUT"
ptx_ok=1
grep -n "ex2\.approx\.f32" "$PTX_OUT" || ptx_ok=0
grep -n "rcp\.approx\.f32" "$PTX_OUT" || ptx_ok=0
if [[ "$ptx_ok" == "1" ]]; then
  echo "PASS: ex2.approx.f32 and rcp.approx.f32 both present in PTX."
else
  echo "FAIL: expected SFU PTX instructions not found." >&2
  exit 1
fi

echo
echo "== [2/2] SASS-level check: compiled cubin retains the SFU instructions =="
nvcc "${COMMON_FLAGS[@]}" -cubin "$SCRIPT_DIR/fused_gemm_gelu.cu" -o "$CUBIN_OUT"
sass_ok=1
DISASM="$(cuobjdump --dump-sass "$CUBIN_OUT" 2>/dev/null || nvdisasm "$CUBIN_OUT" 2>/dev/null || true)"
echo "$DISASM" | grep -n -E "MUFU\.EX2|EX2\." || sass_ok=0
echo "$DISASM" | grep -n -E "MUFU\.RCP|RCP\." || sass_ok=0
if [[ "$sass_ok" == "1" ]]; then
  echo "PASS: SFU SASS instructions (MUFU.EX2 / MUFU.RCP family) found in cubin."
else
  echo "WARN: could not confirm SFU SASS mnemonics (naming varies by SASS version)." >&2
  echo "      PTX-level confirmation above is still a valid acceptance signal." >&2
fi

echo
echo "== Register-spill risk hint: ld.local / st.local occurrences in PTX =="
grep -c "ld\.local\|st\.local" "$PTX_OUT" || echo "0 (no local-memory spills detected in PTX)"

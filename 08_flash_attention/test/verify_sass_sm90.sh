#!/usr/bin/env bash
# Stage 8 static acceptance: compile the Sm90 FlashAttention kernel for sm_90a
# and confirm the fused mainloop is built from the intended hardware paths.
# No Hopper GPU required.
#   HGMMA (SS)     GEMM1 S = Q*K^T, both operands from SMEM descriptors
#   HGMMA (RS)     GEMM2 O += P*V, P from registers, V transposed descriptor
#   UTMALDG        TMA loads of Q/K/V
#   UTMASTG        TMA store of O
#   SYNCS          mbarrier traffic of PipelineTmaAsync / Q barrier
#   MUFU.EX2       online-softmax exponentials
#   USETMAXREG     producer/consumer register reallocation
# Plus, per kernel, that the overlapped schedules really run the softmax while
# GEMM2 is in flight, and no local-memory spills (LDL/STL).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARCH=90a

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
CUBIN_OUT="$WORK_DIR/flash_attn_fwd_sm90.cubin"
trap 'rm -rf "$WORK_DIR"' EXIT

SRC="$SCRIPT_DIR/../kernels/flash_attn_fwd_sm90.cu"
nvcc -arch="sm_${ARCH}" -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
  -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" -I"$ROOT_DIR/common" \
  -I"$SCRIPT_DIR/../kernels" -Xptxas -v -cubin "$SRC" -o "$CUBIN_OUT" 2> "$WORK_DIR/ptxas.log"

echo "== ptxas resource usage =="
grep -E "Compiling entry|Used [0-9]+ registers|spill" "$WORK_DIR/ptxas.log" || true

SASS="$WORK_DIR/flash_attn_fwd_sm90.sass"
cuobjdump --dump-sass "$CUBIN_OUT" > "$SASS"

kernels=$(grep -c "Function :" "$SASS")
echo
echo "== SASS checks over $kernels kernel instantiations (head_dim x causal) =="
ok=1
check() {
  local label="$1" pattern="$2"
  local n
  n=$(grep -cE "$pattern" "$SASS" || true)
  if [[ "$n" -gt 0 ]]; then
    printf "PASS  %-12s %5d  (%s)\n" "$label" "$n" "$pattern"
  else
    printf "FAIL  %-12s %5d  (%s)\n" "$label" "$n" "$pattern"
    ok=0
  fi
}
check "HGMMA_SS"   'HGMMA\.[0-9x]+\.F32 R[0-9]+, gdesc'
check "HGMMA_RS"   'HGMMA\.[0-9x]+\.F32 R[0-9]+, R[0-9]+, gdesc\[UR[0-9]+\]\.tnspB'
check "UTMALDG"    'UTMALDG'
check "UTMASTG"    'UTMASTG'
check "SYNCS"      'SYNCS'
check "MUFU.EX2"   'MUFU\.EX2'
check "USETMAXREG" 'USETMAXREG'

# Overlapped schedules (FlashAttnSchedule 1 = intra-WG overlap, 2 = ping-pong):
# the softmax must execute between "WGMMA wait <= 1" (GEMM1 done, GEMM2 still
# in flight) and "wait <= 0". Every such window needs the exponentials; a
# window without them means ptxas drained the pipeline early again.
# Ping-pong kernels must also hand turns over with named-barrier arrives.
echo
echo "== Overlap checks per kernel (window = WGMMA wait<=1 .. wait<=0) =="
overlap_report=$(awk '
  function finish() {
    if (name == "") return
    sched = name; sub(/.*FlashAttnScheduleE/, "", sched); sched = substr(sched, 1, 1)
    # FusedAttentionMainloopILi<D>ELi<M>ELi<N>ELi<S>ELb<causal>E...
    cfg = name; sub(/.*FusedAttentionMainloopILi/, "", cfg)
    split(cfg, f, "E")
    tag = sprintf("D=%-3s %-6s", f[1], (substr(f[5], 3, 1) == "1" ? "causal" : ""))
    if (sched == "0") {
      status = (windows == 0) ? "PASS" : "FAIL"
      printf "%s  %s serial    windows=%d (expect 0)\n", status, tag, windows
    } else {
      status = (windows > 0 && empty == 0) ? "PASS" : "FAIL"
      if (sched == "2" && arrives == 0) status = "FAIL"
      printf "%s  %s %-8s  windows=%d without_softmax=%d min_EX2_per_window=%d bar.arrive=%d\n",
             status, tag, (sched == "1" ? "overlap" : "pingpong"), windows, empty, minex, arrives
    }
  }
  /Function :/ { finish(); name = $3; windows = 0; empty = 0; minex = 1e9; arrives = 0; inwin = 0; next }
  /WARPGROUP\.DEPBAR\.LE gsb0, 0x1/ { inwin = 1; ex = 0; next }
  /WARPGROUP\.DEPBAR\.LE gsb0, 0x0/ {
    if (inwin) { windows++; if (ex < 32) empty++; if (ex < minex) minex = ex }
    inwin = 0; next
  }
  /MUFU\.EX2/ { if (inwin) ex++ }
  /BAR\.ARV/ { arrives++ }
  END { finish() }' "$SASS")
echo "$overlap_report"
if echo "$overlap_report" | grep -q '^FAIL'; then
  ok=0
fi

echo
# GEMM operands come from smem through WGMMA descriptors and the epilogue math
# runs on the accumulator registers; smem is only written to stage O for the
# TMA store. Any shared-memory load means a stage started reading smem back.
smem_loads=$(grep -cE '\bLDS(M)?\b' "$SASS" || true)
if [[ "$smem_loads" -eq 0 ]]; then
  echo "PASS  no shared-memory loads (LDS/LDSM): epilogue computes on accumulator registers"
else
  echo "FAIL  $smem_loads shared-memory LDS/LDSM instructions"
  ok=0
fi

spills=$(grep -cE '\b(LDL|STL)\b' "$SASS" || true)
if [[ "$spills" -eq 0 ]]; then
  echo "PASS  no local-memory spills (LDL/STL)"
else
  echo "FAIL  $spills local-memory LDL/STL instructions"
  ok=0
fi

if [[ "$ok" == "1" ]]; then
  echo "ALL SASS CHECKS PASSED"
else
  echo "SASS CHECKS FAILED" >&2
  exit 1
fi

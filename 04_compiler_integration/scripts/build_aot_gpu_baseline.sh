#!/usr/bin/env bash
# Build stage-4 unfused GPU AOT baseline: linalg -> gpu.launch -> embedded PTX -> host ELF.
#
# Output (on success):
#   04_compiler_integration/docs/work_dirs/aot_gpu_baseline/bench_aot_gpu_unfused
#
# Requires LLVM/MLIR with NVPTX (see check_aot_gpu_prereqs.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE4_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$STAGE4_DIR/.." && pwd)"
WORK_DIR="$STAGE4_DIR/docs/work_dirs/aot_gpu_baseline"
INPUT_MLIR="$STAGE4_DIR/test/bench_unfused.mlir"
OUT_BIN="$WORK_DIR/bench_aot_gpu_unfused"

SM="${CU_EPILOGUE_CUDA_ARCH:-70}"
CUBIN_CHIP="sm_${SM}"
CUBIN_TRIPLE="nvptx64-nvidia-cuda"

mkdir -p "$WORK_DIR"

if ! "$SCRIPT_DIR/check_aot_gpu_prereqs.sh"; then
  exit 1
fi

MLIR_BIN_DIR="$(dirname "$(command -v mlir-opt)")"
MLIR_LIB_DIR="$(cd "$MLIR_BIN_DIR/../lib" && pwd)"
if [[ -n "${CUDA_HOME:-}" ]]; then
  CUDA_ROOT="$CUDA_HOME"
else
  CUDA_ROOT=/usr/local/cuda
fi
CUDA_LIB_DIR="$CUDA_ROOT/lib64"

echo "== [1/4] Linalg -> GPU launches + kernel outlining =="
mlir-opt "$INPUT_MLIR" \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  -o "$WORK_DIR/01_gpu_launch.mlir"

echo "== [2/4] GPU module -> NVVM / embedded cubin (chip=$CUBIN_CHIP) =="
if ! mlir-opt "$WORK_DIR/01_gpu_launch.mlir" \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=$CUBIN_CHIP cubin-triple=$CUBIN_TRIPLE" \
  -o "$WORK_DIR/02_gpu_binary.mlir" 2>"$WORK_DIR/02_nvvm.err"; then
  echo "error: gpu-lower-to-nvvm-pipeline failed (see $WORK_DIR/02_nvvm.err)" >&2
  echo "hint: rebuild LLVM with NVPTX or use an MLIR build that registers nvptx64-nvidia-cuda." >&2
  exit 1
fi

echo "== [3/4] Host side -> LLVM dialect =="
mlir-opt "$WORK_DIR/02_gpu_binary.mlir" \
  --convert-gpu-to-cf \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-math-to-llvm \
  --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o "$WORK_DIR/03_host_llvm.mlir"

mlir-translate "$WORK_DIR/03_host_llvm.mlir" --mlir-to-llvmir -o "$WORK_DIR/04_host.ll"

echo "== [4/4] Link host ELF =="
clang++ -O2 "$WORK_DIR/04_host.ll" -o "$OUT_BIN" \
  -L"$MLIR_LIB_DIR" -L"$CUDA_LIB_DIR" \
  -lmlir_runner_utils -lmlir_c_runner_utils \
  -lcudart -lcuda \
  -Wl,-rpath,"$MLIR_LIB_DIR" -Wl,-rpath,"$CUDA_LIB_DIR"

echo
echo "Built: $OUT_BIN"
echo "Run:   $OUT_BIN"

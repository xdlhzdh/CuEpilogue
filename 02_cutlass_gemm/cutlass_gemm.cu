// Stage 2 - CUTLASS-backed fused Device-level GEMM.
//
// Replaces the hand-rolled tiling of stage 1 with CUTLASS's
// cutlass::gemm::device::Gemm, explicitly configuring ThreadblockShape /
// WarpShape / InstructionShape so the compiler emits Volta Tensor Core
// `mma.sync.aligned.m8n8k4` instructions (verified statically in
// verify_mma_ptx.sh) instead of scalar FMA loops.
#include "cutlass_gemm.cuh"
#include "cuda_utils.cuh"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

namespace cu_epilogue {

namespace {

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = float;

// Sm70 (Volta) Tensor Core HMMA shape: mma.sync.aligned.m8n8k4.
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;

// Volta has no cp.async / multi-stage software pipelining (that is an
// Ampere+ feature) so the pipeline depth must stay at 2.
constexpr int kStages = 2;

using CutlassGemmSm70 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages>;

__global__ void ConvertF32ToF16Kernel(const float *__restrict__ src,
                                      cutlass::half_t *__restrict__ dst,
                                      size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<cutlass::half_t>(src[i]);
}

void ConvertF32ToF16(const float *src, cutlass::half_t *dst, size_t n,
                     cudaStream_t stream) {
  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  ConvertF32ToF16Kernel<<<grid, block, 0, stream>>>(src, dst, n);
}

} // namespace

bool CutlassGemmFp16(int M, int N, int K, float alpha, const float *A_fp32,
                     const float *B_fp32, float beta, float *C_fp32,
                     cudaStream_t stream) {
  cutlass::half_t *A_fp16 = nullptr, *B_fp16 = nullptr;
  CUDA_CHECK(cudaMalloc(&A_fp16, static_cast<size_t>(M) * K * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&B_fp16, static_cast<size_t>(K) * N * sizeof(cutlass::half_t)));
  ConvertF32ToF16(A_fp32, A_fp16, static_cast<size_t>(M) * K, stream);
  ConvertF32ToF16(B_fp32, B_fp16, static_cast<size_t>(K) * N, stream);

  CutlassGemmSm70 gemm_op;
  CutlassGemmSm70::Arguments args(
      {M, N, K},
      {A_fp16, K},   // TensorRef A, leading dim = K (row-major MxK)
      {B_fp16, N},   // TensorRef B, leading dim = N (row-major KxN)
      {C_fp32, N},   // TensorRef C, leading dim = N (row-major MxN)
      {C_fp32, N},   // TensorRef D (output), same as C for in-place update
      {alpha, beta});

  size_t workspace_size = CutlassGemmSm70::get_workspace_size(args);
  void *workspace = nullptr;
  if (workspace_size > 0) CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

  cutlass::Status status = gemm_op.can_implement(args);
  bool ok = (status == cutlass::Status::kSuccess);
  if (ok) {
    status = gemm_op.initialize(args, workspace, stream);
    ok = (status == cutlass::Status::kSuccess);
  }
  if (ok) {
    status = gemm_op(stream);
    ok = (status == cutlass::Status::kSuccess);
  }

  if (workspace) cudaFree(workspace);
  cudaFree(A_fp16);
  cudaFree(B_fp16);
  return ok;
}

} // namespace cu_epilogue

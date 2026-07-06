// Stage 3 - fused CUTLASS GEMM + Inline-PTX Fast-GELU epilogue.
// See fused_gemm_gelu.cuh for the high-level description; this mirrors
// stage 2's device::Gemm configuration exactly except for the
// EpilogueOutputOp, which is swapped from LinearCombination to
// FastGeluLinearCombination (fast_gelu_epilogue_op.cuh) so GELU is fused
// directly into the epilogue's register-resident accumulator write-back.
#include "fused_gemm_gelu.cuh"
#include "fast_gelu_epilogue_op.cuh"
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

using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
constexpr int kStages = 2; // Sm70 has no cp.async multi-stage pipelining.

using FastGeluEpilogueOp = FastGeluLinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator,
    ElementAccumulator>;

using FusedGemmGeluSm70 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    FastGeluEpilogueOp,
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

bool FusedGemmGeluFp16(int M, int N, int K, float alpha, const float *A_fp32,
                       const float *B_fp32, float beta, float *C_fp32,
                       cudaStream_t stream) {
  cutlass::half_t *A_fp16 = nullptr, *B_fp16 = nullptr;
  CUDA_CHECK(cudaMalloc(&A_fp16, static_cast<size_t>(M) * K * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&B_fp16, static_cast<size_t>(K) * N * sizeof(cutlass::half_t)));
  ConvertF32ToF16(A_fp32, A_fp16, static_cast<size_t>(M) * K, stream);
  ConvertF32ToF16(B_fp32, B_fp16, static_cast<size_t>(K) * N, stream);

  FusedGemmGeluSm70 gemm_op;
  FusedGemmGeluSm70::Arguments args(
      {M, N, K},
      {A_fp16, K},
      {B_fp16, N},
      {C_fp32, N},
      {C_fp32, N},
      {alpha, beta});

  size_t workspace_size = FusedGemmGeluSm70::get_workspace_size(args);
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

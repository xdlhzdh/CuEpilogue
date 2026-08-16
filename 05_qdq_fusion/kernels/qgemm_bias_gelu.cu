#include "qgemm_bias_gelu.cuh"
#include "cuda_utils.cuh"
#include "qgemm_epilogue_op.cuh"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include <cstdint>

// Volta has no INT8 Tensor Cores. SIMT INT8 GEMM (dp4a via InstructionShape
// <1,1,4>) writes INT8 D through AffineGeluQuantEpilogue: accumulator dequant
// (sa*sb and zp correction) + Fast-GELU + output Q.

namespace cu_epilogue {
namespace {

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int8_t;
using ElementAccumulator = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using ProbeGemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm50, ThreadblockShape, WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationClamp<ElementC, 1,
                                                      ElementAccumulator, float>,
    Swizzle, 2>;

using ThreadMap =
    typename ProbeGemm::GemmKernel::Epilogue::OutputTileIterator::ThreadMap;

using OutputOp =
    AffineGeluQuantEpilogue<ElementC, 1, ElementAccumulator, float, ThreadMap>;

using Int8GemmSimt = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm50, ThreadblockShape, WarpShape,
    InstructionShape, OutputOp, Swizzle, 2>;

__global__ void RowSumKernel(const int8_t *__restrict__ A,
                             int32_t *__restrict__ sumA, int M, int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M)
    return;
  int32_t s = 0;
  for (int k = 0; k < K; ++k)
    s += static_cast<int32_t>(A[m * K + k]);
  sumA[m] = s;
}

__global__ void ColSumKernel(const int8_t *__restrict__ B,
                             int32_t *__restrict__ sumB, int K, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  int32_t s = 0;
  for (int k = 0; k < K; ++k)
    s += static_cast<int32_t>(B[k * N + n]);
  sumB[n] = s;
}

} // namespace

bool QgemmBiasGeluInt8(int M, int N, int K, const int8_t *A, const int8_t *B,
                       const float *bias, int8_t *D, float scale_a, int32_t zp_a,
                       float scale_b, int32_t zp_b, float scale_d, int32_t zp_d,
                       cudaStream_t stream) {
  int32_t *sumA = nullptr;
  int32_t *sumB = nullptr;
  CUDA_CHECK(cudaMalloc(&sumA, static_cast<size_t>(M) * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&sumB, static_cast<size_t>(N) * sizeof(int32_t)));

  int threads = 256;
  RowSumKernel<<<(M + threads - 1) / threads, threads, 0, stream>>>(A, sumA, M,
                                                                    K);
  ColSumKernel<<<(N + threads - 1) / threads, threads, 0, stream>>>(B, sumB, K,
                                                                    N);

  typename OutputOp::Params epi;
  epi.alpha = scale_a * scale_b;
  epi.sumA = sumA;
  epi.sumB = sumB;
  epi.bias = bias;
  epi.zp_a = zp_a;
  epi.zp_b = zp_b;
  epi.K = K;
  epi.M = M;
  epi.N = N;
  epi.tile_m = ThreadblockShape::kM;
  epi.tile_n = ThreadblockShape::kN;
  epi.inv_scale_d = 1.0f / scale_d;
  epi.zp_d = static_cast<float>(zp_d);

  Int8GemmSimt gemm_op;
  typename Int8GemmSimt::Arguments args({M, N, K}, {A, K}, {B, N}, {D, N},
                                        {D, N}, epi);
  size_t workspace_size = Int8GemmSimt::get_workspace_size(args);
  void *workspace = nullptr;
  if (workspace_size > 0)
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

  cutlass::Status status = gemm_op.can_implement(args);
  bool ok = status == cutlass::Status::kSuccess;
  if (ok) {
    status = gemm_op.initialize(args, workspace, stream);
    ok = status == cutlass::Status::kSuccess;
  }
  if (ok) {
    status = gemm_op(stream);
    ok = status == cutlass::Status::kSuccess;
  }

  if (workspace)
    cudaFree(workspace);
  cudaFree(sumA);
  cudaFree(sumB);
  return ok;
}

} // namespace cu_epilogue

#include "qgemm_bias_gelu.cuh"
#include "cuda_utils.cuh"
#include "fast_gelu_ptx.cuh"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstdint>

// Volta (sm_70) has no INT8 Tensor Cores. Turing+ would use
// OpClassTensorOp / InstructionShape<8,8,16> (mma.sync.aligned.m8n8k16).
// On the project default arch we use CUTLASS SIMT INT8 GEMM (int32 acc),
// then a fused CUDA epilogue: affine zp correction + bias + FastGELU + quantize.

namespace cu_epilogue {
namespace {

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementAccumulator = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using Int8GemmSimt = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator,
                                                 ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

__global__ void RowSumKernel(const int8_t *__restrict__ A, int32_t *__restrict__ sumA,
                             int M, int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M)
    return;
  int32_t s = 0;
  for (int k = 0; k < K; ++k)
    s += static_cast<int32_t>(A[m * K + k]);
  sumA[m] = s;
}

__global__ void ColSumKernel(const int8_t *__restrict__ B, int32_t *__restrict__ sumB,
                             int K, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  int32_t s = 0;
  for (int k = 0; k < K; ++k)
    s += static_cast<int32_t>(B[k * N + n]);
  sumB[n] = s;
}

__global__ void AffineBiasGeluQuantKernel(
    const int32_t *__restrict__ acc, const float *__restrict__ bias,
    const int32_t *__restrict__ sumA, const int32_t *__restrict__ sumB,
    int8_t *__restrict__ D, int M, int N, int K, float scale_a, int32_t zp_a,
    float scale_b, int32_t zp_b, float scale_d, int32_t zp_d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total)
    return;
  int m = idx / N;
  int n = idx % N;
  int32_t raw = acc[idx];
  int32_t corr = raw - zp_a * sumB[n] - zp_b * sumA[m] + zp_a * zp_b * K;
  float y = scale_a * scale_b * static_cast<float>(corr) + bias[n];
  FastGeluPTX gelu;
  y = gelu(y);
  float q = y / scale_d + static_cast<float>(zp_d);
  int qi = static_cast<int>(nearbyintf(q));
  if (qi > 127)
    qi = 127;
  if (qi < -128)
    qi = -128;
  D[idx] = static_cast<int8_t>(qi);
}

} // namespace

bool QgemmBiasGeluInt8(int M, int N, int K, const int8_t *A, const int8_t *B,
                       const float *bias, int8_t *D, float scale_a, int32_t zp_a,
                       float scale_b, int32_t zp_b, float scale_d, int32_t zp_d,
                       cudaStream_t stream) {
  int32_t *acc = nullptr;
  int32_t *sumA = nullptr;
  int32_t *sumB = nullptr;
  CUDA_CHECK(cudaMalloc(&acc, static_cast<size_t>(M) * N * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&sumA, static_cast<size_t>(M) * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&sumB, static_cast<size_t>(N) * sizeof(int32_t)));

  int threads = 256;
  RowSumKernel<<<(M + threads - 1) / threads, threads, 0, stream>>>(A, sumA, M,
                                                                    K);
  ColSumKernel<<<(N + threads - 1) / threads, threads, 0, stream>>>(B, sumB, K,
                                                                    N);

  Int8GemmSimt gemm_op;
  Int8GemmSimt::Arguments args({M, N, K}, {A, K}, {B, N}, {acc, N}, {acc, N},
                               {1, 0});
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

  if (ok) {
    int total = M * N;
    AffineBiasGeluQuantKernel<<<(total + threads - 1) / threads, threads, 0,
                                stream>>>(acc, bias, sumA, sumB, D, M, N, K,
                                          scale_a, zp_a, scale_b, zp_b, scale_d,
                                          zp_d);
  }

  if (workspace)
    cudaFree(workspace);
  cudaFree(acc);
  cudaFree(sumA);
  cudaFree(sumB);
  return ok;
}

} // namespace cu_epilogue

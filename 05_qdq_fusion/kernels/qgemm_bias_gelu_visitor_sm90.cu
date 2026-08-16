// Stage 5 visitor path - CUTLASS 3.x Sm90 INT8 GEMM + LinCombEltAct
// (Fast-GELU + affine quantize). Patterned on
// 03_fastgelu_epilogue/fused_gemm_gelu_visitor_sm90.cu.
#include "qgemm_bias_gelu_visitor_sm90.cuh"
#include "cuda_utils.cuh"
#include "fast_gelu_quant.cuh"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

using namespace cute;

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace cu_epilogue {
namespace {

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::RowMajor;
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
using ElementD = int8_t;
using LayoutD = cutlass::layout::RowMajor;
using ElementAccumulator = int32_t;
using ElementCompute = float;
using ElementScalar = float;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 4;
constexpr int AlignmentD = 16;

using TileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using FusionOperation =
    cutlass::epilogue::fusion::LinCombEltAct<FastGeluQuant, ElementD,
                                             ElementCompute, ElementC,
                                             ElementScalar>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape_MNK,
        ClusterShape_MNK, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD, EpilogueSchedule,
        FusionOperation>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA,
        AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

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

// C[m,n] = bias[n] + sa*sb * (-za*sumB[n] - zb*sumA[m] + za*zb*K)
__global__ void FillAffineCKernel(
    float *__restrict__ C, const float *__restrict__ bias,
    const int32_t *__restrict__ sumA, const int32_t *__restrict__ sumB, int M,
    int N, int K, float scale_a, int32_t zp_a, float scale_b, int32_t zp_b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N)
    return;
  int m = idx / N;
  int n = idx % N;
  float sab = scale_a * scale_b;
  float corr = sab * static_cast<float>(-zp_a * sumB[n] - zp_b * sumA[m] +
                                        zp_a * zp_b * K);
  C[idx] = corr + bias[n];
}

} // namespace

bool QgemmBiasGeluInt8VisitorSm90(int M, int N, int K, const int8_t *A,
                                  const int8_t *B, const float *bias, int8_t *D,
                                  float scale_a, int32_t zp_a, float scale_b,
                                  int32_t zp_b, float scale_d, int32_t zp_d,
                                  cudaStream_t stream) {
  int32_t *sumA = nullptr;
  int32_t *sumB = nullptr;
  float *C = nullptr;
  CUDA_CHECK(cudaMalloc(&sumA, static_cast<size_t>(M) * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&sumB, static_cast<size_t>(N) * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&C, static_cast<size_t>(M) * N * sizeof(float)));

  int threads = 256;
  RowSumKernel<<<(M + threads - 1) / threads, threads, 0, stream>>>(A, sumA, M,
                                                                    K);
  ColSumKernel<<<(N + threads - 1) / threads, threads, 0, stream>>>(B, sumB, K,
                                                                    N);
  int total = M * N;
  FillAffineCKernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
      C, bias, sumA, sumB, M, N, K, scale_a, zp_a, scale_b, zp_b);

  int const L = 1;
  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, L},
      {A, stride_A, B, stride_B},
      {{}, C, stride_C, D, stride_D},
  };
  arguments.epilogue.thread.alpha = scale_a * scale_b;
  arguments.epilogue.thread.beta = 1.0f;
  arguments.epilogue.thread.activation.inv_scale_d = 1.0f / scale_d;
  arguments.epilogue.thread.activation.zp_d = static_cast<float>(zp_d);

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0)
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

  cutlass::Status status = gemm_op.can_implement(arguments);
  bool ok = (status == cutlass::Status::kSuccess);
  if (ok) {
    status = gemm_op.initialize(arguments, workspace, stream);
    ok = (status == cutlass::Status::kSuccess);
  }
  if (ok) {
    status = gemm_op.run(stream);
    ok = (status == cutlass::Status::kSuccess);
  }

  if (workspace)
    cudaFree(workspace);
  cudaFree(C);
  cudaFree(sumA);
  cudaFree(sumB);
  return ok;
}

} // namespace cu_epilogue

#else // !CUTLASS_ARCH_MMA_SM90_SUPPORTED

namespace cu_epilogue {

bool QgemmBiasGeluInt8VisitorSm90(int M, int N, int K, const int8_t *A,
                                  const int8_t *B, const float *bias, int8_t *D,
                                  float scale_a, int32_t zp_a, float scale_b,
                                  int32_t zp_b, float scale_d, int32_t zp_d,
                                  cudaStream_t stream) {
  (void)M;
  (void)N;
  (void)K;
  (void)A;
  (void)B;
  (void)bias;
  (void)D;
  (void)scale_a;
  (void)zp_a;
  (void)scale_b;
  (void)zp_b;
  (void)scale_d;
  (void)zp_d;
  (void)stream;
  return false;
}

} // namespace cu_epilogue

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

// Stage 3 visitor path - CUTLASS 3.x Sm90 fused GEMM + Fast-GELU.
// Patterned on examples/49_hopper_gemm_with_collective_builder and the
// LinCombEltAct unit tests under test/unit/gemm/device/sm90_gemm_*_bias_elementwise.cu.
#include "fused_gemm_gelu_visitor_sm90.cuh"
#include "cuda_utils.cuh"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

// CUTLASS 3.x epilogue fusion headers (operations.hpp / CollectiveBuilder)
// use unqualified sizeof_bits_v / is_base_of_v from the cute namespace.
using namespace cute;

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

#include "fast_gelu_activation.cuh"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace cu_epilogue {
namespace {

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
using ElementD = float;
using LayoutD = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementScalar = float;

constexpr int AlignmentA = 8; // 128-bit / sizeof(half)
constexpr int AlignmentB = 8;
constexpr int AlignmentC = 4; // 128-bit / sizeof(float)
constexpr int AlignmentD = 4;

using TileShape_MNK = Shape<_128, _128, _64>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using FusionOperation =
    cutlass::epilogue::fusion::LinCombEltAct<FastGelu, ElementD, ElementCompute,
                                             ElementC, ElementScalar>;

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

bool FusedGemmGeluVisitorSm90(int M, int N, int K, float alpha,
                              const float *A_fp32, const float *B_fp32,
                              float beta, float *C_fp32, cudaStream_t stream) {
  cutlass::half_t *A_fp16 = nullptr, *B_fp16 = nullptr;
  CUDA_CHECK(cudaMalloc(&A_fp16,
                        static_cast<size_t>(M) * K * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&B_fp16,
                        static_cast<size_t>(K) * N * sizeof(cutlass::half_t)));
  ConvertF32ToF16(A_fp32, A_fp16, static_cast<size_t>(M) * K, stream);
  ConvertF32ToF16(B_fp32, B_fp16, static_cast<size_t>(K) * N, stream);

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
      {A_fp16, stride_A, B_fp16, stride_B},
      {{}, C_fp32, stride_C, C_fp32, stride_D},
  };
  arguments.epilogue.thread.alpha = alpha;
  arguments.epilogue.thread.beta = beta;

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0) CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

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

  if (workspace) cudaFree(workspace);
  cudaFree(A_fp16);
  cudaFree(B_fp16);
  return ok;
}

} // namespace cu_epilogue

#else // !CUTLASS_ARCH_MMA_SM90_SUPPORTED

namespace cu_epilogue {

bool FusedGemmGeluVisitorSm90(int M, int N, int K, float alpha,
                              const float *A_fp32, const float *B_fp32,
                              float beta, float *C_fp32, cudaStream_t stream) {
  (void)M;
  (void)N;
  (void)K;
  (void)alpha;
  (void)A_fp32;
  (void)B_fp32;
  (void)beta;
  (void)C_fp32;
  (void)stream;
  return false;
}

} // namespace cu_epilogue

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

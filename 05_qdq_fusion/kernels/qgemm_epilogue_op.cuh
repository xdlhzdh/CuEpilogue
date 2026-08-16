#pragma once
// CUTLASS SIMT EpilogueOutputOp for fused QGEMM:
//   D = Q(FastGELU(sa*sb * (acc - za*sumB[n] - zb*sumA[m] + za*zb*K) + bias[n]))
//
// Acc dequant (the DQ(A)@DQ(B) identity) and output Q live here, matching
// stage 3's LinearCombination-shaped functor. Affine zp needs (m,n); those
// are recovered from the SIMT output thread map + identity swizzle.
#include "fast_gelu_ptx.cuh"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cstdint>

namespace cu_epilogue {

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_, typename ThreadMap_>
class AffineGeluQuantEpilogue {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementOutput_;
  using ThreadMap = ThreadMap_;

  static int const kCount = Count;
  static bool const kIsHeavy = true;
  static cutlass::FloatRoundStyle const kRound =
      cutlass::FloatRoundStyle::round_to_nearest;

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentSource = cutlass::Array<ElementSource, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

  static int const kFragAccesses =
      ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
      ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
      ThreadMap::kElementsPerAccess;

  struct Params {
    ElementCompute alpha = ElementCompute(1);
    int32_t const *sumA = nullptr;
    int32_t const *sumB = nullptr;
    float const *bias = nullptr;
    int32_t zp_a = 0;
    int32_t zp_b = 0;
    int32_t K = 0;
    int M = 0;
    int N = 0;
    int tile_m = 128;
    int tile_n = 128;
    float inv_scale_d = 1.0f;
    float zp_d = 0.0f;
  };

private:
  Params params_;
  FastGeluPTX gelu_;
  mutable int call_idx_;

  CUTLASS_DEVICE
  cutlass::MatrixCoord threadStartAfterIters(int iters) const {
    cutlass::gemm::GemmCoord tb =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>::
            get_tile_offset(/*log_tile=*/0);
    cutlass::MatrixCoord thread_off = ThreadMap::initial_offset(threadIdx.x);
    int row = tb.m() * params_.tile_m + thread_off.row();
    int col = tb.n() * params_.tile_n + thread_off.column();

    int state0 = 0, state1 = 0, state2 = 0;
    for (int t = 0; t < iters; ++t) {
      ++state0;
      row += ThreadMap::Shape::kRow;
      if (state0 == ThreadMap::Count::kRow) {
        state0 = 0;
        ++state1;
        row += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow *
               ThreadMap::Count::kRow;
        if (state1 == ThreadMap::Count::kGroup) {
          state1 = 0;
          ++state2;
          row += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                 ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
          if (state2 == ThreadMap::Count::kCluster) {
            state2 = 0;
            row += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
                   ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
          }
        }
      }
    }
    return cutlass::MatrixCoord(row, col);
  }

  CUTLASS_DEVICE
  void coordsForCall(int call, int &m, int &n) const {
    int iter = call / kFragAccesses;
    int i = call % kFragAccesses;
    cutlass::MatrixCoord start = threadStartAfterIters(iter);

    int column = i % ThreadMap::Iterations::kColumn;
    int frag_row = i / ThreadMap::Iterations::kColumn;
    int row = frag_row % ThreadMap::Iterations::kRow;
    int group = (frag_row / ThreadMap::Iterations::kRow) %
                ThreadMap::Iterations::kGroup;
    int cluster = frag_row / (ThreadMap::Iterations::kRow *
                              ThreadMap::Iterations::kGroup);
    int row_offset = row * ThreadMap::Delta::kRow +
                     group * ThreadMap::Delta::kGroup +
                     cluster * ThreadMap::Delta::kCluster;
    int col_offset = column * ThreadMap::Delta::kColumn;
    m = start.row() + row_offset;
    n = start.column() + col_offset;
  }

  CUTLASS_DEVICE
  ElementOutput applyOne(ElementAccumulator acc) const {
    int m = 0, n = 0;
    coordsForCall(call_idx_++, m, n);
    float y = 0.0f;
    if (m >= 0 && n >= 0 && m < params_.M && n < params_.N) {
      int32_t raw = static_cast<int32_t>(acc);
      int32_t corr = raw - params_.zp_a * params_.sumB[n] -
                     params_.zp_b * params_.sumA[m] +
                     params_.zp_a * params_.zp_b * params_.K;
      y = static_cast<float>(params_.alpha) * static_cast<float>(corr) +
          params_.bias[n];
      y = gelu_(y);
    }
    float q = y * params_.inv_scale_d + params_.zp_d;
    int qi = static_cast<int>(nearbyintf(q));
    if (qi > 127)
      qi = 127;
    if (qi < -128)
      qi = -128;
    return static_cast<ElementOutput>(qi);
  }

public:
  CUTLASS_HOST_DEVICE
  explicit AffineGeluQuantEpilogue(Params const &params)
      : params_(params), call_idx_(0) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return false; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int /*k_partition*/, int /*k_partition_count*/) {}

  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const & /*source*/) const {
    return (*this)(accumulator);
  }

  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i)
      out[i] = applyOne(accumulator[i]);
    return out;
  }
};

} // namespace cu_epilogue

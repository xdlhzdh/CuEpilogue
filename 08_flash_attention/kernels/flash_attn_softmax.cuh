#pragma once
// Register-level pieces of the fused attention mainloop that operate on
// WGMMA accumulator fragments:
//   - (row, col) views of the Sm90 GMMA C-fragment layout,
//   - sequence-length / causal masking of the score tile,
//   - the online-softmax state machine (running max, running sum, rescale),
//   - re-interpreting the P accumulator as the A operand of an RS WGMMA.
//
// Sm90 GMMA C fragment for an M=64 x N tile, per thread:
//   values ((2, 2, N/8), MMA_M, MMA_N)
//   mode (0,0): two adjacent columns, mode (0,1): row and row+8,
//   mode (0,2): column blocks of 8.
// The 4 lanes of a quad (lane % 4) own disjoint columns of the same rows, so a
// full row reduction is a thread-local reduction plus two xor-shuffles.

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

#include <cmath>

namespace cu_epilogue {
namespace fmha {

using namespace cute;

// ((2,2,V),MMA_M,MMA_N) -> ((2,MMA_M),(2,V,MMA_N)), i.e. (row, col).
template <class Layout>
CUTE_HOST_DEVICE constexpr auto convert_layout_acc_rowcol(Layout const &acc) {
  static_assert(decltype(rank(acc))::value == 3, "expected (V,MMA_M,MMA_N)");
  static_assert(decltype(rank<0>(acc))::value == 3, "expected GMMA C layout");
  return make_layout(make_layout(get<0, 1>(acc), get<1>(acc)),
                     make_layout(get<0, 0>(acc), get<0, 2>(acc), get<2>(acc)));
}

// Accumulator of a 64xN GMMA -> A-operand register fragment of a 64xKx16
// RS GMMA with K = N. For 16-bit A, ALayout_64x16 == CLayout_64x16, so each
// k-block of 16 is two consecutive 8-column blocks of the accumulator:
// ((2,2,N/8),MMA_M,1) -> ((2,2,2),MMA_M,N/16).
template <class Layout>
CUTE_HOST_DEVICE constexpr auto convert_layout_acc_Aregs(Layout const &acc) {
  static_assert(decltype(size<0, 0>(acc))::value == 2, "GMMA C layout");
  static_assert(decltype(size<0, 1>(acc))::value == 2, "GMMA C layout");
  static_assert(decltype(size<2>(acc))::value == 1,
                "one GMMA atom must span the whole N of GEMM1");
  auto l = logical_divide(get<0>(acc), Shape<Underscore, Underscore, _2>{});
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)),
                     get<1>(acc), get<2, 1>(l));
}

// Element-wise type conversion into a fresh register tensor with the same
// layout. The source must be a compact register fragment.
template <class To, class Engine, class Layout>
CUTE_DEVICE auto convert_type(Tensor<Engine, Layout> const &src) {
  using From = typename Engine::value_type;
  constexpr int N = decltype(size(src))::value;
  Tensor dst = make_tensor<To>(src.layout());
  cutlass::NumericArrayConverter<To, From, N> converter;
  *reinterpret_cast<cutlass::Array<To, N> *>(dst.data()) =
      converter(*reinterpret_cast<cutlass::Array<From, N> const *>(src.data()));
  return dst;
}

// Writes To(src) into dst. dst must view the same compact element order as
// src (e.g. the accumulator re-viewed through convert_layout_acc_Aregs).
template <class EngineS, class LayoutS, class EngineD, class LayoutD>
CUTE_DEVICE void convert_into(Tensor<EngineS, LayoutS> const &src,
                              Tensor<EngineD, LayoutD> &dst) {
  using From = typename EngineS::value_type;
  using To = typename EngineD::value_type;
  constexpr int N = decltype(size(src))::value;
  static_assert(decltype(size(dst))::value == N, "size mismatch");
  cutlass::NumericArrayConverter<To, From, N> converter;
  *reinterpret_cast<cutlass::Array<To, N> *>(dst.data()) =
      converter(*reinterpret_cast<cutlass::Array<From, N> const *>(src.data()));
}

// Compiler-only fence (no instruction): the values must be materialised in
// registers at this point. Volatile asm statements keep their relative order,
// so placing this before a warpgroup_wait<> keeps the producing arithmetic
// from being sunk below the wait.
template <int N> CUTE_DEVICE void fence_registers(float (&regs)[N]) {
  CUTE_UNROLL
  for (int i = 0; i < N; ++i) {
    asm volatile("" : "+f"(regs[i])::"memory");
  }
}

struct MaxOp {
  CUTE_DEVICE float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct SumOp {
  CUTE_DEVICE float operator()(float a, float b) const { return a + b; }
};

template <class Op> CUTE_DEVICE float quad_allreduce(float v, Op op) {
  v = op(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = op(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return v;
}

// Sets out-of-range scores to -inf. A key column is valid when
//   col < seqlen_k                         (tail of the last K block), and
//   col <= row + causal_diag  if causal    (bottom-right aligned diagonal,
//                                           causal_diag = seqlen_k - seqlen_q).
// tScS is the identity tensor partitioned exactly like tSrS; row_offset /
// col_offset move its tile-local coordinates into sequence coordinates.
template <bool IsCausal, class TensorS, class TensorC>
CUTE_DEVICE void apply_mask(TensorS &tSrS, TensorC const &tScS, int row_offset,
                            int col_offset, int seqlen_k, int causal_diag) {
  Tensor S = make_tensor(tSrS.data(), convert_layout_acc_rowcol(tSrS.layout()));
  Tensor C = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
  CUTE_UNROLL
  for (int r = 0; r < size<0>(S); ++r) {
    int const row = row_offset + get<0>(C(r, 0));
    int col_limit = seqlen_k;
    if constexpr (IsCausal) {
      col_limit = min(col_limit, row + causal_diag + 1);
    }
    CUTE_UNROLL
    for (int c = 0; c < size<1>(S); ++c) {
      if (col_offset + get<1>(C(r, c)) >= col_limit) {
        S(r, c) = -INFINITY;
      }
    }
  }
}

// Per-thread online softmax state for the kNRows accumulator rows the thread
// owns. Scores stay in raw (unscaled) units; softmax_scale is folded into the
// exp2 argument: exp(scale * (s - m)) == exp2(scale_log2 * s - scale_log2 * m).
template <int kNRows> struct OnlineSoftmax {
  static constexpr int kRows = kNRows;
  float row_max[kNRows];
  float row_sum[kNRows];  // thread-partial; quad-reduced only in finalize()
  float scale_log2;

  CUTE_DEVICE explicit OnlineSoftmax(float softmax_scale_log2)
      : scale_log2(softmax_scale_log2) {
    CUTE_UNROLL
    for (int r = 0; r < kNRows; ++r) {
      row_max[r] = -INFINITY;
      row_sum[r] = 0.0f;
    }
  }

  // One K/V block, minus the accum_O update: raise the running max with this
  // block's scores, decay row_sum by scale_o[r] = exp(old_max - new_max),
  // overwrite tSrS with P = exp(scale * (S - new_max)) and add P's row sums.
  // The caller applies scale_o to accum_O with rescale_o(), either right away
  // (serial schedule) or deferred until the GEMM2 still accumulating into
  // accum_O has retired (overlapped schedules).
  template <class TensorS>
  CUTE_DEVICE void online_softmax(TensorS &tSrS, float (&scale_o)[kNRows]) {
    Tensor S = make_tensor(tSrS.data(), convert_layout_acc_rowcol(tSrS.layout()));
    static_assert(decltype(size<0>(S))::value == kNRows, "row count mismatch");

    CUTE_UNROLL
    for (int r = 0; r < kNRows; ++r) {
      float block_max = row_max[r];
      CUTE_UNROLL
      for (int c = 0; c < size<1>(S); ++c) {
        block_max = fmaxf(block_max, S(r, c));
      }
      float const new_max = quad_allreduce(block_max, MaxOp{});
      // A row with every key masked so far keeps max == -inf; use 0 as the
      // exponent offset so exp2(-inf - offset) stays 0 instead of NaN.
      float const max_scaled = new_max == -INFINITY ? 0.0f : new_max * scale_log2;
      scale_o[r] = exp2f(row_max[r] * scale_log2 - max_scaled);
      row_max[r] = new_max;
      row_sum[r] *= scale_o[r];

      float block_sum = 0.0f;
      CUTE_UNROLL
      for (int c = 0; c < size<1>(S); ++c) {
        float const p = exp2f(S(r, c) * scale_log2 - max_scaled);
        S(r, c) = p;
        block_sum += p;
      }
      row_sum[r] += block_sum;
    }
  }

  template <class TensorO>
  CUTE_DEVICE static void rescale_o(TensorO &tOrO, float const (&scale_o)[kNRows]) {
    Tensor O = make_tensor(tOrO.data(), convert_layout_acc_rowcol(tOrO.layout()));
    static_assert(decltype(size<0>(O))::value == kNRows, "row count mismatch");
    CUTE_UNROLL
    for (int r = 0; r < kNRows; ++r) {
      CUTE_UNROLL
      for (int c = 0; c < size<1>(O); ++c) {
        O(r, c) *= scale_o[r];
      }
    }
  }

  // O /= row_sum and natural-log LSE = scale * max + log(sum). Fully masked
  // rows (sum == 0) produce O = 0 and LSE = +inf.
  template <class TensorO>
  CUTE_DEVICE void finalize(TensorO &tOrO, float softmax_scale,
                            float (&lse)[kNRows]) {
    Tensor O = make_tensor(tOrO.data(), convert_layout_acc_rowcol(tOrO.layout()));
    CUTE_UNROLL
    for (int r = 0; r < kNRows; ++r) {
      float const sum = quad_allreduce(row_sum[r], SumOp{});
      bool const empty = !(sum > 0.0f);
      float const inv_sum = empty ? 0.0f : 1.0f / sum;
      lse[r] = empty ? INFINITY : row_max[r] * softmax_scale + logf(sum);
      CUTE_UNROLL
      for (int c = 0; c < size<1>(O); ++c) {
        O(r, c) *= inv_sum;
      }
    }
  }
};

} // namespace fmha
} // namespace cu_epilogue

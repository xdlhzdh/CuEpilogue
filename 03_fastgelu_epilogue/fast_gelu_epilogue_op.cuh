#pragma once
// Stage 3 - CUTLASS Epilogue functor that injects FastGeluPTX.
//
// Mirrors the public interface of cutlass::epilogue::thread::LinearCombination
// (Params struct, is_source_needed(), the two operator() overloads) so it
// is a drop-in replacement for the EpilogueOutputOp template parameter of
// cutlass::gemm::device::Gemm. After computing the usual
// `alpha * accumulator + beta * source` linear combination, every element
// is passed through FastGeluPTX before being converted to the output type.
#include "fast_gelu_ptx.cuh"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace cu_epilogue {

template <
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_ = ElementOutput_,
    typename ElementCompute_ = ElementOutput_,
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class FastGeluLinearCombination {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementOutput_;

  static int const kCount = Count;
  static cutlass::FloatRoundStyle const kRound = Round;

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentSource = cutlass::Array<ElementSource, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    ElementCompute const *alpha_ptr;
    ElementCompute const *beta_ptr;

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)), beta(ElementCompute(0)),
          alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {}
  };

private:
  ElementCompute alpha_;
  ElementCompute beta_;
  FastGeluPTX gelu_;

public:
  CUTLASS_HOST_DEVICE
  explicit FastGeluLinearCombination(Params const &params) {
    alpha_ = params.alpha_ptr ? *params.alpha_ptr : params.alpha;
    beta_ = params.beta_ptr ? *params.beta_ptr : params.beta;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return beta_ != ElementCompute(0); }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int /*k_partition_count*/) {
    if (k_partition) beta_ = ElementCompute(0);
  }

  // D = FastGELU(alpha * accumulator + beta * source)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const &source) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementSource, kCount, kRound>
        source_converter;
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, kRound>
        accumulator_converter;
    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    ComputeFragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ElementCompute linear = alpha_ * converted_accumulator[i] +
                              beta_ * converted_source[i];
      result[i] = static_cast<ElementCompute>(gelu_(static_cast<float>(linear)));
    }

    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, kRound>
        destination_converter;
    return destination_converter(result);
  }

  // D = FastGELU(alpha * accumulator)   [beta == 0 fast path]
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, kRound>
        accumulator_converter;
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    ComputeFragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ElementCompute linear = alpha_ * converted_accumulator[i];
      result[i] = static_cast<ElementCompute>(gelu_(static_cast<float>(linear)));
    }

    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, kRound>
        destination_converter;
    return destination_converter(result);
  }
};

} // namespace cu_epilogue

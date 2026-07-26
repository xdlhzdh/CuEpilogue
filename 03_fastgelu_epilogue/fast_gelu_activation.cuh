#pragma once
// CUTLASS-style activation template wrapping FastGeluPTX.
//
// Matches the interface expected by cutlass::epilogue::fusion::LinCombEltAct
// (template <class> class ActivationFn) and Sm90Compute, which instantiates
// ActivationFn<Array<ElementCompute, FragmentSize>>. Scalar float path uses
// the same inline-PTX SFU Fast-GELU as the Sm70 functor epilogue.
#include "fast_gelu_ptx.cuh"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace cu_epilogue {

template <typename T>
struct FastGelu {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  T operator()(T const &value) const {
    FastGeluPTX gelu;
    return static_cast<T>(gelu(static_cast<float>(value)));
  }
};

template <>
struct FastGelu<float> {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  float operator()(float const &value) const {
    FastGeluPTX gelu;
    return gelu(value);
  }
};

template <typename T, int N>
struct FastGelu<cutlass::Array<T, N>> {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const &value) const {
    cutlass::Array<T, N> y;
    FastGelu<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(value[i]);
    }
    return y;
  }
};

} // namespace cu_epilogue

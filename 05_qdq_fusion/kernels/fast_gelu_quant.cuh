#pragma once
// Activation used by the Sm90 QGEMM visitor: Fast-GELU then affine quantize
// to int8-range float (the epilogue NumericArrayConverter writes int8).
#include "fast_gelu_ptx.cuh"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>

#include <cmath>

namespace cu_epilogue {

template <typename T>
struct FastGeluQuant {
  static const bool kIsHeavy = true;

  struct Arguments {
    float inv_scale_d = 1.0f;
    float zp_d = 0.0f;
  };

  CUTLASS_HOST_DEVICE
  T operator()(T const &value, Arguments const &args) const {
    FastGeluPTX gelu;
    float g = gelu(static_cast<float>(value));
    float q = g * args.inv_scale_d + args.zp_d;
#if defined(__CUDA_ARCH__)
    int qi = static_cast<int>(nearbyintf(q));
#else
    int qi = static_cast<int>(std::nearbyint(q));
#endif
    if (qi > 127)
      qi = 127;
    if (qi < -128)
      qi = -128;
    return static_cast<T>(qi);
  }
};

template <typename T, int N>
struct FastGeluQuant<cutlass::Array<T, N>> {
  static const bool kIsHeavy = true;
  using Arguments = typename FastGeluQuant<T>::Arguments;

  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const &value,
                                  Arguments const &args) const {
    cutlass::Array<T, N> y;
    FastGeluQuant<T> op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i)
      y[i] = op(value[i], args);
    return y;
  }
};

} // namespace cu_epilogue

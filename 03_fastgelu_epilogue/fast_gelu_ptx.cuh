#pragma once
// Stage 3 - Inline PTX Fast-GELU (spec 3.3).
//
// Algebraic approximation:  GELU(x) ~= x / (1 + 2^(-2.455492 * x))
//
// Implemented with hand-written inline PTX so NVCC cannot "unroll" it into
// a generic expf()-based sequence: ex2.approx.f32 and rcp.approx.f32 are
// hardware SFU (Special Function Unit) instructions, giving near single
// -cycle-throughput transcendental approximations at the cost of a small,
// application-acceptable amount of numerical error (see gelu_ref.hpp for
// the CPU-side error budget check).
#include <cutlass/cutlass.h>

namespace cu_epilogue {

struct FastGeluPTX {
  CUTLASS_HOST_DEVICE
  float operator()(float x) const {
#if defined(__CUDA_ARCH__)
    float res;
    const float constant = -2.455492f;
    asm volatile(
        "{ \n\t"
        " .reg .f32 t, r, e, p; \n\t" // declare PTX virtual registers
        " mul.f32 t, %1, %2; \n\t"    // t = x * constant
        " ex2.approx.f32 e, t; \n\t"  // hardware fast exponential approx (SFU)
        " add.f32 r, e, 1.0; \n\t"    // r = e + 1.0
        " rcp.approx.f32 p, r; \n\t"  // hardware fast reciprocal approx (SFU)
        " mul.f32 %0, %1, p; \n\t"    // res = x * p
        "} \n\t"
        : "=f"(res)
        : "f"(x), "f"(constant));
    return res;
#else
    // Host-side fallback (used only if this ever gets instantiated for a
    // host compilation path); mirrors the exact same algebraic formula
    // using standard math so behavior stays consistent off-device.
    const float constant = -2.455492f;
    float e = exp2f(constant * x);
    return x / (1.0f + e);
#endif
  }
};

} // namespace cu_epilogue

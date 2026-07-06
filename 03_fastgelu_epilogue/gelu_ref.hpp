#pragma once
// CPU reference implementations used to bound the numerical error
// introduced by the SFU-approximated Fast-GELU (spec 3.3 acceptance
// criterion: "数值误差在 AI 推理允许的 epsilon 范围内").
#include <cmath>

namespace cu_epilogue {

// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
inline float GeluExact(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

// Same algebraic approximation as FastGeluPTX, evaluated with standard
// library math (no inline PTX) - this isolates the *algorithm's* intrinsic
// approximation error from the *hardware instruction's* extra error
// (ex2.approx / rcp.approx are lower precision than IEEE exp2/reciprocal).
inline float GeluFastApproxRef(float x) {
  const float constant = -2.455492f;
  return x / (1.0f + std::exp2(constant * x));
}

} // namespace cu_epilogue

#pragma once
// Affine per-tensor INT8 quantize: q = clamp(round(x / scale + zp), -128, 127).
// This is the A/B quantization step that sits *outside* the fused QGEMM
// (the fused kernel takes already-quantized INT8 A/B).
#include <cmath>
#include <cstdint>
#include <vector>

namespace cu_epilogue {

inline int8_t QuantizeAffine(float x, float scale, int32_t zp) {
  float q = x / scale + static_cast<float>(zp);
  int qi = static_cast<int>(std::nearbyint(q));
  if (qi > 127)
    qi = 127;
  if (qi < -128)
    qi = -128;
  return static_cast<int8_t>(qi);
}

inline void QuantizeAffineTensor(const std::vector<float> &src,
                                 std::vector<int8_t> &dst, float scale,
                                 int32_t zp) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] = QuantizeAffine(src[i], scale, zp);
}

} // namespace cu_epilogue

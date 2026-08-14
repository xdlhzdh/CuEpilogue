#include "qgemm_bias_gelu.cuh"

#include <cstdint>

extern "C" void cutlass_qgemm_bias_gelu(
    const int8_t *A, const int8_t *B, const float *bias, int8_t *D, int64_t M,
    int64_t N, int64_t K, float scale_a, int32_t zp_a, float scale_b,
    int32_t zp_b, float scale_d, int32_t zp_d) {
  cu_epilogue::QgemmBiasGeluInt8(static_cast<int>(M), static_cast<int>(N),
                                 static_cast<int>(K), A, B, bias, D, scale_a,
                                 zp_a, scale_b, zp_b, scale_d, zp_d, nullptr);
}

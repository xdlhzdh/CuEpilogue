#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cu_epilogue {

bool QgemmBiasGeluInt8(int M, int N, int K, const int8_t *A, const int8_t *B,
                       const float *bias, int8_t *D, float scale_a, int32_t zp_a,
                       float scale_b, int32_t zp_b, float scale_d, int32_t zp_d,
                       cudaStream_t stream = nullptr);

} // namespace cu_epilogue

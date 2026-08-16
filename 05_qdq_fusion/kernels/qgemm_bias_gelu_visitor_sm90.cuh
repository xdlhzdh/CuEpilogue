#pragma once
// Stage 5 visitor path - CUTLASS 3.x Sm90 CollectiveBuilder + LinCombEltAct.
// Same math as qgemm_bias_gelu.cuh:
//   D = Q(FastGELU(sa*sb * (acc - za*sumB - zb*sumA + za*zb*K) + bias))
// INT8 Tensor Core mainloop; affine zp is folded into the C tensor that the
// epilogue adds before Fast-GELU + quantize. Requires Hopper (sm_90a) to
// compile a real kernel; device correctness needs cc >= 90.
#include <cuda_runtime.h>
#include <cstdint>

namespace cu_epilogue {

bool QgemmBiasGeluInt8VisitorSm90(int M, int N, int K, const int8_t *A,
                                  const int8_t *B, const float *bias, int8_t *D,
                                  float scale_a, int32_t zp_a, float scale_b,
                                  int32_t zp_b, float scale_d, int32_t zp_d,
                                  cudaStream_t stream = nullptr);

} // namespace cu_epilogue

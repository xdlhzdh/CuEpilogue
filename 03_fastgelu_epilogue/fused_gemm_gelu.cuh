#pragma once
// Stage 3 - fused CUTLASS GEMM + Inline-PTX Fast-GELU epilogue.
// Same problem setup as stage 2 (FP16 inputs, FP32 accumulate/output,
// Sm70 Tensor Core tiling) but D = FastGELU(alpha * A*B + beta * C)
// computed entirely inside the epilogue - no extra kernel launch and no
// intermediate tensor round-trip through global memory.
#include <cuda_runtime.h>

namespace cu_epilogue {

bool FusedGemmGeluFp16(int M, int N, int K, float alpha, const float *A_fp32,
                       const float *B_fp32, float beta, float *C_fp32,
                       cudaStream_t stream = nullptr);

} // namespace cu_epilogue

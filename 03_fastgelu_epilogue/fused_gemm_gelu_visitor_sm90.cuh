#pragma once
// Stage 3 (visitor path) - CUTLASS 3.x Sm90 CollectiveBuilder + EVT fusion.
//
// Same problem as fused_gemm_gelu.cuh (D = FastGELU(alpha * A*B + beta * C))
// but injects Fast-GELU via cutlass::epilogue::fusion::LinCombEltAct
// (Sm90 EVT / FusionCallbacks) instead of a 2.x-style EpilogueOutputOp
// functor. Requires Hopper (sm_90) at compile time; device correctness
// needs a compute-capability >= 90 GPU.
#include <cuda_runtime.h>

namespace cu_epilogue {

bool FusedGemmGeluVisitorSm90(int M, int N, int K, float alpha,
                              const float *A_fp32, const float *B_fp32,
                              float beta, float *C_fp32,
                              cudaStream_t stream = nullptr);

} // namespace cu_epilogue

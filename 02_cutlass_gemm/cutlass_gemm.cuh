#pragma once
// Stage 2 - host-side launch declaration for the CUTLASS-backed fused GEMM.
//
// Data type & layout (spec 3.2):
//   - Operands A, B: FP16 (cutlass::half_t), Row-Major.
//   - Accumulator:   FP32.
//   - Output C:      FP32, Row-Major (so it's directly comparable against
//                    the same CPU reference used by stage 1).
//
// Threadblock/Warp tiling is configured in cutlass_gemm.cu for
// cutlass::arch::Sm70 (Volta - matches the project's real target hardware,
// 2x Tesla V100). Sm70 Tensor Cores execute `mma.sync.aligned.m8n8k4`
// instructions; verify_mma_ptx.sh checks the generated PTX for these.
#include <cuda_runtime.h>

namespace cu_epilogue {

// A, B are host-provided as FP32; internally converted to FP16 before the
// CUTLASS kernel runs (a real deployment would keep data resident in FP16,
// this conversion exists purely so callers can share the same FP32 test
// harness/reference as stage 1).
//
// Returns true on success (cutlass::Status::kSuccess), false otherwise -
// callers should check this before trusting the output, since e.g. an
// unsupported problem size returns kErrorInvalidProblem without throwing.
bool CutlassGemmFp16(int M, int N, int K, float alpha, const float *A_fp32,
                     const float *B_fp32, float beta, float *C_fp32,
                     cudaStream_t stream = nullptr);

} // namespace cu_epilogue

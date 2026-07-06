// Stage 4 - runtime shim exposing the Stage 3 CUTLASS fused GEMM+GELU
// kernel as a plain `extern "C"` C ABI function, so it can be:
//   1. Compiled into a standalone .so (`libcutlass_fused_gemm_gelu_runtime.so`)
//      per spec 3.4's "动态链接库封装" requirement.
//   2. Called directly from LLVM IR lowered out of the `func.call
//      @cutlass_fused_gemm_gelu` op emitted by EmitExternalCall.cpp.
//
// ABI note: the MLIR-side call keeps `memref<?x?xf32>` operand types for
// readability/testability of the IR (see 04_compiler_integration/test).
// When that IR is *fully* lowered to LLVM for real execution on the target
// GPU, the final `-convert-func-to-llvm` pass must be run with
// `use-bare-ptr-memref-call-conv=1` so each memref argument arrives here as
// a bare `float*` matching this signature, rather than as an unpacked
// memref descriptor struct. This project does not execute that final
// lowering step in the compile-only sandbox it was scaffolded in (no GPU
// available there) - see docs/environment.md.
#include "fused_gemm_gelu.cuh"

#include <cstdint>

extern "C" void cutlass_fused_gemm_gelu(const float *A, const float *B,
                                        float *D, int64_t M, int64_t N,
                                        int64_t K, float alpha, float beta) {
  cu_epilogue::FusedGemmGeluFp16(static_cast<int>(M), static_cast<int>(N),
                                 static_cast<int>(K), alpha, A, B, beta, D,
                                 /*stream=*/nullptr);
}

// Stage 4 - runtime shim exposing the Stage 3 CUTLASS fused GEMM+GELU
// kernel as a plain `extern "C"` C ABI function, so it can be:
//   1. Compiled into a standalone .so (`libcutlass_fused_gemm_gelu_runtime.so`)
//      per spec 3.4's "动态链接库封装" requirement.
//   2. Called directly from LLVM IR lowered out of the `func.call
//      @cutlass_fused_gemm_gelu` op emitted by EmitExternalCall.cpp.
//
// ABI note: the MLIR-side call passes each buffer as `!llvm.ptr` (extracted
// from the memref's aligned pointer), so the default `-convert-func-to-llvm`
// lowering already matches this signature one-to-one; no memref descriptor
// or bare-pointer calling convention is involved.
//
// Memory contract: A, B and D must be device pointers (row-major, contiguous);
// this function never copies between host and device - data placement is the
// caller's (inference framework's) job.
#include "device_pointer_check.cuh"
#include "fused_gemm_gelu.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" void cutlass_fused_gemm_gelu(const float *A, const float *B,
                                        float *D, int64_t M, int64_t N,
                                        int64_t K, float alpha, float beta) {
  constexpr const char *kApi = "cutlass_fused_gemm_gelu";
  cu_epilogue::RequireDevicePointer(A, kApi, "A");
  cu_epilogue::RequireDevicePointer(B, kApi, "B");
  cu_epilogue::RequireDevicePointer(D, kApi, "D");
  if (!cu_epilogue::FusedGemmGeluFp16(static_cast<int>(M), static_cast<int>(N),
                                      static_cast<int>(K), alpha, A, B, beta, D,
                                      /*stream=*/nullptr)) {
    std::fprintf(stderr, "%s: kernel rejected M=%lld N=%lld K=%lld\n", kApi,
                 static_cast<long long>(M), static_cast<long long>(N),
                 static_cast<long long>(K));
    std::abort();
  }
}

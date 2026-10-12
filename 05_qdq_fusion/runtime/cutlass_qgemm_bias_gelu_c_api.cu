// Memory contract: A, B, bias and D must be device pointers (row-major,
// contiguous); this function never copies between host and device - data
// placement is the caller's (inference framework's) job.
#include "device_pointer_check.cuh"
#include "qgemm_bias_gelu.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" void cutlass_qgemm_bias_gelu(
    const int8_t *A, const int8_t *B, const float *bias, int8_t *D, int64_t M,
    int64_t N, int64_t K, float scale_a, int32_t zp_a, float scale_b,
    int32_t zp_b, float scale_d, int32_t zp_d) {
  constexpr const char *kApi = "cutlass_qgemm_bias_gelu";
  cu_epilogue::RequireDevicePointer(A, kApi, "A");
  cu_epilogue::RequireDevicePointer(B, kApi, "B");
  cu_epilogue::RequireDevicePointer(bias, kApi, "bias");
  cu_epilogue::RequireDevicePointer(D, kApi, "D");
  if (!cu_epilogue::QgemmBiasGeluInt8(static_cast<int>(M), static_cast<int>(N),
                                      static_cast<int>(K), A, B, bias, D,
                                      scale_a, zp_a, scale_b, zp_b, scale_d,
                                      zp_d, nullptr)) {
    std::fprintf(stderr, "%s: kernel rejected M=%lld N=%lld K=%lld\n", kApi,
                 static_cast<long long>(M), static_cast<long long>(N),
                 static_cast<long long>(K));
    std::abort();
  }
}

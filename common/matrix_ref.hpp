#pragma once
// Host-side (CPU) reference utilities shared by the correctness tests of
// stages 1-3: random matrix generation and a naive triple-loop GEMM used as
// the ground truth that GPU kernels are checked against.

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_fp16.h>

namespace cu_epilogue {

// Fills `data` with pseudo-random values in [-scale, scale], seeded
// deterministically so correctness tests are reproducible.
inline void FillRandom(std::vector<float> &data, float scale = 1.0f,
                       uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (auto &v : data) v = dist(rng);
}

// Row-major C = alpha * A(MxK) * B(KxN) + beta * C, computed on the CPU in
// double precision internally to serve as an accurate ground truth.
inline void ReferenceGemmRowMajor(int M, int N, int K, float alpha,
                                  const std::vector<float> &A,
                                  const std::vector<float> &B, float beta,
                                  std::vector<float> &C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(A[m * K + k]) *
               static_cast<double>(B[k * N + n]);
      }
      double prev = beta != 0.0f ? static_cast<double>(C[m * N + n]) : 0.0;
      C[m * N + n] = static_cast<float>(alpha * acc + beta * prev);
    }
  }
}

// Round a float to IEEE binary16 and back, matching the device-side
// `static_cast<cutlass::half_t>(float)` conversion used before GEMM.
inline float RoundToFp16(float value) {
  return __half2float(__float2half(value));
}

// Row-major GEMM reference with operands pre-quantized to FP16, so the
// CPU ground truth matches what the Tensor Core kernel actually computes.
inline void ReferenceGemmFp16InputsRowMajor(int M, int N, int K, float alpha,
                                              const std::vector<float> &A,
                                              const std::vector<float> &B,
                                              float beta,
                                              std::vector<float> &C) {
  std::vector<float> a_fp16(A.size()), b_fp16(B.size());
  for (size_t i = 0; i < A.size(); ++i) a_fp16[i] = RoundToFp16(A[i]);
  for (size_t i = 0; i < B.size(); ++i) b_fp16[i] = RoundToFp16(B[i]);
  ReferenceGemmRowMajor(M, N, K, alpha, a_fp16, b_fp16, beta, C);
}

// Returns {max_abs_error, max_relative_error} between two equally-sized
// buffers. Relative error is computed against max(|ref|, 1e-6) to avoid
// division blow-up near zero.
inline std::pair<float, float> MaxErrors(const std::vector<float> &ref,
                                         const std::vector<float> &got) {
  float max_abs = 0.0f, max_rel = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    float abs_err = std::fabs(ref[i] - got[i]);
    float denom = std::max(std::fabs(ref[i]), 1e-6f);
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, abs_err / denom);
  }
  return {max_abs, max_rel};
}

} // namespace cu_epilogue

#include "qgemm_bias_gelu.cuh"
#include "cuda_utils.cuh"
#include "gelu_ref.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace cu_epilogue;

namespace {

int8_t Quantize(float y, float scale_d, int32_t zp_d) {
  float q = y / scale_d + static_cast<float>(zp_d);
  int qi = static_cast<int>(std::nearbyint(q));
  if (qi > 127)
    qi = 127;
  if (qi < -128)
    qi = -128;
  return static_cast<int8_t>(qi);
}

void Reference(int M, int N, int K, const std::vector<int8_t> &A,
               const std::vector<int8_t> &B, const std::vector<float> &bias,
               std::vector<int8_t> &D, float sa, int32_t za, float sb,
               int32_t zb, float sd, int32_t zd) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        double a = (static_cast<int>(A[m * K + k]) - za) * static_cast<double>(sa);
        double b = (static_cast<int>(B[k * N + n]) - zb) * static_cast<double>(sb);
        acc += a * b;
      }
      float y = GeluFastApproxRef(static_cast<float>(acc) + bias[n]);
      D[m * N + n] = Quantize(y, sd, zd);
    }
  }
}

bool RunCase(int M, int N, int K) {
  std::vector<int8_t> hA(static_cast<size_t>(M) * K);
  std::vector<int8_t> hB(static_cast<size_t>(K) * N);
  std::vector<float> hBias(N, 0.1f);
  std::vector<int8_t> hD(static_cast<size_t>(M) * N, 0);
  std::vector<int8_t> hRef(hD.size(), 0);

  for (size_t i = 0; i < hA.size(); ++i)
    hA[i] = static_cast<int8_t>((static_cast<int>(i) % 17) - 8);
  for (size_t i = 0; i < hB.size(); ++i)
    hB[i] = static_cast<int8_t>((static_cast<int>(i) % 13) - 6);

  const float sa = 0.05f, sb = 0.04f, sd = 0.08f;
  const int32_t za = 1, zb = -2, zd = 3;
  Reference(M, N, K, hA, hB, hBias, hRef, sa, za, sb, zb, sd, zd);

  int8_t *dA = nullptr, *dB = nullptr, *dD = nullptr;
  float *dBias = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size()));
  CUDA_CHECK(cudaMalloc(&dB, hB.size()));
  CUDA_CHECK(cudaMalloc(&dD, hD.size()));
  CUDA_CHECK(cudaMalloc(&dBias, hBias.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), hBias.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  bool ok = QgemmBiasGeluInt8(M, N, K, dA, dB, dBias, dD, sa, za, sb, zb, sd, zd,
                              nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hD.data(), dD, hD.size(), cudaMemcpyDeviceToHost));

  int mismatches = 0;
  int maxAbs = 0;
  for (size_t i = 0; i < hD.size(); ++i) {
    int diff = std::abs(static_cast<int>(hD[i]) - static_cast<int>(hRef[i]));
    if (diff > maxAbs)
      maxAbs = diff;
    if (diff > 1)
      ++mismatches;
  }
  bool pass = ok && mismatches == 0;
  std::printf("[qgemm_bias_gelu] M=%d N=%d K=%d launched=%d max_abs=%d mismatches=%d -> %s\n",
              M, N, K, ok, maxAbs, mismatches, pass ? "PASS" : "FAIL");
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);
  cudaFree(dBias);
  return pass;
}

} // namespace

int main() {
  bool all_ok = RunCase(64, 64, 64);
  all_ok &= RunCase(32, 48, 40);
  std::printf(all_ok ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
  return all_ok ? 0 : 1;
}

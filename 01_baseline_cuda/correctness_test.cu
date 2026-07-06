// Stage 1 correctness test: runs both the naive and shared-memory tiled
// SGEMM kernels and checks their output against a CPU triple-loop
// reference implementation within a tight tolerance.
//
// NOTE: this binary needs an actual CUDA device to *run*. In a
// compile-only sandbox (no GPU attached) it will build successfully but
// exit with a CUDA runtime error when executed - that's expected. See
// docs/environment.md for what has/hasn't been verified where.
#include "sgemm_baseline.cuh"
#include "cuda_utils.cuh"
#include "matrix_ref.hpp"

#include <cstdio>
#include <vector>

using namespace cu_epilogue;

namespace {

bool RunCase(const char *name,
            void (*kernel)(int, int, int, float, const float *,
                           const float *, float, float *, cudaStream_t),
            int M, int N, int K) {
  std::vector<float> hA(static_cast<size_t>(M) * K);
  std::vector<float> hB(static_cast<size_t>(K) * N);
  std::vector<float> hC(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> hRef = hC;

  FillRandom(hA, 1.0f, /*seed=*/1);
  FillRandom(hB, 1.0f, /*seed=*/2);

  const float alpha = 1.0f, beta = 0.0f;
  ReferenceGemmRowMajor(M, N, K, alpha, hA, hB, beta, hRef);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  kernel(M, N, K, alpha, dA, dB, beta, dC, nullptr);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  auto [max_abs, max_rel] = MaxErrors(hRef, hC);
  bool ok = max_rel < 1e-2f; // fp32 accumulation tolerance for MxK up to a few hundred
  std::printf("[%s] M=%d N=%d K=%d max_abs_err=%.6g max_rel_err=%.6g -> %s\n",
             name, M, N, K, max_abs, max_rel, ok ? "PASS" : "FAIL");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return ok;
}

} // namespace

int main() {
  bool all_ok = true;
  all_ok &= RunCase("sgemm_naive", SgemmNaive, 256, 256, 256);
  all_ok &= RunCase("sgemm_tiled_smem", SgemmTiledSmem, 256, 256, 256);
  all_ok &= RunCase("sgemm_tiled_smem_nonmultiple", SgemmTiledSmem, 200, 130, 77);
  std::printf(all_ok ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
  return all_ok ? 0 : 1;
}

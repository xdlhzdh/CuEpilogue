// Stage 2 correctness test: CUTLASS FP16 GEMM vs. CPU FP32 reference.
// A looser tolerance than stage 1 is used since inputs are downcast to
// FP16 (~3 decimal digits of mantissa precision) before the Tensor Core
// kernel runs.
//
// NOTE: like stage 1, this needs a real GPU to *run* - see
// docs/environment.md.
#include "cutlass_gemm.cuh"
#include "cuda_utils.cuh"
#include "matrix_ref.hpp"

#include <cstdio>
#include <vector>

using namespace cu_epilogue;

namespace {

bool RunCase(int M, int N, int K) {
  std::vector<float> hA(static_cast<size_t>(M) * K);
  std::vector<float> hB(static_cast<size_t>(K) * N);
  std::vector<float> hC(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> hRef = hC;

  FillRandom(hA, 1.0f, /*seed=*/11);
  FillRandom(hB, 1.0f, /*seed=*/12);

  const float alpha = 1.0f, beta = 0.0f;
  ReferenceGemmRowMajor(M, N, K, alpha, hA, hB, beta, hRef);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice));

  bool launched_ok = CutlassGemmFp16(M, N, K, alpha, dA, dB, beta, dC, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

  auto [max_abs, max_rel] = MaxErrors(hRef, hC);
  bool ok = launched_ok && max_rel < 5e-2f; // FP16 input precision tolerance
  std::printf("[cutlass_gemm_fp16] M=%d N=%d K=%d launched_ok=%d "
             "max_abs_err=%.6g max_rel_err=%.6g -> %s\n",
             M, N, K, launched_ok, max_abs, max_rel, ok ? "PASS" : "FAIL");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return ok;
}

} // namespace

int main() {
  bool all_ok = true;
  all_ok &= RunCase(256, 256, 256);
  all_ok &= RunCase(512, 256, 128);
  std::printf(all_ok ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
  return all_ok ? 0 : 1;
}

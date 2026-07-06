// Stage 3 correctness test: fused CUTLASS GEMM + Inline-PTX Fast-GELU vs.
// two CPU references:
//   1. GeluExact(CPU GEMM)         - end-to-end error incl. the Fast-GELU
//                                    algebraic approximation itself.
//   2. GeluFastApproxRef(CPU GEMM) - isolates just the extra error the
//                                    ex2.approx/rcp.approx SFU hardware
//                                    instructions add on top of the same
//                                    algebraic formula (spec 3.3 epsilon
//                                    acceptance criterion).
//
// NOTE: needs a real GPU to run - see docs/environment.md.
#include "fused_gemm_gelu.cuh"
#include "gelu_ref.hpp"
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
  std::vector<float> hLinear = hC; // alpha*A*B (beta=0)

  FillRandom(hA, 1.0f, /*seed=*/21);
  FillRandom(hB, 1.0f, /*seed=*/22);

  const float alpha = 1.0f, beta = 0.0f;
  ReferenceGemmRowMajor(M, N, K, alpha, hA, hB, beta, hLinear);

  std::vector<float> refExact(hLinear.size());
  std::vector<float> refApprox(hLinear.size());
  for (size_t i = 0; i < hLinear.size(); ++i) {
    refExact[i] = GeluExact(hLinear[i]);
    refApprox[i] = GeluFastApproxRef(hLinear[i]);
  }

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice));

  bool launched_ok = FusedGemmGeluFp16(M, N, K, alpha, dA, dB, beta, dC, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

  auto [max_abs_exact, max_rel_exact] = MaxErrors(refExact, hC);
  auto [max_abs_sfu, max_rel_sfu] = MaxErrors(refApprox, hC);

  // Tolerances: 0.05 abs error is a commonly accepted epsilon for GELU
  // approximations feeding downstream inference layers; the SFU-only error
  // (vs. the same formula computed in full precision) should be much
  // tighter since it only reflects ex2.approx/rcp.approx precision loss.
  bool ok = launched_ok && max_abs_exact < 0.05f && max_abs_sfu < 0.01f;
  std::printf(
      "[fused_gemm_gelu] M=%d N=%d K=%d launched_ok=%d "
      "vs_exact_gelu(max_abs=%.6g,max_rel=%.6g) "
      "vs_sfu_precision(max_abs=%.6g,max_rel=%.6g) -> %s\n",
      M, N, K, launched_ok, max_abs_exact, max_rel_exact, max_abs_sfu,
      max_rel_sfu, ok ? "PASS" : "FAIL");

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

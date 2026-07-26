// Stage 3 visitor-path correctness test: Sm90 CollectiveBuilder +
// LinCombEltAct Fast-GELU vs the same CPU references as the functor path.
//
// Skips (exit 0) when the attached GPU has compute capability < 90, so
// V100 / Ampere hosts can still build and ctest-pass without Hopper silicon.
#include "fused_gemm_gelu_visitor_sm90.cuh"
#include "gelu_ref.hpp"
#include "cuda_utils.cuh"
#include "matrix_ref.hpp"

#include <cstdio>
#include <vector>

using namespace cu_epilogue;

namespace {

bool DeviceSupportsSm90() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                            device) != cudaSuccess)
    return false;
  if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                            device) != cudaSuccess)
    return false;
  return major >= 9;
}

bool RunCase(int M, int N, int K) {
  std::vector<float> hA(static_cast<size_t>(M) * K);
  std::vector<float> hB(static_cast<size_t>(K) * N);
  std::vector<float> hC(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> hLinear = hC;

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
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  bool launched_ok =
      FusedGemmGeluVisitorSm90(M, N, K, alpha, dA, dB, beta, dC, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  auto [max_abs_exact, max_rel_exact] = MaxErrors(refExact, hC);
  auto [max_abs_sfu, max_rel_sfu] = MaxErrors(refApprox, hC);

  bool ok = launched_ok && max_abs_exact < 0.05f && max_abs_sfu < 0.01f;
  std::printf(
      "[fused_gemm_gelu_visitor_sm90] M=%d N=%d K=%d launched_ok=%d "
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
  if (!DeviceSupportsSm90()) {
    std::printf(
        "SKIP: Sm90 visitor correctness requires Hopper (cc >= 90); "
        "current device is below that. Compile/static SFU checks still apply.\n");
    return 0;
  }

  bool all_ok = true;
  all_ok &= RunCase(256, 256, 256);
  all_ok &= RunCase(512, 256, 128);
  std::printf(all_ok ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
  return all_ok ? 0 : 1;
}

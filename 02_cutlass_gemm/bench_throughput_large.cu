// Large-matrix throughput benchmark (spec 3.2 acceptance: 阶段二计算吞吐量
// 显著优于阶段一). stage1/2_correctness_test use small 256-512 sized
// matrices chosen so a CPU triple-loop reference is fast to compute; at
// that size CUTLASS's 128x128 threadblock tiling only produces a handful
// of threadblocks, nowhere near enough to occupy the V100's 80 SMs (see
// README "性能类验收结果" for the `ncu`-measured ~1.3% SM utilization at
// 256^3). This program skips the CPU reference entirely and just times +
// profiles both kernels at a size large enough to actually saturate the
// GPU, so `ncu`/wall-clock throughput numbers mean something.
//
// Usage: bench_throughput_large [M] [N] [K]   (default 4096 4096 4096)
#include "sgemm_baseline.cuh"
#include "cutlass_gemm.cuh"
#include "cuda_utils.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cu_epilogue;

int main(int argc, char **argv) {
  const int M = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int N = argc > 2 ? std::atoi(argv[2]) : 4096;
  const int K = argc > 3 ? std::atoi(argv[3]) : 4096;
  const double kGFlop = 2.0 * static_cast<double>(M) * N * K / 1e9;
  const int kWarmup = 2, kIters = 5;

  std::vector<float> hA(static_cast<size_t>(M) * K, 1.0f);
  std::vector<float> hB(static_cast<size_t>(K) * N, 1.0f);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, static_cast<size_t>(M) * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  const float alpha = 1.0f, beta = 0.0f;
  std::printf("M=%d N=%d K=%d (%.2f GFLOP)\n", M, N, K, kGFlop);

  // --- Stage 1: hand-written shared-memory tiled SGEMM (FP32) ---
  for (int i = 0; i < kWarmup; ++i)
    SgemmTiledSmem(M, N, K, alpha, dA, dB, beta, dC, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIters; ++i)
    SgemmTiledSmem(M, N, K, alpha, dA, dB, beta, dC, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double stage1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / kIters;
  // kGFlop [1e9 FLOP] / stage_ms [1e-3 s] = 1e12 FLOP/s = TFLOP/s already.
  std::printf("[stage1] SgemmTiledSmem:   %.4f ms/call  (%.2f TFLOPS)\n",
             stage1_ms, kGFlop / stage1_ms);

  // --- Stage 2: CUTLASS FP16 Tensor Core GEMM ---
  bool launched_ok = CutlassGemmFp16(M, N, K, alpha, dA, dB, beta, dC, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  if (!launched_ok) {
    std::fprintf(stderr, "CutlassGemmFp16 failed to launch at this problem size\n");
    return 1;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIters; ++i)
    CutlassGemmFp16(M, N, K, alpha, dA, dB, beta, dC, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t3 = std::chrono::high_resolution_clock::now();
  double stage2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / kIters;
  std::printf("[stage2] CutlassGemmFp16:  %.4f ms/call  (%.2f TFLOPS)\n",
             stage2_ms, kGFlop / stage2_ms);

  std::printf("speedup (stage1_ms / stage2_ms): %.2fx\n", stage1_ms / stage2_ms);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}

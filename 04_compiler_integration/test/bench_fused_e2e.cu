// Stage 4 end-to-end latency benchmark - "fused" side (spec 3.4
// acceptance: fused external-call path vs. "基础指令展开" baseline).
//
// Times N back-to-back calls to the exact same CUTLASS Fast-GELU fused
// kernel (03_fastgelu_epilogue) that `libcutlass_fused_gemm_gelu_runtime.so`
// wraps behind the `extern "C" cutlass_fused_gemm_gelu` symbol the compiler
// pass (EmitExternalCall.cpp) emits a `func.call` to. Calling the kernel
// directly here (rather than driving it through fused-opt + mlir-runner,
// as the unfused baseline in bench_unfused.mlir does) sidesteps a real
// MLIR limitation: the bare-pointer calling convention needed so memref
// args arrive at the C API as plain `float*` does not support
// *dynamically*-shaped memref function declarations, and the production
// pass always canonicalizes call operands to memref<?x?xf32> for
// generality across problem sizes. Pattern-matching correctness of that
// pass is already covered by stage4_fuse_gemm_gelu_test; this program only
// measures the latency of the call it would emit.
//
// Prints a single number to stdout: total wall-clock milliseconds for all
// iterations (so benchmark_e2e.sh can parse it directly).
#include "fused_gemm_gelu.cuh"
#include "cuda_utils.cuh"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace cu_epilogue;

int main() {
  const int M = 128, N = 128, K = 128;
  const int kIters = 20;

  std::vector<float> hA(static_cast<size_t>(M) * K, 1.0f);
  std::vector<float> hB(static_cast<size_t>(K) * N, 1.0f);

  float *dA = nullptr, *dB = nullptr, *dD = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dD, static_cast<size_t>(M) * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Warm-up: pay the one-time CUDA context/driver init cost outside the
  // timed region, same as any realistic caller would (e.g. a long-running
  // inference server), so the benchmark reflects steady-state latency.
  FusedGemmGeluFp16(M, N, K, 1.0f, dA, dB, 0.0f, dD, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIters; ++i) {
    FusedGemmGeluFp16(M, N, K, 1.0f, dA, dB, 0.0f, dD, nullptr);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("%.6f\n", total_ms);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);
  return 0;
}

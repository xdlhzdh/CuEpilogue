#pragma once
// Shared CUDA helpers: error checking and a simple GPU event timer.
// Used across stages 1-3 so each stage's .cu files stay focused on the
// actual kernel / algorithm under test.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: '%s' failed: %s\n", __FILE__,  \
                    __LINE__, #expr, cudaGetErrorString(_err));              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace cu_epilogue {

// RAII wrapper around cudaEvent_t based GPU-side timing. Only meaningful when
// run on an actual device; unused in compile-only verification.
class GpuTimer {
public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer &) = delete;
  GpuTimer &operator=(const GpuTimer &) = delete;

  void Start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
  void Stop() { CUDA_CHECK(cudaEventRecord(stop_, 0)); }

  // Returns elapsed time in milliseconds. Must be called after Stop() and a
  // sync point (e.g. cudaEventSynchronize or cudaDeviceSynchronize).
  float ElapsedMs() {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop_));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

} // namespace cu_epilogue

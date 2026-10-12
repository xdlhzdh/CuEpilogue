// Stage 4 GPU end-to-end test. This program plays the inference framework:
// it owns device memory, uploads the inputs/weights, calls the AOT-compiled
// model (e2e_device_model.mlir, fused into @cutlass_fused_gemm_gelu) through
// MLIR's C interface with device-backed memref descriptors, and checks the
// result against a CPU reference. It also checks that the C API rejects a
// host pointer (run in a child process, since the API aborts).
#include "cuda_utils.cuh"
#include "gelu_ref.hpp"
#include "matrix_ref.hpp"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using Memref2D = StridedMemRefType<float, 2>;

extern "C" void _mlir_ciface_gemm_gelu(Memref2D *result, Memref2D *A,
                                       Memref2D *B, Memref2D *out);

namespace {

constexpr char kHostWeightFlag[] = "--host-weight";

Memref2D RowMajorView(float *data, int64_t rows, int64_t cols) {
  Memref2D view;
  view.basePtr = data;
  view.data = data;
  view.offset = 0;
  view.sizes[0] = rows;
  view.sizes[1] = cols;
  view.strides[0] = cols;
  view.strides[1] = 1;
  return view;
}

float *Upload(const std::vector<float> &host) {
  float *device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  return device;
}

bool RunCase(int M, int N, int K) {
  std::vector<float> hA(static_cast<size_t>(M) * K);
  std::vector<float> hB(static_cast<size_t>(K) * N);
  cu_epilogue::FillRandom(hA, 1.0f, 1);
  cu_epilogue::FillRandom(hB, 1.0f, 2);

  float *dA = Upload(hA);
  float *dB = Upload(hB);
  float *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, static_cast<size_t>(M) * N * sizeof(float)));

  Memref2D a = RowMajorView(dA, M, K);
  Memref2D b = RowMajorView(dB, K, N);
  Memref2D out = RowMajorView(dOut, M, N);
  Memref2D result{};
  _mlir_ciface_gemm_gelu(&result, &a, &b, &out);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hOut(static_cast<size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, hOut.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // The kernel rounds A/B to fp16 before the GEMM; mirror that on the CPU.
  std::vector<float> ref(hOut.size(), 0.0f);
  cu_epilogue::ReferenceGemmFp16InputsRowMajor(M, N, K, 1.0f, hA, hB, 0.0f,
                                                ref);
  for (float &v : ref)
    v = cu_epilogue::GeluFastApproxRef(v);
  auto [maxAbs, maxRel] = cu_epilogue::MaxErrors(ref, hOut);

  bool resultAliasesOut = result.data == dOut;
  bool pass = resultAliasesOut && maxAbs <= 1e-2f;
  std::printf("[stage4 e2e] M=%d N=%d K=%d max_abs=%.3g max_rel=%.3g "
              "result_aliases_out=%d -> %s\n",
              M, N, K, maxAbs, maxRel, resultAliasesOut, pass ? "PASS" : "FAIL");
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  return pass;
}

// Child-process body: B stays in host memory, which the C API must reject.
int RunWithHostWeight() {
  const int M = 64, N = 64, K = 64;
  std::vector<float> hA(static_cast<size_t>(M) * K, 0.5f);
  std::vector<float> hB(static_cast<size_t>(K) * N, 0.25f);
  float *dA = Upload(hA);
  float *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, static_cast<size_t>(M) * N * sizeof(float)));
  Memref2D a = RowMajorView(dA, M, K);
  Memref2D b = RowMajorView(hB.data(), K, N);
  Memref2D out = RowMajorView(dOut, M, N);
  Memref2D result{};
  _mlir_ciface_gemm_gelu(&result, &a, &b, &out);
  std::fprintf(stderr, "host weight was not rejected\n");
  return 0;
}

bool HostWeightRejected() {
  pid_t pid = fork();
  if (pid == 0) {
    char flag[sizeof(kHostWeightFlag)];
    std::memcpy(flag, kHostWeightFlag, sizeof(kHostWeightFlag));
    char *args[] = {const_cast<char *>("e2e_device_test"), flag, nullptr};
    execv("/proc/self/exe", args);
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  bool pass = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
  std::printf("[stage4 e2e] host-memory weight rejected by the C API -> %s\n",
              pass ? "PASS" : "FAIL");
  return pass;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], kHostWeightFlag) == 0)
    return RunWithHostWeight();

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("SKIP: no CUDA device visible\n");
    return 0;
  }

  bool ok = RunCase(37, 128, 64);
  ok &= RunCase(128, 128, 128);
  ok &= HostWeightRejected();
  std::printf(ok ? "ALL STAGE-4 DEVICE E2E TESTS PASSED\n"
                 : "SOME STAGE-4 DEVICE E2E TESTS FAILED\n");
  return ok ? 0 : 1;
}

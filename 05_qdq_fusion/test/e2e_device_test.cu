// Stage 5 GPU end-to-end test. This program plays the inference framework:
// it owns device memory, uploads the quantized activations / weights / bias,
// calls the AOT-compiled model (e2e_device_model.mlir, fused into
// @cutlass_qgemm_bias_gelu) through MLIR's C interface with device-backed
// memref descriptors, and checks the int8 result against a CPU reference
// (+-1 for y / scale_d vs. y * (1 / scale_d) rounding). It also checks that
// the C API rejects a host pointer (run in a child process, since the API
// aborts).
#include "cuda_utils.cuh"
#include "gelu_ref.hpp"
#include "quantize_affine.hpp"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using MemrefI8 = StridedMemRefType<int8_t, 2>;
using MemrefF32Vec = StridedMemRefType<float, 1>;

extern "C" void _mlir_ciface_qdq_gemm_bias_gelu(
    MemrefI8 *result, MemrefI8 *A, MemrefI8 *B, MemrefF32Vec *bias, float sa,
    int32_t za, float sb, int32_t zb, float sd, int32_t zd, MemrefI8 *dst);

namespace {

constexpr char kHostWeightFlag[] = "--host-weight";
// The model is compiled for static 64x64x64 shapes.
constexpr int kM = 64, kN = 64, kK = 64;
constexpr float kSa = 0.05f, kSb = 0.04f, kSd = 0.08f;
constexpr int32_t kZa = 1, kZb = -2, kZd = 3;

MemrefI8 RowMajorView(int8_t *data, int64_t rows, int64_t cols) {
  MemrefI8 view;
  view.basePtr = data;
  view.data = data;
  view.offset = 0;
  view.sizes[0] = rows;
  view.sizes[1] = cols;
  view.strides[0] = cols;
  view.strides[1] = 1;
  return view;
}

MemrefF32Vec VectorView(float *data, int64_t size) {
  MemrefF32Vec view;
  view.basePtr = data;
  view.data = data;
  view.offset = 0;
  view.sizes[0] = size;
  view.strides[0] = 1;
  return view;
}

template <typename T> T *Upload(const std::vector<T> &host) {
  T *device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return device;
}

struct HostInputs {
  std::vector<int8_t> A, B;
  std::vector<float> bias;
};

HostInputs MakeInputs() {
  std::vector<float> fA(static_cast<size_t>(kM) * kK);
  std::vector<float> fB(static_cast<size_t>(kK) * kN);
  for (size_t i = 0; i < fA.size(); ++i)
    fA[i] = static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.05f;
  for (size_t i = 0; i < fB.size(); ++i)
    fB[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.04f;
  HostInputs in;
  cu_epilogue::QuantizeAffineTensor(fA, in.A, kSa, kZa);
  cu_epilogue::QuantizeAffineTensor(fB, in.B, kSb, kZb);
  in.bias.resize(kN);
  for (int n = 0; n < kN; ++n)
    in.bias[n] = 0.1f * static_cast<float>((n % 5) - 2);
  return in;
}

std::vector<int8_t> Reference(const HostInputs &in) {
  std::vector<int8_t> ref(static_cast<size_t>(kM) * kN);
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      double acc = 0.0;
      for (int k = 0; k < kK; ++k)
        acc += (static_cast<int>(in.A[m * kK + k]) - kZa) *
               static_cast<double>(kSa) *
               (static_cast<int>(in.B[k * kN + n]) - kZb) *
               static_cast<double>(kSb);
      float y = cu_epilogue::GeluFastApproxRef(static_cast<float>(acc) +
                                               in.bias[n]);
      ref[m * kN + n] = cu_epilogue::QuantizeAffine(y, kSd, kZd);
    }
  }
  return ref;
}

bool RunCase() {
  HostInputs in = MakeInputs();
  int8_t *dA = Upload(in.A);
  int8_t *dB = Upload(in.B);
  float *dBias = Upload(in.bias);
  int8_t *dDst = nullptr;
  CUDA_CHECK(cudaMalloc(&dDst, static_cast<size_t>(kM) * kN));

  MemrefI8 a = RowMajorView(dA, kM, kK);
  MemrefI8 b = RowMajorView(dB, kK, kN);
  MemrefF32Vec bias = VectorView(dBias, kN);
  MemrefI8 dst = RowMajorView(dDst, kM, kN);
  MemrefI8 result{};
  _mlir_ciface_qdq_gemm_bias_gelu(&result, &a, &b, &bias, kSa, kZa, kSb, kZb,
                                  kSd, kZd, &dst);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int8_t> hDst(static_cast<size_t>(kM) * kN);
  CUDA_CHECK(cudaMemcpy(hDst.data(), dDst, hDst.size(), cudaMemcpyDeviceToHost));
  std::vector<int8_t> ref = Reference(in);

  int maxAbs = 0, mismatches = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    int diff = std::abs(static_cast<int>(hDst[i]) - static_cast<int>(ref[i]));
    maxAbs = std::max(maxAbs, diff);
    mismatches += diff > 1;
  }
  bool resultAliasesDst = result.data == dDst;
  bool pass = resultAliasesDst && mismatches == 0;
  std::printf("[stage5 e2e] M=%d N=%d K=%d max_abs=%d mismatches=%d "
              "result_aliases_dst=%d -> %s\n",
              kM, kN, kK, maxAbs, mismatches, resultAliasesDst,
              pass ? "PASS" : "FAIL");
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dBias);
  cudaFree(dDst);
  return pass;
}

// Child-process body: B stays in host memory, which the C API must reject.
int RunWithHostWeight() {
  HostInputs in = MakeInputs();
  int8_t *dA = Upload(in.A);
  float *dBias = Upload(in.bias);
  int8_t *dDst = nullptr;
  CUDA_CHECK(cudaMalloc(&dDst, static_cast<size_t>(kM) * kN));
  MemrefI8 a = RowMajorView(dA, kM, kK);
  MemrefI8 b = RowMajorView(in.B.data(), kK, kN);
  MemrefF32Vec bias = VectorView(dBias, kN);
  MemrefI8 dst = RowMajorView(dDst, kM, kN);
  MemrefI8 result{};
  _mlir_ciface_qdq_gemm_bias_gelu(&result, &a, &b, &bias, kSa, kZa, kSb, kZb,
                                  kSd, kZd, &dst);
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
  std::printf("[stage5 e2e] host-memory weight rejected by the C API -> %s\n",
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

  bool ok = RunCase();
  ok &= HostWeightRejected();
  std::printf(ok ? "ALL STAGE-5 DEVICE E2E TESTS PASSED\n"
                 : "SOME STAGE-5 DEVICE E2E TESTS FAILED\n");
  return ok ? 0 : 1;
}

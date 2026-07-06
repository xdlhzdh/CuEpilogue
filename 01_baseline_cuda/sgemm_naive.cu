// Stage 1 - Naive Native CUDA SGEMM (single-precision, row-major).
//
// C = alpha * A(MxK) * B(KxN) + beta * C
//
// One thread computes one output element, reading directly from global
// memory on every iteration of the K loop. This intentionally has *no*
// tiling / shared-memory reuse so that stage 1's profile (see profile.sh)
// exposes the Global Memory -> Registers data movement cost that stage
// 1b (sgemm_tiled_smem.cu) and later stages fix.
#include "sgemm_baseline.cuh"

#include <cuda_runtime.h>

namespace cu_epilogue {

__global__ void SgemmNaiveKernel(int M, int N, int K, float alpha,
                                 const float *__restrict__ A,
                                 const float *__restrict__ B, float beta,
                                 float *__restrict__ C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

void SgemmNaive(int M, int N, int K, float alpha, const float *A,
               const float *B, float beta, float *C, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  SgemmNaiveKernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace cu_epilogue

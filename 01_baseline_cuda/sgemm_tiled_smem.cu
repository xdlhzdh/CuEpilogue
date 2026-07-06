// Stage 1 - Shared-Memory Tiled Native CUDA SGEMM.
//
// Classic square-tile GEMM: each threadblock cooperatively stages a
// BLOCK_SIZE x BLOCK_SIZE tile of A and B into shared memory, then every
// thread accumulates its output element from the fast on-chip tiles instead
// of re-reading global memory K times. This amortizes global memory traffic
// by a factor of BLOCK_SIZE compared to sgemm_naive.cu, which is exactly
// the effect stage 1's nsys/ncu profile (profile.sh) is meant to quantify.
#include "sgemm_baseline.cuh"

#include <cuda_runtime.h>

namespace cu_epilogue {

namespace {
constexpr int kBlockSize = 32; // BLOCK_SIZE x BLOCK_SIZE threadblock/tile.
}

__global__ void SgemmTiledSmemKernel(int M, int N, int K, float alpha,
                                     const float *__restrict__ A,
                                     const float *__restrict__ B, float beta,
                                     float *__restrict__ C) {
  __shared__ float As[kBlockSize][kBlockSize];
  __shared__ float Bs[kBlockSize][kBlockSize];

  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * kBlockSize + ty;
  int col = blockIdx.x * kBlockSize + tx;

  float acc = 0.0f;
  int num_tiles = (K + kBlockSize - 1) / kBlockSize;
  for (int t = 0; t < num_tiles; ++t) {
    int a_col = t * kBlockSize + tx;
    int b_row = t * kBlockSize + ty;

    As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kBlockSize; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

void SgemmTiledSmem(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C,
                    cudaStream_t stream) {
  dim3 block(kBlockSize, kBlockSize);
  dim3 grid((N + kBlockSize - 1) / kBlockSize, (M + kBlockSize - 1) / kBlockSize);
  SgemmTiledSmemKernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta,
                                                   C);
}

} // namespace cu_epilogue

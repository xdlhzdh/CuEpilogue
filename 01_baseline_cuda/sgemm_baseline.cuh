#pragma once
// Shared host-side launch declarations for the two stage-1 SGEMM variants.
#include <cuda_runtime.h>

namespace cu_epilogue {

// Naive: one thread per output element, no shared-memory reuse.
void SgemmNaive(int M, int N, int K, float alpha, const float *A,
               const float *B, float beta, float *C,
               cudaStream_t stream = nullptr);

// Tiled: cooperative Shared Memory tiles of A and B, amortizing global
// memory traffic across BLOCK_SIZE threads per tile.
void SgemmTiledSmem(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C,
                    cudaStream_t stream = nullptr);

} // namespace cu_epilogue

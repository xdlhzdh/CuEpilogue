#pragma once
// Device-pointer contract check for the external C APIs (stages 4/5). The
// APIs never copy between host and device: the caller (e.g. the inference
// framework) owns data placement, so a host address here is a caller bug that
// would otherwise surface as an illegal-address fault inside the kernel.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace cu_epilogue {

inline void RequireDevicePointer(const void *ptr, const char *api,
                                 const char *arg) {
  cudaPointerAttributes attrs{};
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    std::fprintf(stderr, "%s: cudaPointerGetAttributes(%s) failed: %s\n", api,
                 arg, cudaGetErrorString(err));
    std::abort();
  }
  if (ptr && (attrs.type == cudaMemoryTypeDevice ||
              attrs.type == cudaMemoryTypeManaged))
    return;
  std::fprintf(stderr,
               "%s: argument '%s' (%p) is not device memory (cudaMemoryType "
               "%d); the caller must pass device pointers.\n",
               api, arg, ptr, static_cast<int>(attrs.type));
  std::abort();
}

} // namespace cu_epilogue

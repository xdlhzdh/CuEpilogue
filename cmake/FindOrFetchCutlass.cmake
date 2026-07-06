# Fetches NVIDIA CUTLASS (header-only, 3.x branch with CuTe) at configure time.
# Pinned to a release tag for reproducibility. CUTLASS 3.5.1 still fully supports
# SM70 (Volta) GEMM kernels alongside newer Ampere/Hopper templates.
include(FetchContent)

set(CUTLASS_GIT_TAG "v3.5.1" CACHE STRING "CUTLASS git tag/branch to fetch")

# CUTLASS's own CMake build compiles example/test binaries we don't need and which
# would slow down configuration considerably. We only need the header-only library
# target `CUTLASS`, so we fetch the source and add just the include directories
# instead of add_subdirectory()'ing the whole (heavy) CUTLASS CMake project.
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG        ${CUTLASS_GIT_TAG}
  GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  message(STATUS "Fetching CUTLASS ${CUTLASS_GIT_TAG} (header-only, this may take a while)...")
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
  FetchContent_Populate(cutlass)
endif()

add_library(cutlass_headers INTERFACE)
target_include_directories(cutlass_headers INTERFACE
  ${cutlass_SOURCE_DIR}/include
  ${cutlass_SOURCE_DIR}/tools/util/include
)
# CUTLASS 3.x device-level GEMM headers require relaxed-constexpr + extended-lambda.
target_compile_options(cutlass_headers INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

set(CUTLASS_SOURCE_DIR ${cutlass_SOURCE_DIR} CACHE INTERNAL "CUTLASS source dir")

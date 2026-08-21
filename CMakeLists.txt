cmake_minimum_required(VERSION 3.20)

# CUDA architecture must be set before project() so it applies to CMAKE_CUDA_ARCHITECTURES.
# Default target: sm_70 (Volta / Tesla V100), matching the real hardware this project
# will eventually run and be profiled on. Spec suggests "Ampere or newer"; override with
# -DCU_EPILOGUE_CUDA_ARCH=80 (Ampere), 86 (Ampere/RTX30), 89 (Ada), 90 (Hopper) as needed.
set(CU_EPILOGUE_CUDA_ARCH "70" CACHE STRING "Target CUDA compute capability (SM), e.g. 70/80/86/89/90")
set(CMAKE_CUDA_ARCHITECTURES ${CU_EPILOGUE_CUDA_ARCH})

project(CuEpilogue LANGUAGES C CXX)

option(CU_EPILOGUE_ENABLE_CUDA "Enable CUDA stages (1-3): requires nvcc" ON)
option(CU_EPILOGUE_ENABLE_MLIR "Enable MLIR stages (4-5): requires LLVM/MLIR" ON)
option(CU_EPILOGUE_ENABLE_TVM "Enable stage 6 TVM Relax BYOC tests (requires python3+tvm)" OFF)
option(CU_EPILOGUE_BUILD_TESTS "Build correctness / verification tests" ON)
# Side-by-side CUTLASS 3.x Sm90 EVT / FusionCallbacks path (stage 3 Fast-GELU
# and stage 5 INT8 QGEMM). Compiles those targets for sm_90a regardless of
# CU_EPILOGUE_CUDA_ARCH; device correctness skips on GPUs with compute
# capability < 90. Stage 4 runtime and stage 5 C ABI still use the default
# (Sm70 functor / SIMT) kernels.
option(CU_EPILOGUE_ENABLE_SM90_VISITOR "Build Sm90 CollectiveBuilder + LinCombEltAct visitor epilogue" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Needed so the stage-3 static lib can be re-linked into the stage-4 shared
# runtime (cutlass_fused_gemm_gelu_runtime.so).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CU_EPILOGUE_BUILD_TESTS)
  enable_testing()
endif()

message(STATUS "CuEpilogue: target CUDA arch = sm_${CU_EPILOGUE_CUDA_ARCH}")
message(STATUS "CuEpilogue: CUDA stages enabled = ${CU_EPILOGUE_ENABLE_CUDA}")
message(STATUS "CuEpilogue: MLIR stages enabled = ${CU_EPILOGUE_ENABLE_MLIR}")
message(STATUS "CuEpilogue: TVM stage 6 enabled = ${CU_EPILOGUE_ENABLE_TVM}")
message(STATUS "CuEpilogue: Sm90 visitor path   = ${CU_EPILOGUE_ENABLE_SM90_VISITOR}")

if(CU_EPILOGUE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(WARNING "nvcc not found - disabling CUDA stages (1-3). Run scripts/setup_env.sh to install CUDA Toolkit.")
    set(CU_EPILOGUE_ENABLE_CUDA OFF)
  endif()
endif()

if(CU_EPILOGUE_ENABLE_CUDA)
  include(${CMAKE_SOURCE_DIR}/cmake/FindOrFetchCutlass.cmake)

  add_library(cu_epilogue_common INTERFACE)
  target_include_directories(cu_epilogue_common INTERFACE
    ${CMAKE_SOURCE_DIR}/common
  )

  add_subdirectory(01_baseline_cuda)
  add_subdirectory(02_cutlass_gemm)
  add_subdirectory(03_fastgelu_epilogue)
endif()

if(CU_EPILOGUE_ENABLE_MLIR)
  add_subdirectory(04_compiler_integration)
  add_subdirectory(05_qdq_fusion)
endif()

if(CU_EPILOGUE_ENABLE_TVM)
  add_subdirectory(06_tvm_cutlass_byoc)
endif()

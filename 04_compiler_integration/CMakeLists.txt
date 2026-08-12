# Stage 4 - MLIR compiler backend integration. Builds entirely against a
# pre-installed LLVM/MLIR (find_package below), independent of whether the
# CUDA stages (1-3) are enabled - this stage needs no GPU/nvcc to build or
# to run its tests.
find_package(MLIR REQUIRED CONFIG)

message(STATUS "CuEpilogue: found MLIR ${MLIR_PACKAGE_VERSION} in ${MLIR_DIR}")
message(STATUS "CuEpilogue: found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

add_mlir_library(FusedGemmGeluPasses
  lib/FuseGemmGeluPattern.cpp
  lib/EmitExternalCall.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRMathDialect
)
target_include_directories(FusedGemmGeluPasses PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(FUSED_OPT_LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  FusedGemmGeluPasses
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
)

add_llvm_executable(fused-opt tools/fused-opt.cpp)
llvm_update_compile_flags(fused-opt)
target_link_libraries(fused-opt PRIVATE ${FUSED_OPT_LIBS})
mlir_check_all_link_libraries(fused-opt)

if(CU_EPILOGUE_BUILD_TESTS)
  enable_testing()
  add_test(
    NAME stage4_fuse_gemm_gelu_test
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/run_tests.sh $<TARGET_FILE:fused-opt>
  )
endif()

# Runtime shim (.so wrapping the Stage 3 CUTLASS kernel behind a plain C
# ABI, spec 3.4's "动态链接库封装") - only buildable when the CUDA stages
# are enabled, since it links against the fused_gemm_gelu (Stage 3) target.
if(CU_EPILOGUE_ENABLE_CUDA AND TARGET fused_gemm_gelu)
  add_library(cutlass_fused_gemm_gelu_runtime SHARED
    runtime/cutlass_fused_gemm_gelu_c_api.cu
  )
  target_link_libraries(cutlass_fused_gemm_gelu_runtime PRIVATE fused_gemm_gelu)
  set_target_properties(cutlass_fused_gemm_gelu_runtime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
else()
  message(STATUS "CuEpilogue: skipping cutlass_fused_gemm_gelu_runtime.so (CUDA stages disabled)")
endif()

# End-to-end latency benchmark (spec 3.4 acceptance: fused external-call
# path vs. "基础指令展开" baseline) - see test/bench_fused_e2e.cu and
# benchmark_e2e.sh. Only buildable when the CUDA stages are enabled, since
# it links against the stage 3 fused_gemm_gelu kernel directly.
if(CU_EPILOGUE_ENABLE_CUDA AND TARGET fused_gemm_gelu)
  add_executable(bench_fused_e2e test/bench_fused_e2e.cu)
  target_link_libraries(bench_fused_e2e PRIVATE fused_gemm_gelu)
endif()

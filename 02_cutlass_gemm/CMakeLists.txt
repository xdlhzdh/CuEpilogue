add_library(cutlass_gemm STATIC
  cutlass_gemm.cu
)
target_include_directories(cutlass_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cutlass_gemm PUBLIC cu_epilogue_common cutlass_headers)
set_target_properties(cutlass_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CU_EPILOGUE_BUILD_TESTS)
  add_executable(stage2_correctness_test correctness_test.cu)
  target_link_libraries(stage2_correctness_test PRIVATE cutlass_gemm)
  add_test(NAME stage2_correctness_test COMMAND stage2_correctness_test)
endif()

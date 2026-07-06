add_library(fused_gemm_gelu STATIC
  fused_gemm_gelu.cu
)
target_include_directories(fused_gemm_gelu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fused_gemm_gelu PUBLIC cu_epilogue_common cutlass_headers)
set_target_properties(fused_gemm_gelu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CU_EPILOGUE_BUILD_TESTS)
  add_executable(stage3_correctness_test correctness_test.cu)
  target_link_libraries(stage3_correctness_test PRIVATE fused_gemm_gelu)
  add_test(NAME stage3_correctness_test COMMAND stage3_correctness_test)
endif()

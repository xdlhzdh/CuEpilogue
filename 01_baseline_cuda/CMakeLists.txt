add_library(sgemm_baseline STATIC
  sgemm_naive.cu
  sgemm_tiled_smem.cu
)
target_include_directories(sgemm_baseline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sgemm_baseline PUBLIC cu_epilogue_common)
set_target_properties(sgemm_baseline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CU_EPILOGUE_BUILD_TESTS)
  add_executable(stage1_correctness_test correctness_test.cu)
  target_link_libraries(stage1_correctness_test PRIVATE sgemm_baseline)
  add_test(NAME stage1_correctness_test COMMAND stage1_correctness_test)
endif()

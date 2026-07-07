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

# Large-matrix throughput benchmark (spec 3.2: 阶段二吞吐量显著优于阶段一) -
# see bench_throughput_large.cu for why the small correctness-test sizes
# can't demonstrate this. Links both stage 1 and stage 2 kernels directly;
# not a ctest (it's a perf benchmark, run manually / via benchmark script).
add_executable(bench_throughput_large bench_throughput_large.cu)
target_link_libraries(bench_throughput_large PRIVATE sgemm_baseline cutlass_gemm)

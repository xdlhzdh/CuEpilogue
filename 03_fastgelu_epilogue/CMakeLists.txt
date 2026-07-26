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

# CUTLASS 3.x Sm90 visitor / EVT path (side-by-side with the Sm70 functor above).
if(CU_EPILOGUE_ENABLE_SM90_VISITOR)
  add_library(fused_gemm_gelu_visitor_sm90 STATIC
    fused_gemm_gelu_visitor_sm90.cu
  )
  target_include_directories(fused_gemm_gelu_visitor_sm90 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(fused_gemm_gelu_visitor_sm90 PUBLIC cu_epilogue_common cutlass_headers)
  # Hopper WGMMA kernels require sm_90a (sets __CUDA_ARCH_FEAT_SM90_ALL);
  # plain sm_90 compiles to a stub that only printf-aborts.
  # Separable compilation is required for GemmUniversalAdapter device link.
  set_target_properties(fused_gemm_gelu_visitor_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 90a
  )

  if(CU_EPILOGUE_BUILD_TESTS)
    add_executable(stage3_visitor_correctness_test correctness_test_visitor.cu)
    target_link_libraries(stage3_visitor_correctness_test PRIVATE fused_gemm_gelu_visitor_sm90)
    set_target_properties(stage3_visitor_correctness_test PROPERTIES CUDA_ARCHITECTURES 90a)
    add_test(NAME stage3_visitor_correctness_test COMMAND stage3_visitor_correctness_test)
  endif()
endif()

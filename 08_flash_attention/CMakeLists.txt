# Stage 8 - hand-written Sm90 FlashAttention forward.
# GEMM1 (Q*K^T) + online softmax + GEMM2 (P*V) in one CuTe mainloop, fed by
# TMA through cutlass::PipelineTmaAsync. WGMMA needs sm_90a regardless of
# CU_EPILOGUE_CUDA_ARCH; the correctness test SKIPs on cc < 90.
#
# Whole-program (non-rdc) compilation on purpose: with relocatable device code
# ptxas drops setmaxnreg and serializes wgmma ("Extern calls" in the kernel).
add_library(flash_attn_fwd_sm90 STATIC
  kernels/flash_attn_fwd_sm90.cu
)
target_include_directories(flash_attn_fwd_sm90 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_link_libraries(flash_attn_fwd_sm90 PUBLIC cu_epilogue_common cutlass_headers)
set_target_properties(flash_attn_fwd_sm90 PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES 90a
)

if(CU_EPILOGUE_BUILD_TESTS)
  add_executable(stage8_flash_attn_correctness_test kernels/correctness_test.cu)
  target_link_libraries(stage8_flash_attn_correctness_test PRIVATE flash_attn_fwd_sm90)
  set_target_properties(stage8_flash_attn_correctness_test PROPERTIES
    CUDA_ARCHITECTURES 90a
  )
  add_test(NAME stage8_flash_attn_correctness_test
    COMMAND stage8_flash_attn_correctness_test)

  add_test(NAME stage8_flash_attn_sass_check
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/verify_sass_sm90.sh ${CUTLASS_SOURCE_DIR})
endif()

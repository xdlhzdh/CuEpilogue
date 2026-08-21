# Stage 6 - TVM Relax BYOC over stage-5 CUTLASS QGEMM .so (no new CUDA kernel).
# Enabled with -DCU_EPILOGUE_ENABLE_TVM=ON. Tests SKIP when python3+tvm are missing.

set(STAGE6_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STAGE6_PYTEST ${STAGE6_DIR}/scripts/run_pytest.sh)

if(CU_EPILOGUE_BUILD_TESTS)
  enable_testing()

  # Ensure the stage-5 shared library exists when MLIR+CUDA builds are on.
  if(TARGET cutlass_qgemm_bias_gelu_runtime)
    add_custom_target(stage6_runtime_dep DEPENDS cutlass_qgemm_bias_gelu_runtime)
  endif()

  add_test(
    NAME stage6_tvm_tests
    COMMAND bash ${STAGE6_PYTEST}
            ${CMAKE_BINARY_DIR}
            ${STAGE6_DIR}
  )
  set_tests_properties(stage6_tvm_tests PROPERTIES
    WORKING_DIRECTORY ${STAGE6_DIR}
    ENVIRONMENT "CU_EPILOGUE_BUILD_DIR=${CMAKE_BINARY_DIR};PYTHONPATH=${STAGE6_DIR}/python:$ENV{PYTHONPATH}"
  )
  if(TARGET cutlass_qgemm_bias_gelu_runtime)
    set_tests_properties(stage6_tvm_tests PROPERTIES
      DEPENDS cutlass_qgemm_bias_gelu_runtime
    )
  endif()
endif()

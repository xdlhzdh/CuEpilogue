# Stage 7 - TVM DLight + MetaSchedule hybrid scheduling (Python/Relax).
# Enabled with -DCU_EPILOGUE_ENABLE_TVM=ON (same flag as stage 6).
# Tests SKIP when python3+tvm are missing; M3/M4 subsets SKIP without GPU.

set(STAGE7_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STAGE7_PYTEST ${STAGE7_DIR}/scripts/run_pytest.sh)
set(STAGE6_DIR ${CMAKE_SOURCE_DIR}/06_tvm_cutlass_byoc)

if(CU_EPILOGUE_BUILD_TESTS)
  enable_testing()

  add_test(
    NAME stage7_tvm_tests
    COMMAND bash ${STAGE7_PYTEST}
            ${CMAKE_BINARY_DIR}
            ${STAGE7_DIR}
            ${STAGE6_DIR}
  )
  set_tests_properties(stage7_tvm_tests PROPERTIES
    WORKING_DIRECTORY ${STAGE7_DIR}
    ENVIRONMENT "CU_EPILOGUE_BUILD_DIR=${CMAKE_BINARY_DIR};PYTHONPATH=${STAGE7_DIR}/python:${STAGE6_DIR}/python:$ENV{PYTHONPATH}"
  )
endif()

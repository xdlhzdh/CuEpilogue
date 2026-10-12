# Stage 5 - QDQ fusion at Linalg-on-Tensor, then OSB, then func.call.
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/CutlassQGemm/CutlassQGemmOps.td)
mlir_tablegen(CutlassQGemmOps.h.inc -gen-op-decls)
mlir_tablegen(CutlassQGemmOps.cpp.inc -gen-op-defs)
mlir_tablegen(CutlassQGemmDialect.h.inc -gen-dialect-decls -dialect=cutlass)
mlir_tablegen(CutlassQGemmDialect.cpp.inc -gen-dialect-defs -dialect=cutlass)
add_public_tablegen_target(CutlassQGemmIncGen)

add_mlir_library(CutlassQGemmPasses
  lib/CutlassQGemmDialect.cpp
  lib/CutlassQGemmOps.cpp
  lib/BufferizableOpInterfaceImpl.cpp
  lib/FuseQdqPattern.cpp
  lib/LowerToCall.cpp

  DEPENDS
  CutlassQGemmIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRLLVMDialect
)
target_include_directories(CutlassQGemmPasses PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(QDQ_OPT_LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  CutlassQGemmPasses
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
)

add_llvm_executable(qdq-opt tools/qdq-opt.cpp)
add_dependencies(qdq-opt CutlassQGemmIncGen)
llvm_update_compile_flags(qdq-opt)
target_link_libraries(qdq-opt PRIVATE ${QDQ_OPT_LIBS})
mlir_check_all_link_libraries(qdq-opt)

if(CU_EPILOGUE_BUILD_TESTS)
  enable_testing()
  add_test(
    NAME stage5_qdq_fusion_test
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/run_tests.sh $<TARGET_FILE:qdq-opt>
  )
endif()

if(CU_EPILOGUE_ENABLE_CUDA AND TARGET cu_epilogue_common)
  add_library(qgemm_bias_gelu STATIC kernels/qgemm_bias_gelu.cu)
  target_include_directories(qgemm_bias_gelu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_SOURCE_DIR}/03_fastgelu_epilogue
  )
  target_link_libraries(qgemm_bias_gelu PUBLIC cu_epilogue_common cutlass_headers)
  set_target_properties(qgemm_bias_gelu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_library(cutlass_qgemm_bias_gelu_runtime SHARED
    runtime/cutlass_qgemm_bias_gelu_c_api.cu
  )
  target_link_libraries(cutlass_qgemm_bias_gelu_runtime PRIVATE qgemm_bias_gelu)
  set_target_properties(cutlass_qgemm_bias_gelu_runtime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  if(CU_EPILOGUE_BUILD_TESTS)
    add_executable(stage5_qgemm_correctness_test kernels/correctness_test.cu)
    target_link_libraries(stage5_qgemm_correctness_test PRIVATE qgemm_bias_gelu)
    add_test(NAME stage5_qgemm_correctness_test COMMAND stage5_qgemm_correctness_test)
  endif()

  # GPU end-to-end test: AOT-compile test/e2e_device_model.mlir to an object
  # and link it into a harness that owns device memory like an inference
  # framework would. Prints SKIP when no GPU is visible at test time.
  find_program(CU_EPILOGUE_MLIR_TRANSLATE mlir-translate HINTS ${LLVM_TOOLS_BINARY_DIR})
  find_program(CU_EPILOGUE_LLC llc HINTS ${LLVM_TOOLS_BINARY_DIR})
  if(CU_EPILOGUE_BUILD_TESTS AND CU_EPILOGUE_MLIR_TRANSLATE AND CU_EPILOGUE_LLC)
    set(STAGE5_E2E_MODEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test/e2e_device_model.mlir)
    set(STAGE5_E2E_MODEL ${CMAKE_CURRENT_BINARY_DIR}/e2e_device_model)
    add_custom_command(
      OUTPUT ${STAGE5_E2E_MODEL}.o
      COMMAND qdq-opt ${STAGE5_E2E_MODEL_SRC}
        --fuse-qdq-qgemm-bias-gelu
        "--one-shot-bufferize=bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map"
        --lower-cutlass-qgemm-to-call
        --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm
        --reconcile-unrealized-casts
        -o ${STAGE5_E2E_MODEL}.llvm.mlir
      COMMAND ${CU_EPILOGUE_MLIR_TRANSLATE} --mlir-to-llvmir
        ${STAGE5_E2E_MODEL}.llvm.mlir -o ${STAGE5_E2E_MODEL}.ll
      COMMAND ${CU_EPILOGUE_LLC} -filetype=obj -relocation-model=pic
        ${STAGE5_E2E_MODEL}.ll -o ${STAGE5_E2E_MODEL}.o
      DEPENDS qdq-opt ${STAGE5_E2E_MODEL_SRC}
      COMMENT "AOT-compiling stage 5 e2e model"
      VERBATIM
    )
    set_source_files_properties(${STAGE5_E2E_MODEL}.o PROPERTIES
      EXTERNAL_OBJECT TRUE GENERATED TRUE
    )
    add_executable(stage5_device_e2e_test
      test/e2e_device_test.cu ${STAGE5_E2E_MODEL}.o
    )
    target_include_directories(stage5_device_e2e_test PRIVATE
      ${CMAKE_SOURCE_DIR}/03_fastgelu_epilogue
    )
    target_link_libraries(stage5_device_e2e_test PRIVATE
      cutlass_qgemm_bias_gelu_runtime cu_epilogue_common
    )
    add_test(NAME stage5_device_e2e_test COMMAND stage5_device_e2e_test)
  endif()

  # CUTLASS 3.x Sm90 visitor / EVT path (side-by-side with the SIMT kernel).
  # Hopper WGMMA needs sm_90a; plain sm_90 compiles to a stub. Default C ABI
  # .so stays on the SIMT kernel above.
  if(CU_EPILOGUE_ENABLE_SM90_VISITOR)
    add_library(qgemm_bias_gelu_visitor_sm90 STATIC
      kernels/qgemm_bias_gelu_visitor_sm90.cu
    )
    target_include_directories(qgemm_bias_gelu_visitor_sm90 PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels
      ${CMAKE_SOURCE_DIR}/03_fastgelu_epilogue
    )
    target_link_libraries(qgemm_bias_gelu_visitor_sm90 PUBLIC
      cu_epilogue_common cutlass_headers
    )
    set_target_properties(qgemm_bias_gelu_visitor_sm90 PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES 90a
    )

    if(CU_EPILOGUE_BUILD_TESTS)
      add_executable(stage5_visitor_correctness_test
        kernels/correctness_test_visitor.cu
      )
      target_link_libraries(stage5_visitor_correctness_test PRIVATE
        qgemm_bias_gelu_visitor_sm90
      )
      set_target_properties(stage5_visitor_correctness_test PROPERTIES
        CUDA_ARCHITECTURES 90a
      )
      add_test(NAME stage5_visitor_correctness_test
        COMMAND stage5_visitor_correctness_test)
    endif()
  endif()
endif()

#ifndef CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H
#define CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"

namespace cu_epilogue {

// A matched (matmul, activation) pair found by FuseGemmGeluPattern.cpp,
// ready to be lowered to an external call by EmitExternalCall.cpp.
struct FusedGemmGeluMatch {
  mlir::linalg::MatmulOp matmul;
  mlir::linalg::GenericOp activation;
};

// Replaces `match.matmul` + `match.activation` with a call to the external
// `@cutlass_fused_gemm_gelu` symbol (declaring it in the enclosing module
// on first use). Validates operand shapes/strides before emitting (spec
// 4.3's "IR 语义不对齐" risk: reject rather than silently miscompile).
mlir::LogicalResult emitExternalFusedCall(mlir::OpBuilder &builder,
                                          const FusedGemmGeluMatch &match);

} // namespace cu_epilogue

#endif // CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H

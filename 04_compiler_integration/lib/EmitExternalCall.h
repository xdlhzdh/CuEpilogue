#ifndef CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H
#define CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"

namespace cu_epilogue {

/// A matched (matmul, GELU-activation) pair found by FuseGemmGeluPattern.cpp,
/// ready to be lowered to an external call by EmitExternalCall.cpp.
///
/// `alpha` / `beta` are the epilogue scalars for
///   D = FastGELU(alpha * (A @ B) + beta * C)
/// extracted from the activation body when possible, otherwise normalized to
/// the identity defaults (1.0, 0.0).
struct FusedGemmGeluMatch {
  mlir::linalg::MatmulOp matmul;
  mlir::linalg::GenericOp activation;
  float alpha = 1.0f;
  float beta = 0.0f;
};

/// Replaces `match.matmul` + `match.activation` with a call to the external
/// `@cutlass_fused_gemm_gelu` symbol (declaring it in the enclosing module
/// on first use). Validates operand element types / shapes / identity layout
/// before emitting (spec 4.3: reject rather than silently miscompile).
mlir::LogicalResult emitExternalFusedCall(mlir::OpBuilder &builder,
                                          const FusedGemmGeluMatch &match);

} // namespace cu_epilogue

#endif // CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H

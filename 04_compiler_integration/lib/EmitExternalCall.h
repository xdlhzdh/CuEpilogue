#ifndef CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H
#define CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"

namespace cu_epilogue {

/// A matched (matmul, GELU-activation) pair found by FuseGemmGeluPattern.cpp,
/// ready to be lowered to an external call by EmitExternalCall.cpp.
///
/// The match always means D = FastGELU(A @ B). `zeroInit` is the
/// `linalg.fill 0.0` of the matmul's local accumulator buffer; it is erased
/// along with the matmul.
struct FusedGemmGeluMatch {
  mlir::linalg::MatmulOp matmul;
  mlir::linalg::GenericOp activation;
  mlir::linalg::FillOp zeroInit;
};

/// Replaces `match.matmul` + `match.activation` with a call to the external
/// `@cutlass_fused_gemm_gelu` symbol (declaring it in the enclosing module
/// on first use). Validates operand element types / shapes / identity layout
/// before emitting (spec 4.3: reject rather than silently miscompile).
mlir::LogicalResult emitExternalFusedCall(mlir::OpBuilder &builder,
                                          const FusedGemmGeluMatch &match);

} // namespace cu_epilogue

#endif // CU_EPILOGUE_LIB_EMIT_EXTERNAL_CALL_H

#include "EmitExternalCall.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace cu_epilogue {

namespace {
constexpr char kExternalSymbol[] = "cutlass_fused_gemm_gelu";

// The runtime C API (04_compiler_integration/runtime) speaks row-major
// contiguous FP32 buffers; this is the canonical MLIR-side signature every
// matched call site is cast to before calling.
Type CanonicalMemRefType(MLIRContext *ctx) {
  return MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                        Float32Type::get(ctx));
}

func::FuncOp getOrInsertExternalDecl(OpBuilder &builder, ModuleOp module,
                                     Location loc) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kExternalSymbol))
    return existing;

  MLIRContext *ctx = builder.getContext();
  Type memrefTy = CanonicalMemRefType(ctx);
  Type i64Ty = builder.getI64Type();
  Type f32Ty = builder.getF32Type();
  auto funcType = builder.getFunctionType(
      {memrefTy, memrefTy, memrefTy, i64Ty, i64Ty, i64Ty, f32Ty, f32Ty}, {});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcOp = func::FuncOp::create(builder, loc, kExternalSymbol, funcType);
  funcOp.setPrivate(); // declaration only - no body means "extern" on lowering
  return funcOp;
}

// Rejects memrefs the runtime C API cannot safely accept: anything other
// than a rank-2, contiguous row-major (identity layout), f32 buffer. This
// is the "Shape/Stride 校验" mitigation from spec 4.4 - we fail the pass
// with a diagnostic instead of silently emitting a call with an
// incompatible ABI.
LogicalResult validateOperand(Operation *op, Value memref,
                              const char *role) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type)
    return op->emitError() << role << " is not a memref value";
  if (type.getRank() != 2)
    return op->emitError() << role << " must be rank-2, got rank "
                           << type.getRank();
  if (!type.getElementType().isF32())
    return op->emitError() << role
                           << " must have f32 elements (runtime API is FP32)";
  if (!type.getLayout().isIdentity())
    return op->emitError()
           << role
           << " must be contiguous row-major (identity layout); got "
           << type;
  return success();
}

Value castToCanonical(OpBuilder &builder, Location loc, Value memref) {
  Type canonical = CanonicalMemRefType(builder.getContext());
  if (memref.getType() == canonical) return memref;
  return memref::CastOp::create(builder, loc, canonical, memref);
}

// Erases `matmulOutput`'s defining memref.alloc if the fusion left it with
// no remaining users - keeps the rewritten IR free of dead intermediate
// buffers instead of relying on a later canonicalization pass.
void eraseIfDeadAlloc(Value value) {
  if (!value.use_empty()) return;
  if (auto allocOp = value.getDefiningOp<memref::AllocOp>()) allocOp.erase();
}

} // namespace

LogicalResult emitExternalFusedCall(OpBuilder &builder,
                                    const FusedGemmGeluMatch &match) {
  linalg::MatmulOp matmul = match.matmul;
  linalg::GenericOp activation = match.activation;
  Location loc = matmul.getLoc();

  if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1)
    return matmul.emitError("expected linalg.matmul with 2 inputs, 1 output");
  if (activation.getInputs().size() != 1 || activation.getOutputs().size() != 1)
    return activation.emitError(
        "expected elementwise linalg.generic with 1 input, 1 output");

  Value a = matmul.getInputs()[0];
  Value b = matmul.getInputs()[1];
  Value d = activation.getOutputs()[0];

  if (failed(validateOperand(matmul, a, "GEMM operand A")) ||
      failed(validateOperand(matmul, b, "GEMM operand B")) ||
      failed(validateOperand(activation, d, "fused output D")))
    return failure();

  auto aType = cast<MemRefType>(a.getType());
  auto bType = cast<MemRefType>(b.getType());
  int64_t staticM = aType.getDimSize(0);
  int64_t staticK = aType.getDimSize(1);
  int64_t staticKb = bType.getDimSize(0);
  if (!ShapedType::isDynamic(staticK) && !ShapedType::isDynamic(staticKb) &&
      staticK != staticKb)
    return matmul.emitError()
           << "inner dimension mismatch: A has K=" << staticK << " but B has K="
           << staticKb;
  (void)staticM;

  ModuleOp module = matmul->getParentOfType<ModuleOp>();
  OpBuilder::InsertionGuard guard(builder);
  // Insert at the *activation* op's position, not the matmul's: `d` (the
  // activation's output buffer) is typically allocated between the two
  // matched ops, so it only dominates program points from there onward.
  builder.setInsertionPoint(activation);

  func::FuncOp externalFn = getOrInsertExternalDecl(builder, module, loc);

  Value m = memref::DimOp::create(builder, loc, a, 0);
  Value k = memref::DimOp::create(builder, loc, a, 1);
  Value n = memref::DimOp::create(builder, loc, b, 1);
  Type i64Ty = builder.getI64Type();
  Value mI64 = arith::IndexCastOp::create(builder, loc, i64Ty, m);
  Value nI64 = arith::IndexCastOp::create(builder, loc, i64Ty, n);
  Value kI64 = arith::IndexCastOp::create(builder, loc, i64Ty, k);

  // The matched matmul carries no explicit alpha/beta scaling attributes at
  // this IR level, so the fusion represents the common "D = Act(A @ B)"
  // case: alpha = 1, beta = 0 (no accumulation onto a pre-existing D).
  Value alpha = arith::ConstantFloatOp::create(builder, loc, builder.getF32Type(),
                                               llvm::APFloat(1.0f));
  Value beta = arith::ConstantFloatOp::create(builder, loc, builder.getF32Type(),
                                              llvm::APFloat(0.0f));

  Value aCast = castToCanonical(builder, loc, a);
  Value bCast = castToCanonical(builder, loc, b);
  Value dCast = castToCanonical(builder, loc, d);

  func::CallOp::create(
      builder, loc, externalFn,
      ValueRange{aCast, bCast, dCast, mI64, nI64, kI64, alpha, beta});

  Value matmulOutput = matmul.getOutputs()[0];
  activation.erase();
  matmul.erase();
  eraseIfDeadAlloc(matmulOutput);

  return success();
}

} // namespace cu_epilogue

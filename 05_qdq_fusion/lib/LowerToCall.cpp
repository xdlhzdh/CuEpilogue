#include "CutlassQGemm/CutlassQGemmOps.h"
#include "CutlassQGemm/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace cu_epilogue;

namespace {
constexpr char kExternalSymbol[] = "cutlass_qgemm_bias_gelu";

Type dynI8Matrix(MLIRContext *ctx) {
  return MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                         IntegerType::get(ctx, 8));
}
Type dynF32Vec(MLIRContext *ctx) {
  return MemRefType::get({ShapedType::kDynamic}, Float32Type::get(ctx));
}

LogicalResult validateI8Matrix(Operation *op, Value v, const char *role) {
  auto ty = dyn_cast<MemRefType>(v.getType());
  if (!ty)
    return op->emitError() << role << " is not a memref";
  if (ty.getRank() != 2)
    return op->emitError() << role << " must be rank-2";
  if (!ty.getElementType().isInteger(8))
    return op->emitError() << role << " must have i8 elements";
  if (!ty.getLayout().isIdentity())
    return op->emitError() << role << " must be identity layout";
  return success();
}

LogicalResult validateF32Vec(Operation *op, Value v, const char *role) {
  auto ty = dyn_cast<MemRefType>(v.getType());
  if (!ty)
    return op->emitError() << role << " is not a memref";
  if (ty.getRank() != 1)
    return op->emitError() << role << " must be rank-1";
  if (!ty.getElementType().isF32())
    return op->emitError() << role << " must have f32 elements";
  if (!ty.getLayout().isIdentity())
    return op->emitError() << role << " must be identity layout";
  return success();
}

Value castMemref(OpBuilder &b, Location loc, Value v, Type canonical) {
  if (v.getType() == canonical)
    return v;
  return memref::CastOp::create(b, loc, canonical, v);
}

func::FuncOp getOrInsertDecl(OpBuilder &builder, ModuleOp module, Location loc) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kExternalSymbol))
    return existing;
  MLIRContext *ctx = builder.getContext();
  auto fnTy = builder.getFunctionType(
      {dynI8Matrix(ctx), dynI8Matrix(ctx), dynF32Vec(ctx), dynI8Matrix(ctx),
       builder.getI64Type(), builder.getI64Type(), builder.getI64Type(),
       builder.getF32Type(), builder.getI32Type(), builder.getF32Type(),
       builder.getI32Type(), builder.getF32Type(), builder.getI32Type()},
      {});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto fn = func::FuncOp::create(builder, loc, kExternalSymbol, fnTy);
  fn.setPrivate();
  return fn;
}

class LowerCutlassQgemmToCallPass
    : public PassWrapper<LowerCutlassQgemmToCallPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCutlassQgemmToCallPass)

  StringRef getArgument() const override { return "lower-cutlass-qgemm-to-call"; }
  StringRef getDescription() const override {
    return "Lower memref cutlass.qgemm_bias_gelu to func.call "
           "@cutlass_qgemm_bias_gelu";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, CutlassDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<QgemmBiasGeluOp> ops;
    module.walk([&](QgemmBiasGeluOp op) { ops.push_back(op); });
    if (ops.empty())
      return;

    IRRewriter rewriter(&getContext());
    for (QgemmBiasGeluOp op : ops) {
      if (op.getNumResults() != 0) {
        op.emitError("expected memref form (run one-shot-bufferize first)");
        signalPassFailure();
        return;
      }
      if (failed(validateI8Matrix(op, op.getA(), "A")) ||
          failed(validateI8Matrix(op, op.getB(), "B")) ||
          failed(validateF32Vec(op, op.getBias(), "bias")) ||
          failed(validateI8Matrix(op, op.getD(), "D"))) {
        signalPassFailure();
        return;
      }

      Location loc = op.getLoc();
      rewriter.setInsertionPoint(op);
      func::FuncOp fn = getOrInsertDecl(rewriter, module, loc);

      Value a = op.getA();
      Value m = memref::DimOp::create(rewriter, loc, a, 0);
      Value k = memref::DimOp::create(rewriter, loc, a, 1);
      Value n = memref::DimOp::create(rewriter, loc, op.getB(), 1);
      Type i64Ty = rewriter.getI64Type();
      Value mI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, m);
      Value nI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, n);
      Value kI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, k);

      Type aTy = dynI8Matrix(rewriter.getContext());
      Type biasTy = dynF32Vec(rewriter.getContext());
      func::CallOp::create(
          rewriter, loc, fn,
          ValueRange{castMemref(rewriter, loc, op.getA(), aTy),
                     castMemref(rewriter, loc, op.getB(), aTy),
                     castMemref(rewriter, loc, op.getBias(), biasTy),
                     castMemref(rewriter, loc, op.getD(), aTy), mI64, nI64,
                     kI64, op.getScaleA(), op.getZpA(), op.getScaleB(),
                     op.getZpB(), op.getScaleD(), op.getZpD()});
      rewriter.eraseOp(op);
    }
  }
};

} // namespace

namespace cu_epilogue {

std::unique_ptr<Pass> createLowerCutlassQgemmToCallPass() {
  return std::make_unique<LowerCutlassQgemmToCallPass>();
}

void registerLowerCutlassQgemmToCallPass() {
  PassRegistration<LowerCutlassQgemmToCallPass>();
}

} // namespace cu_epilogue

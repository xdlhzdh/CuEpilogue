#include "CutlassQGemm/CutlassQGemmOps.h"
#include "CutlassQGemm/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace cu_epilogue;

namespace {
constexpr char kExternalSymbol[] = "cutlass_qgemm_bias_gelu";

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

// Address of element 0. The validators guarantee an identity layout, whose
// offset is statically 0, so the aligned base pointer is enough. The pointer
// does not keep the buffer alive: buffer deallocation must run before this
// lowering (its deallocs then already follow the fused op), not after.
Value toBarePointer(OpBuilder &b, Location loc, Value memref) {
  Value index = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, memref);
  Value i64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), index);
  return LLVM::IntToPtrOp::create(
      b, loc, LLVM::LLVMPointerType::get(b.getContext()), i64);
}

// Matches the runtime C API (runtime/cutlass_qgemm_bias_gelu_c_api.cu).
// Buffers are passed as `!llvm.ptr`, not memref: a memref argument lowers to
// an unpacked descriptor under the default LLVM calling convention, and the
// bare-pointer convention rejects dynamically shaped memrefs. The C API only
// accepts device pointers; placing the buffers is the caller's job.
func::FuncOp getOrInsertDecl(OpBuilder &builder, ModuleOp module, Location loc) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kExternalSymbol))
    return existing;
  Type ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto fnTy = builder.getFunctionType(
      {ptrTy, ptrTy, ptrTy, ptrTy,
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
                    memref::MemRefDialect, LLVM::LLVMDialect, CutlassDialect>();
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

      func::CallOp::create(
          rewriter, loc, fn,
          ValueRange{toBarePointer(rewriter, loc, op.getA()),
                     toBarePointer(rewriter, loc, op.getB()),
                     toBarePointer(rewriter, loc, op.getBias()),
                     toBarePointer(rewriter, loc, op.getD()), mI64, nI64,
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

#include "CutlassQGemm/CutlassQGemmOps.h"
#include "CutlassQGemm/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace cu_epilogue;

namespace {

bool allParallelIdentity(linalg::GenericOp generic) {
  for (utils::IteratorType iter : generic.getIteratorTypesArray())
    if (iter != utils::IteratorType::parallel)
      return false;
  for (AffineMap map : generic.getIndexingMapsArray())
    if (!map.isIdentity())
      return false;
  return true;
}

bool isFloatConstantNear(Value v, double expected, double tol = 1e-4) {
  APFloat ap(0.0);
  if (!matchPattern(v, m_ConstantFloat(&ap)))
    return false;
  return std::abs(ap.convertToDouble() - expected) <= tol;
}

/// Fast-GELU: yield(divf(x, addf(exp2(mulf(x, -2.455492)), 1)))
bool matchFastGeluBody(Value yielded, Value blockInput) {
  auto div = yielded.getDefiningOp<arith::DivFOp>();
  if (!div)
    return false;
  if (div.getLhs() != blockInput)
    return false;
  auto add = div.getRhs().getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;
  Value expCandidate;
  if (isFloatConstantNear(add.getRhs(), 1.0))
    expCandidate = add.getLhs();
  else if (isFloatConstantNear(add.getLhs(), 1.0))
    expCandidate = add.getRhs();
  else
    return false;
  auto exp2 = expCandidate.getDefiningOp<math::Exp2Op>();
  if (!exp2)
    return false;
  auto mul = exp2.getOperand().getDefiningOp<arith::MulFOp>();
  if (!mul)
    return false;
  Value x;
  if (isFloatConstantNear(mul.getRhs(), -2.455492))
    x = mul.getLhs();
  else if (isFloatConstantNear(mul.getLhs(), -2.455492))
    x = mul.getRhs();
  else
    return false;
  return x == blockInput;
}

bool isElementwiseTensorGeneric(linalg::GenericOp generic, unsigned numIns,
                                Type inElem, Type outElem) {
  if (generic.getNumDpsInputs() != static_cast<int>(numIns) ||
      generic.getNumDpsInits() != 1 || generic.getNumResults() != 1)
    return false;
  auto outTy = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!outTy || outTy.getElementType() != outElem)
    return false;
  for (unsigned i = 0; i < numIns; ++i) {
    auto inTy = dyn_cast<RankedTensorType>(generic.getInputs()[i].getType());
    if (!inTy)
      return false;
    if (i == 0 && inTy.getElementType() != inElem)
      return false;
  }
  return true;
}

/// DQ: sitofp(i8) then optional sub(zp) then mul(scale).
bool matchDequant(linalg::GenericOp generic, Value &scale, Value &zp) {
  if (!isElementwiseTensorGeneric(generic, 1, IntegerType::get(generic.getContext(), 8),
                                  Float32Type::get(generic.getContext())))
    return false;
  if (!allParallelIdentity(generic))
    return false;

  Block &block = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  Value y = yield.getOperand(0);
  auto mul = y.getDefiningOp<arith::MulFOp>();
  if (!mul)
    return false;

  Value scaled, sc;
  if (mul.getLhs().getDefiningOp<arith::SubFOp>() ||
      mul.getLhs().getDefiningOp<arith::SIToFPOp>()) {
    scaled = mul.getLhs();
    sc = mul.getRhs();
  } else if (mul.getRhs().getDefiningOp<arith::SubFOp>() ||
             mul.getRhs().getDefiningOp<arith::SIToFPOp>()) {
    scaled = mul.getRhs();
    sc = mul.getLhs();
  } else {
    return false;
  }
  scale = sc;

  Value inArg = block.getArgument(0);
  if (auto sub = scaled.getDefiningOp<arith::SubFOp>()) {
    Value sitofpVal, zpVal;
    if (sub.getLhs().getDefiningOp<arith::SIToFPOp>()) {
      sitofpVal = sub.getLhs();
      zpVal = sub.getRhs();
    } else if (sub.getRhs().getDefiningOp<arith::SIToFPOp>()) {
      sitofpVal = sub.getRhs();
      zpVal = sub.getLhs();
    } else {
      return false;
    }
    auto sitofp = sitofpVal.getDefiningOp<arith::SIToFPOp>();
    if (!sitofp || sitofp.getIn() != inArg)
      return false;
    if (auto zpCast = zpVal.getDefiningOp<arith::SIToFPOp>())
      zp = zpCast.getIn();
    else
      zp = zpVal;
    return true;
  }

  auto sitofp = scaled.getDefiningOp<arith::SIToFPOp>();
  if (!sitofp || sitofp.getIn() != inArg)
    return false;
  OpBuilder b(generic);
  zp = arith::ConstantOp::create(b, generic.getLoc(), b.getI32IntegerAttr(0));
  return true;
}

bool matchBiasAdd(linalg::GenericOp generic, Value acc, Value &bias) {
  if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1 ||
      generic.getNumResults() != 1)
    return false;
  if (generic.getInputs()[0] != acc)
    return false;

  ArrayRef<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;
  if (!maps[0].isIdentity() || !maps[2].isIdentity())
    return false;
  // bias map: (i, j) -> (j)
  if (maps[1].getNumDims() != 2 || maps[1].getNumResults() != 1)
    return false;
  if (!isa<AffineDimExpr>(maps[1].getResult(0)) ||
      cast<AffineDimExpr>(maps[1].getResult(0)).getPosition() != 1)
    return false;

  for (utils::IteratorType iter : generic.getIteratorTypesArray())
    if (iter != utils::IteratorType::parallel)
      return false;

  Block &block = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto add = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;
  Value a0 = block.getArgument(0);
  Value a1 = block.getArgument(1);
  if (!((add.getLhs() == a0 && add.getRhs() == a1) ||
        (add.getLhs() == a1 && add.getRhs() == a0)))
    return false;
  bias = generic.getInputs()[1];
  return true;
}

bool matchGeluGeneric(linalg::GenericOp generic, Value expectedInput) {
  if (!isElementwiseTensorGeneric(generic, 1, Float32Type::get(generic.getContext()),
                                  Float32Type::get(generic.getContext())))
    return false;
  if (generic.getInputs()[0] != expectedInput)
    return false;
  if (!allParallelIdentity(generic))
    return false;
  Block &block = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  return matchFastGeluBody(yield.getOperand(0), block.getArgument(0));
}

bool matchQuant(linalg::GenericOp generic, Value expectedInput, Value &scale,
                Value &zp) {
  if (!isElementwiseTensorGeneric(generic, 1, Float32Type::get(generic.getContext()),
                                  IntegerType::get(generic.getContext(), 8)))
    return false;
  if (generic.getInputs()[0] != expectedInput)
    return false;
  if (!allParallelIdentity(generic))
    return false;

  Block &block = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto fptosi = yield.getOperand(0).getDefiningOp<arith::FPToSIOp>();
  if (!fptosi)
    return false;
  Value q = fptosi.getIn();
  Value inArg = block.getArgument(0);

  if (auto add = q.getDefiningOp<arith::AddFOp>()) {
    Value divVal, zpVal;
    if (add.getLhs().getDefiningOp<arith::DivFOp>()) {
      divVal = add.getLhs();
      zpVal = add.getRhs();
    } else if (add.getRhs().getDefiningOp<arith::DivFOp>()) {
      divVal = add.getRhs();
      zpVal = add.getLhs();
    } else {
      return false;
    }
    auto div = divVal.getDefiningOp<arith::DivFOp>();
    if (!div || div.getLhs() != inArg)
      return false;
    scale = div.getRhs();
    if (auto zpCast = zpVal.getDefiningOp<arith::SIToFPOp>())
      zp = zpCast.getIn();
    else
      zp = zpVal;
    return true;
  }

  auto div = q.getDefiningOp<arith::DivFOp>();
  if (!div || div.getLhs() != inArg)
    return false;
  scale = div.getRhs();
  OpBuilder b(generic);
  zp = arith::ConstantOp::create(b, generic.getLoc(), b.getI32IntegerAttr(0));
  return true;
}

bool hasExactlyOneUser(Value v) {
  return v.hasOneUse();
}

class FuseQdqQgemmBiasGeluPass
    : public PassWrapper<FuseQdqQgemmBiasGeluPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseQdqQgemmBiasGeluPass)

  StringRef getArgument() const override { return "fuse-qdq-qgemm-bias-gelu"; }
  StringRef getDescription() const override {
    return "Fuse tensor DQ + matmul + bias + Fast-GELU + Q into "
           "cutlass.qgemm_bias_gelu";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<CutlassDialect, func::FuncDialect, linalg::LinalgDialect,
                    arith::ArithDialect, math::MathDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    SmallVector<SmallVector<Operation *, 8>> toErase;
    SmallVector<QgemmBiasGeluOp> created;

    getOperation().walk([&](linalg::MatmulOp matmul) {
      if (matmul.getNumResults() != 1)
        return;
      if (matmul.getNumDpsInputs() != 2 || matmul.getNumDpsInits() != 1)
        return;

      Value a = matmul.getInputs()[0];
      Value b = matmul.getInputs()[1];
      auto dqA = a.getDefiningOp<linalg::GenericOp>();
      auto dqB = b.getDefiningOp<linalg::GenericOp>();
      if (!dqA || !dqB)
        return;

      Value sa, za, sb, zb;
      if (!matchDequant(dqA, sa, za) || !matchDequant(dqB, sb, zb))
        return;
      if (!hasExactlyOneUser(dqA.getResult(0)) ||
          !hasExactlyOneUser(dqB.getResult(0)))
        return;

      Value acc = matmul.getResult(0);
      if (!hasExactlyOneUser(acc))
        return;
      auto biasOp = dyn_cast<linalg::GenericOp>(*acc.getUsers().begin());
      if (!biasOp)
        return;
      Value bias;
      if (!matchBiasAdd(biasOp, acc, bias) || !hasExactlyOneUser(biasOp.getResult(0)))
        return;

      Value biased = biasOp.getResult(0);
      auto geluOp = dyn_cast<linalg::GenericOp>(*biased.getUsers().begin());
      if (!geluOp || !matchGeluGeneric(geluOp, biased) ||
          !hasExactlyOneUser(geluOp.getResult(0)))
        return;

      Value geluOut = geluOp.getResult(0);
      auto qOp = dyn_cast<linalg::GenericOp>(*geluOut.getUsers().begin());
      if (!qOp)
        return;
      Value sd, zd;
      if (!matchQuant(qOp, geluOut, sd, zd))
        return;

      Value aI8 = dqA.getInputs()[0];
      Value bI8 = dqB.getInputs()[0];
      Value dInit = qOp.getOutputs()[0];

      OpBuilder builder(qOp);
      auto fused = QgemmBiasGeluOp::create(builder, matmul.getLoc(), aI8, bI8,
                                           bias, sa, za, sb, zb, sd, zd, dInit);
      qOp.getResult(0).replaceAllUsesWith(fused.getResult().front());

      SmallVector<Operation *, 8> eraseList{qOp, geluOp, biasOp, matmul, dqA,
                                            dqB};
      toErase.push_back(std::move(eraseList));
    });

    for (auto &ops : toErase) {
      for (Operation *op : ops) {
        if (op && op->use_empty())
          op->erase();
      }
    }
  }
};

} // namespace

namespace cu_epilogue {

std::unique_ptr<Pass> createFuseQdqQgemmBiasGeluPass() {
  return std::make_unique<FuseQdqQgemmBiasGeluPass>();
}

void registerFuseQdqQgemmBiasGeluPass() {
  PassRegistration<FuseQdqQgemmBiasGeluPass>();
}

} // namespace cu_epilogue

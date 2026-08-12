#include "FusedGemmGelu/Passes.h"
#include "EmitExternalCall.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace cu_epilogue {
namespace {

/// True if `v` is an f32/f64 arith.constant whose value is within `tol` of
/// `expected`. Used to recognize Fast-GELU's magic coefficient and the
/// additive `1.0` in `1 + 2^(...)` without depending on bit-identical APFloat
/// encodings after bufferize/canonicalize.
bool isFloatConstantNear(Value v, double expected, double tol = 1e-4) {
  APFloat ap(0.0);
  if (!matchPattern(v, m_ConstantFloat(&ap)))
    return false;
  return std::abs(ap.convertToDouble() - expected) <= tol;
}

/// If `v` is an f32/f64 arith.constant, write it to `out` and return true.
bool matchStaticFloat(Value v, float &out) {
  APFloat ap(0.0);
  if (!matchPattern(v, m_ConstantFloat(&ap)))
    return false;

  // APFloat::convert is in-place and returns opStatus, not a new APFloat.
  bool losesInfo = false;
  ap.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  out = ap.convertToFloat();
  return true;
}

/// Peel an optional leading scale: `x` or `arith.mulf(x, alpha)` /
/// `arith.mulf(alpha, x)`. Returns the unscaled value and writes a static
/// `alpha` (defaults to 1.0 when no mulf is present). Dynamic (non-constant)
/// scales are rejected — after fusion the generic region is erased, so the
/// scale must be materializable as a host-side float constant.
FailureOr<Value> peelStaticAlpha(Value v, Value expectedInput, float &alpha) {
  alpha = 1.0f;
  if (v == expectedInput)
    return v;

  auto mul = v.getDefiningOp<arith::MulFOp>();
  if (!mul)
    return failure();

  Value lhs = mul.getLhs();
  Value rhs = mul.getRhs();
  if (lhs == expectedInput && matchStaticFloat(rhs, alpha))
    return lhs;
  if (rhs == expectedInput && matchStaticFloat(lhs, alpha))
    return rhs;
  return failure();
}

/// True when `v` is `base` (scale = 1) or `arith.mulf(base, cst)` / `mulf(cst, base)`.
bool matchScaledOperand(Value v, Value base, float &scale) {
  if (!base)
    return false;
  if (v == base) {
    scale = 1.0f;
    return true;
  }
  if (auto mul = v.getDefiningOp<arith::MulFOp>()) {
    if (mul.getLhs() == base && matchStaticFloat(mul.getRhs(), scale))
      return true;
    if (mul.getRhs() == base && matchStaticFloat(mul.getLhs(), scale))
      return true;
  }
  return false;
}

/// Recognize `linear` as `alpha * blockInput + beta * blockInit` with static
/// coefficients (defaults alpha=1, beta=0 when terms are absent).
bool extractAlphaBetaFromLinear(Value linear, Value blockInput, Value blockInit,
                                float &alpha, float &beta) {
  alpha = 1.0f;
  beta = 0.0f;

  if (matchScaledOperand(linear, blockInput, alpha)) {
    beta = 0.0f;
    return true;
  }

  auto add = linear.getDefiningOp<arith::AddFOp>();
  if (!add || !blockInit)
    return false;

  float lhsIn = 1.0f, rhsIn = 1.0f, lhsInit = 1.0f, rhsInit = 1.0f;
  bool hasLhsIn = matchScaledOperand(add.getLhs(), blockInput, lhsIn);
  bool hasRhsIn = matchScaledOperand(add.getRhs(), blockInput, rhsIn);
  bool hasLhsInit = matchScaledOperand(add.getLhs(), blockInit, lhsInit);
  bool hasRhsInit = matchScaledOperand(add.getRhs(), blockInit, rhsInit);

  if (hasLhsIn && hasRhsInit && !hasLhsInit && !hasRhsIn) {
    alpha = lhsIn;
    beta = rhsInit;
    return true;
  }
  if (hasRhsIn && hasLhsInit && !hasRhsInit && !hasLhsIn) {
    alpha = rhsIn;
    beta = lhsInit;
    return true;
  }
  return false;
}

/// Spec 3.3 Fast-GELU algebraic form used by fuse_pattern.mlir:
///   GELU(x) ~= x / (1 + 2^(-2.455492 * x))
/// Recognized as:
///   yield(divf(x, addf(exp2(mulf(x, c)), 1)))   with c ≈ -2.455492
/// where `x` is the linear epilogue value `alpha * in + beta * init`.
bool matchFastGeluAlgebraic(Value yielded, Value blockInput, Value blockInit,
                            float &alpha, float &beta) {
  auto div = yielded.getDefiningOp<arith::DivFOp>();
  if (!div)
    return false;

  Value numer = div.getLhs();
  Value denom = div.getRhs();

  auto add = denom.getDefiningOp<arith::AddFOp>();
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

  // One side of the mul must be the Fast-GELU constant; the other is `x`.
  Value x;
  if (isFloatConstantNear(mul.getRhs(), -2.455492))
    x = mul.getLhs();
  else if (isFloatConstantNear(mul.getLhs(), -2.455492))
    x = mul.getRhs();
  else
    return false;

  // Fast-GELU is `x / (1 + 2^(c*x))` — numerator must be the same `x`.
  if (numer != x)
    return false;

  return extractAlphaBetaFromLinear(x, blockInput, blockInit, alpha, beta);
}

/// Classic erf-based GELU: somewhere in the def-use chain of `yielded` there
/// is a `math.erf` whose operand ultimately depends on the block input.
/// We accept this as GELU without reconstructing the full
/// `0.5*x*(1+erf(x/sqrt(2)))` tree (canonicalization may rearrange it).
bool matchErfBasedGelu(Value yielded, Value blockInput, float &alpha) {
  alpha = 1.0f;
  bool sawErf = false;
  bool dependsOnInput = false;

  SmallVector<Value, 8> vals{yielded};
  DenseSet<Value> seenVals;
  while (!vals.empty()) {
    Value cur = vals.pop_back_val();
    if (!seenVals.insert(cur).second)
      continue;
    if (cur == blockInput)
      dependsOnInput = true;

    if (auto mul = cur.getDefiningOp<arith::MulFOp>()) {
      float maybeAlpha = 1.0f;
      if (mul.getLhs() == blockInput && matchStaticFloat(mul.getRhs(), maybeAlpha))
        alpha = maybeAlpha;
      else if (mul.getRhs() == blockInput &&
               matchStaticFloat(mul.getLhs(), maybeAlpha))
        alpha = maybeAlpha;
    }

    if (auto erf = cur.getDefiningOp<math::ErfOp>()) {
      sawErf = true;
      vals.push_back(erf.getOperand());
      continue;
    }

    Operation *def = cur.getDefiningOp();
    if (!def)
      continue;
    for (Value operand : def->getOperands())
      vals.push_back(operand);
  }

  return sawErf && dependsOnInput;
}

/// Tanh-based GELU approximation: body contains `math.tanh` whose result
/// feeds into the yielded value and depends on the block input.
bool matchTanhBasedGelu(Value yielded, Value blockInput, float &alpha) {
  alpha = 1.0f;
  bool sawTanh = false;
  bool dependsOnInput = false;

  SmallVector<Value, 8> vals{yielded};
  DenseSet<Value> seenVals;
  while (!vals.empty()) {
    Value cur = vals.pop_back_val();
    if (!seenVals.insert(cur).second)
      continue;
    if (cur == blockInput)
      dependsOnInput = true;

    if (auto mul = cur.getDefiningOp<arith::MulFOp>()) {
      float maybeAlpha = 1.0f;
      if (mul.getLhs() == blockInput && matchStaticFloat(mul.getRhs(), maybeAlpha))
        alpha = maybeAlpha;
      else if (mul.getRhs() == blockInput &&
               matchStaticFloat(mul.getLhs(), maybeAlpha))
        alpha = maybeAlpha;
    }

    if (auto tanh = cur.getDefiningOp<math::TanhOp>()) {
      sawTanh = true;
      vals.push_back(tanh.getOperand());
      continue;
    }

    Operation *def = cur.getDefiningOp();
    if (!def)
      continue;
    for (Value operand : def->getOperands())
      vals.push_back(operand);
  }

  return sawTanh && dependsOnInput;
}

/// Forward-compatible: a single `math.gelu` (or similarly named) op on the
/// (optionally scaled) block input. Not present in all MLIR versions.
bool matchNamedGeluOp(Value yielded, Value blockInput, float &alpha) {
  Operation *def = yielded.getDefiningOp();
  if (!def)
    return false;
  StringRef name = def->getName().getStringRef();
  if (name != "math.gelu" && name != "math.GELU")
    return false;
  if (def->getNumOperands() != 1)
    return false;
  return succeeded(peelStaticAlpha(def->getOperand(0), blockInput, alpha));
}

/// Optional epilogue term `+ beta * init` folded into the same generic.
/// When the bufferized init block-arg appears in an `arith.addf` either bare
/// (beta = 1) or as `mulf(init, beta_cst)`, record that static coefficient.
/// Dynamic coefficients are ignored — the region is erased on rewrite.
void tryExtractBetaFromAdd(Value blockInit, float &beta) {
  beta = 0.0f;
  if (!blockInit)
    return;

  for (Operation *user : blockInit.getUsers()) {
    auto add = dyn_cast<arith::AddFOp>(user);
    if (!add)
      continue;

    if (add.getLhs() == blockInit || add.getRhs() == blockInit) {
      beta = 1.0f;
      return;
    }
  }

  // Also accept `addf(x, mulf(init, beta))` where the mul's result is an
  // operand of some addf that uses blockInit indirectly via the mul.
  for (Operation *user : blockInit.getUsers()) {
    auto mul = dyn_cast<arith::MulFOp>(user);
    if (!mul)
      continue;
    float b = 0.0f;
    if (mul.getLhs() == blockInit && matchStaticFloat(mul.getRhs(), b)) {
      // Confirm the scaled init feeds an addf (linear combination).
      for (Operation *mulUser : mul->getUsers()) {
        if (isa<arith::AddFOp>(mulUser)) {
          beta = b;
          return;
        }
      }
    }
    if (mul.getRhs() == blockInit && matchStaticFloat(mul.getLhs(), b)) {
      for (Operation *mulUser : mul->getUsers()) {
        if (isa<arith::AddFOp>(mulUser)) {
          beta = b;
          return;
        }
      }
    }
  }
}

/// Inspect the `linalg.generic` region and decide whether it is a recognized
/// GELU (Fast-GELU / erf / tanh / named math.gelu). On success, writes the
/// extracted static `alpha` / `beta` (defaults 1.0 / 0.0).
bool matchGeluBody(linalg::GenericOp generic, float &alpha, float &beta) {
  alpha = 1.0f;
  beta = 0.0f;

  Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return false;
  Block &block = region.front();
  // Bufferized unary elementwise generic: bb args are (input, init/out).
  if (block.getNumArguments() < 1)
    return false;

  Value blockInput = block.getArgument(0);
  Value blockInit =
      block.getNumArguments() >= 2 ? block.getArgument(1) : Value();

  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  Value yielded = yield.getOperand(0);

  // Prefer the project's exact Fast-GELU lowering; then erf / tanh / named.
  bool matchedFastGelu =
      matchFastGeluAlgebraic(yielded, blockInput, blockInit, alpha, beta);
  bool isGelu = matchedFastGelu || matchNamedGeluOp(yielded, blockInput, alpha) ||
                matchErfBasedGelu(yielded, blockInput, alpha) ||
                matchTanhBasedGelu(yielded, blockInput, alpha);
  if (!isGelu)
    return false;

  // Fast-GELU path already extracted beta from the linear epilogue; erf/tanh
  // paths may still fold a bare `+ init` term via the legacy beta helper.
  if (!matchedFastGelu)
    tryExtractBetaFromAdd(blockInit, beta);
  return true;
}

/// Structural gate + semantic GELU recognition.
/// Returns true only when `generic` is a bufferized, all-parallel, identity-
/// map unary elementwise op that reads `expectedInput` *and* whose region
/// implements a recognized GELU. Writes extracted alpha/beta on success.
bool isElementwiseGeluOf(linalg::GenericOp generic, Value expectedInput,
                         float &alpha, float &beta) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;
  if (generic.getInputs()[0] != expectedInput)
    return false;
  // Tensor-semantics generics still have SSA results; require bufferized form.
  if (generic.getNumResults() != 0)
    return false;

  for (utils::IteratorType iter : generic.getIteratorTypesArray())
    if (iter != utils::IteratorType::parallel)
      return false;

  for (AffineMap map : generic.getIndexingMapsArray())
    if (!map.isIdentity())
      return false;

  // Element type must be a floating-point type the runtime can accept
  // (emit path currently requires f32 memrefs; reject early here too).
  auto inType = dyn_cast<MemRefType>(expectedInput.getType());
  if (!inType || !inType.getElementType().isF32())
    return false;
  auto outType = dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
  if (!outType || !outType.getElementType().isF32())
    return false;
  if (!inType.getLayout().isIdentity() || !outType.getLayout().isIdentity())
    return false;

  return matchGeluBody(generic, alpha, beta);
}

class FuseGemmGeluPass
    : public PassWrapper<FuseGemmGeluPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseGemmGeluPass)

  StringRef getArgument() const override { return "fuse-gemm-gelu"; }
  StringRef getDescription() const override {
    return "Fuse a bufferized linalg.matmul + GELU linalg.generic into a "
           "call to the external CUTLASS fused GEMM+GELU kernel "
           "(@cutlass_fused_gemm_gelu)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    arith::ArithDialect, math::MathDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<FusedGemmGeluMatch> matches;
    func.walk([&](linalg::MatmulOp matmulOp) {
      // Only memref-based (post-bufferization) matmuls are eligible: a
      // tensor-semantics matmul still has a result; a bufferized one writes
      // in-place to its `outs` operand and has none.
      if (matmulOp.getNumResults() != 0)
        return;
      if (matmulOp.getNumDpsInputs() != 2 || matmulOp.getNumDpsInits() != 1)
        return;

      // Early type/layout check on matmul operands — keeps the rewriter from
      // discovering a "match" that emitExternalFusedCall would later reject.
      for (Value operand :
           {matmulOp.getInputs()[0], matmulOp.getInputs()[1],
            matmulOp.getOutputs()[0]}) {
        auto ty = dyn_cast<MemRefType>(operand.getType());
        if (!ty || !ty.getElementType().isF32() || !ty.getLayout().isIdentity())
          return;
      }

      Value matmulOutput = matmulOp.getOutputs()[0];
      linalg::GenericOp matchedActivation;
      float alpha = 1.0f;
      float beta = 0.0f;
      int otherUses = 0;
      for (Operation *user : matmulOutput.getUsers()) {
        if (user == matmulOp.getOperation())
          continue;
        ++otherUses;
        if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
          float a = 1.0f, b = 0.0f;
          if (isElementwiseGeluOf(generic, matmulOutput, a, b)) {
            matchedActivation = generic;
            alpha = a;
            beta = b;
          }
        }
      }

      // Require the matmul output to be consumed *only* by the matched GELU —
      // otherwise erasing the matmul would change other observers' values.
      if (otherUses == 1 && matchedActivation)
        matches.push_back({matmulOp, matchedActivation, alpha, beta});
    });

    if (matches.empty())
      return;

    IRRewriter rewriter(&getContext());
    for (const FusedGemmGeluMatch &match : matches) {
      if (failed(emitExternalFusedCall(rewriter, match))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFuseGemmGeluPass() {
  return std::make_unique<FuseGemmGeluPass>();
}

void registerFuseGemmGeluPass() { PassRegistration<FuseGemmGeluPass>(); }

} // namespace cu_epilogue

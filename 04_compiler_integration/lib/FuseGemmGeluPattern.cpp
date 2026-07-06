#include "FusedGemmGelu/Passes.h"
#include "EmitExternalCall.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace cu_epilogue {
namespace {

// True if `generic` is a purely elementwise (all-parallel, identity-map)
// unary op reading `expectedInput` - the shape a lowered activation
// (GELU, ReLU, ...) takes once it has been bufferized. We deliberately do
// *not* inspect the op body: matching is structural (spec 3.4's "识别到相连的
// linalg.matmul 与对应的非线性激活 Ops"), matching whatever elementwise
// transform the upstream lowering produced.
bool isElementwiseActivationOf(linalg::GenericOp generic, Value expectedInput) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;
  if (generic.getInputs()[0] != expectedInput) return false;
  if (generic.getNumResults() != 0) return false; // must be bufferized already

  for (utils::IteratorType iter : generic.getIteratorTypesArray())
    if (iter != utils::IteratorType::parallel) return false;

  for (AffineMap map : generic.getIndexingMapsArray())
    if (!map.isIdentity()) return false;

  return true;
}

class FuseGemmGeluPass
    : public PassWrapper<FuseGemmGeluPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseGemmGeluPass)

  StringRef getArgument() const override { return "fuse-gemm-gelu"; }
  StringRef getDescription() const override {
    return "Fuse a bufferized linalg.matmul + elementwise linalg.generic "
          "(activation) pair into a call to the external CUTLASS fused "
          "GEMM+GELU kernel (@cutlass_fused_gemm_gelu)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::SmallVector<FusedGemmGeluMatch> matches;
    func.walk([&](linalg::MatmulOp matmulOp) {
      // Only memref-based (post-bufferization) matmuls are eligible: a
      // tensor-semantics matmul still has a result, a bufferized one
      // writes in-place to its `outs` operand and has none.
      if (matmulOp.getNumResults() != 0) return;
      if (matmulOp.getNumDpsInputs() != 2 || matmulOp.getNumDpsInits() != 1)
        return;

      Value matmulOutput = matmulOp.getOutputs()[0];
      linalg::GenericOp matchedActivation;
      int otherUses = 0;
      for (Operation *user : matmulOutput.getUsers()) {
        if (user == matmulOp.getOperation()) continue;
        ++otherUses;
        if (auto generic = dyn_cast<linalg::GenericOp>(user))
          if (isElementwiseActivationOf(generic, matmulOutput))
            matchedActivation = generic;
      }

      // Require the matmul's output to be consumed *only* by the matched
      // activation - if anything else reads it, erasing the matmul would
      // change that other use's observed value.
      if (otherUses == 1 && matchedActivation)
        matches.push_back({matmulOp, matchedActivation});
    });

    if (matches.empty()) return;

    OpBuilder builder(&getContext());
    for (const FusedGemmGeluMatch &match : matches) {
      if (failed(emitExternalFusedCall(builder, match))) {
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

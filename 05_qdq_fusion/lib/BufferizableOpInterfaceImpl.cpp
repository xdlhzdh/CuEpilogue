#include "CutlassQGemm/CutlassQGemmOps.h"
#include "CutlassQGemm/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace cu_epilogue;

namespace {

struct QgemmBiasGeluOpInterface
    : public DstBufferizableOpInterfaceExternalModel<QgemmBiasGeluOpInterface,
                                                     QgemmBiasGeluOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto qop = cast<QgemmBiasGeluOp>(op);

    auto getIfTensor = [&](Value v) -> FailureOr<Value> {
      if (isa<TensorType>(v.getType()))
        return getBuffer(rewriter, v, options, state);
      return v;
    };

    FailureOr<Value> a = getIfTensor(qop.getA());
    FailureOr<Value> b = getIfTensor(qop.getB());
    FailureOr<Value> bias = getIfTensor(qop.getBias());
    FailureOr<Value> d = getIfTensor(qop.getD());
    if (failed(a) || failed(b) || failed(bias) || failed(d))
      return failure();

    rewriter.setInsertionPoint(op);
    auto newOp = QgemmBiasGeluOp::create(
        rewriter, op->getLoc(), *a, *b, *bias, qop.getScaleA(), qop.getZpA(),
        qop.getScaleB(), qop.getZpB(), qop.getScaleD(), qop.getZpD(), *d);
    (void)newOp;
    replaceOpWithBufferizedValues(rewriter, op, ValueRange{*d});
    return success();
  }
};

} // namespace

void cu_epilogue::registerCutlassBufferizationExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CutlassDialect * /*dialect*/) {
    QgemmBiasGeluOp::attachInterface<QgemmBiasGeluOpInterface>(*ctx);
  });
}

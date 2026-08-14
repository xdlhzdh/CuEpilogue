#include "CutlassQGemm/CutlassQGemmDialect.h"
#include "CutlassQGemm/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  cu_epilogue::registerFuseQdqQgemmBiasGeluPass();
  cu_epilogue::registerLowerCutlassQgemmToCallPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<cu_epilogue::CutlassDialect>();
  cu_epilogue::registerCutlassBufferizationExternalModels(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CuEpilogue qdq-opt\n", registry));
}

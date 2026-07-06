// Stage 4 - `fused-opt`: a small mlir-opt-alike that additionally registers
// the `-fuse-gemm-gelu` pass (FusedGemmGelu/Passes.h) alongside every
// upstream dialect/pass, so a standard pipeline can be driven entirely
// from the command line, e.g.:
//
//   fused-opt input.mlir \
//     --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
//     --fuse-gemm-gelu
#include "FusedGemmGelu/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  cu_epilogue::registerFuseGemmGeluPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CuEpilogue fused-opt\n", registry));
}

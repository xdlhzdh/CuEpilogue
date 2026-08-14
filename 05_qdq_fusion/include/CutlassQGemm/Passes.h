#ifndef CU_EPILOGUE_CUTLASS_QGEMM_PASSES_H
#define CU_EPILOGUE_CUTLASS_QGEMM_PASSES_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace cu_epilogue {

std::unique_ptr<mlir::Pass> createFuseQdqQgemmBiasGeluPass();
std::unique_ptr<mlir::Pass> createLowerCutlassQgemmToCallPass();

void registerFuseQdqQgemmBiasGeluPass();
void registerLowerCutlassQgemmToCallPass();

/// Register BufferizableOpInterface on cutlass.qgemm_bias_gelu so One-Shot
/// Bufferize can convert the DPS tensor form to memref without allocating
/// intermediates for DQ/GELU/Q.
void registerCutlassBufferizationExternalModels(mlir::DialectRegistry &registry);

} // namespace cu_epilogue

#endif

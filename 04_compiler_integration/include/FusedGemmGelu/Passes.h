#ifndef CU_EPILOGUE_FUSEDGEMMGELU_PASSES_H
#define CU_EPILOGUE_FUSEDGEMMGELU_PASSES_H
// Stage 4 - AI compiler backend integration (spec 3.4).
//
// Declares the `-fuse-gemm-gelu` pass: run *after* bufferization, it
// recognizes a bufferized `linalg.matmul` immediately followed by an
// elementwise `linalg.generic` (standing in for a lowered activation such
// as GELU) and replaces both with a single `func.call` to an external
// symbol (`@cutlass_fused_gemm_gelu`) backed by the Stage 2/3 CUTLASS
// kernel, instead of continuing to lower the pair to scalar LLVM IR.
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace cu_epilogue {

std::unique_ptr<mlir::Pass> createFuseGemmGeluPass();

// Registers `-fuse-gemm-gelu` with MLIR's global pass registry so it can be
// selected by name from the fused-opt command line.
void registerFuseGemmGeluPass();

} // namespace cu_epilogue

#endif // CU_EPILOGUE_FUSEDGEMMGELU_PASSES_H

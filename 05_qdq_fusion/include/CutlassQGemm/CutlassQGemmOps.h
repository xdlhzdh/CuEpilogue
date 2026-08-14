#ifndef CU_EPILOGUE_CUTLASS_QGEMM_OPS_H
#define CU_EPILOGUE_CUTLASS_QGEMM_OPS_H

#include "CutlassQGemm/CutlassQGemmDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "CutlassQGemmOps.h.inc"

#endif

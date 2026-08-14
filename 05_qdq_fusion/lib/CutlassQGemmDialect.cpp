#include "CutlassQGemm/CutlassQGemmDialect.h"
#include "CutlassQGemm/CutlassQGemmOps.h"

using namespace mlir;
using namespace cu_epilogue;

#include "CutlassQGemmDialect.cpp.inc"

void CutlassDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "CutlassQGemmOps.cpp.inc"
      >();
}

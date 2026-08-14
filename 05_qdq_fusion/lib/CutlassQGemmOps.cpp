#include "CutlassQGemm/CutlassQGemmOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace cu_epilogue;

#define GET_OP_CLASSES
#include "CutlassQGemmOps.cpp.inc"

static bool isRankedI8(Type t, int64_t rank) {
  if (auto tensor = dyn_cast<RankedTensorType>(t))
    return tensor.getRank() == rank && tensor.getElementType().isInteger(8);
  if (auto memref = dyn_cast<MemRefType>(t))
    return memref.getRank() == rank && memref.getElementType().isInteger(8);
  return false;
}

static bool isRankedF32(Type t, int64_t rank) {
  if (auto tensor = dyn_cast<RankedTensorType>(t))
    return tensor.getRank() == rank && tensor.getElementType().isF32();
  if (auto memref = dyn_cast<MemRefType>(t))
    return memref.getRank() == rank && memref.getElementType().isF32();
  return false;
}

LogicalResult QgemmBiasGeluOp::verify() {
  if (!isRankedI8(getA().getType(), 2))
    return emitOpError("A must be rank-2 i8 tensor or memref");
  if (!isRankedI8(getB().getType(), 2))
    return emitOpError("B must be rank-2 i8 tensor or memref");
  if (!isRankedF32(getBias().getType(), 1))
    return emitOpError("bias must be rank-1 f32 tensor or memref");
  if (!isRankedI8(getD().getType(), 2))
    return emitOpError("D must be rank-2 i8 tensor or memref");

  bool destIsTensor = isa<RankedTensorType>(getD().getType());
  if (destIsTensor) {
    if (getNumResults() != 1)
      return emitOpError("tensor form must have exactly one result");
    if (getResult().empty() || getResult().front().getType() != getD().getType())
      return emitOpError("result type must match destination D");
  } else if (getNumResults() != 0) {
    return emitOpError("memref form must have no results");
  }
  return success();
}

void QgemmBiasGeluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto addIfMemref = [&](unsigned idx, MemoryEffects::Effect *effect) {
    OpOperand &operand = getOperation()->getOpOperand(idx);
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(effect, &operand,
                           SideEffects::DefaultResource::get());
  };
  addIfMemref(0, MemoryEffects::Read::get());  // A
  addIfMemref(1, MemoryEffects::Read::get());  // B
  addIfMemref(2, MemoryEffects::Read::get());  // bias
  addIfMemref(9, MemoryEffects::Read::get());  // D
  addIfMemref(9, MemoryEffects::Write::get()); // D
}

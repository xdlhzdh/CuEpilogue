#ifndef CUTLASS_QGEMM_OPS_TD
#define CUTLASS_QGEMM_OPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cutlass_Dialect : Dialect {
  let name = "cutlass";
  let summary = "Fused CUTLASS QGEMM ops for CuEpilogue stage 5";
  let description = [{
    High-level destination-passing ops that represent a fused
    DQ + INT8 GEMM + Bias + Fast-GELU + Q kernel. Matched on
    Linalg-on-Tensor IR, bufferized in-place by One-Shot Bufferize,
    then lowered to an external C ABI call.
  }];
  let cppNamespace = "::cu_epilogue";
  let useDefaultAttributePrinterParser = 0;
}

class Cutlass_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cutlass_Dialect, mnemonic, traits>;

def AnyTensorOrMemRef : Type<
    CPred<"::llvm::isa<::mlir::ShapedType>($_self)">,
    "ranked tensor or memref">;

def Cutlass_QgemmBiasGeluOp : Cutlass_Op<"qgemm_bias_gelu", [
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Fused INT8 GEMM + bias + Fast-GELU + quantize";
  let description = [{
    Computes `D = Q(FastGELU(DQ(A) @ DQ(B) + bias))` with per-tensor
    affine INT8 quantization (scale + zero-point).

    Destination-passing: `D` is the init/output. Tensor form has one
    result tied to `D`; memref form has no results and writes `D` in
    place.
  }];

  let arguments = (ins
    AnyTensorOrMemRef:$A,
    AnyTensorOrMemRef:$B,
    AnyTensorOrMemRef:$bias,
    F32:$scale_a,
    I32:$zp_a,
    F32:$scale_b,
    I32:$zp_b,
    F32:$scale_d,
    I32:$zp_d,
    AnyTensorOrMemRef:$D
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$A, "::mlir::Value":$B, "::mlir::Value":$bias,
                   "::mlir::Value":$scale_a, "::mlir::Value":$zp_a,
                   "::mlir::Value":$scale_b, "::mlir::Value":$zp_b,
                   "::mlir::Value":$scale_d, "::mlir::Value":$zp_d,
                   "::mlir::Value":$D), [{
      ::llvm::SmallVector<::mlir::Type, 1> resultTypes;
      if (::llvm::isa<::mlir::RankedTensorType>(D.getType()))
        resultTypes.push_back(D.getType());
      build($_builder, $_state, resultTypes, A, B, bias,
            scale_a, zp_a, scale_b, zp_b, scale_d, zp_d, D);
    }]>
  ];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $A `,` $B `,` $bias `,`
              $scale_a `,` $zp_a `,` $scale_b `,` $zp_b `,`
              $scale_d `,` $zp_d
          `:` type($A) `,` type($B) `,` type($bias) `,`
              type($scale_a) `,` type($zp_a) `,` type($scale_b) `,`
              type($zp_b) `,` type($scale_d) `,` type($zp_d) `)`
    `outs` `(` $D `:` type($D) `)`
    attr-dict (`->` type($result)^)?
  }];

  let hasVerifier = 1;
}

#endif

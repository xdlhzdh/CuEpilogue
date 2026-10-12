#include "FusedGemmGelu/Passes.h"
#include "EmitExternalCall.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace cu_epilogue {
namespace {

/// `v` 是 f32/f64 的 arith.constant，且与 `expected` 的差不超过 `tol` 时返回
/// true。用于识别 Fast-GELU 的魔数系数和 `1 + 2^(...)` 中的 `1.0`；经过
/// bufferize/canonicalize 后 APFloat 编码不一定逐位相同，所以按容差比较。
bool isFloatConstantNear(Value v, double expected, double tol = 1e-4) {
  APFloat ap(0.0);
  if (!matchPattern(v, m_ConstantFloat(&ap)))
    return false;
  return std::abs(ap.convertToDouble() - expected) <= tol;
}

/// 识别规格 3.3 的 Fast-GELU 代数形式：
///   GELU(x) ≈ x / (1 + 2^(-2.455492 * x))
/// 对应 IR：
///   yield(divf(x, addf(exp2(mulf(x, c)), 1)))   其中 c ≈ -2.455492
/// 系数 -2.455492 写死在 CUTLASS kernel 的 epilogue 里，这里只做校验、不提取
/// 任何参数。`x` 必须就是 block 参数 `%in`（matmul 结果的一个元素）：带缩放
/// （`%in * cst`）或混入 `%out` 的写法都不融合，因为外部调用固定传
/// alpha = 1.0、beta = 0.0。
bool matchFastGeluAlgebraic(Value yielded, Value blockInput) {
  auto div = yielded.getDefiningOp<arith::DivFOp>();
  if (!div)
    return false;

  Value numer = div.getLhs();
  Value denom = div.getRhs();

  auto add = denom.getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;

  Value expCandidate;
  if (isFloatConstantNear(add.getRhs(), 1.0))
    expCandidate = add.getLhs();
  else if (isFloatConstantNear(add.getLhs(), 1.0))
    expCandidate = add.getRhs();
  else
    return false;

  auto exp2 = expCandidate.getDefiningOp<math::Exp2Op>();
  if (!exp2)
    return false;

  auto mul = exp2.getOperand().getDefiningOp<arith::MulFOp>();
  if (!mul)
    return false;

  // mulf 的一侧必须是 Fast-GELU 常数，另一侧是 `x`。
  Value x;
  if (isFloatConstantNear(mul.getRhs(), -2.455492))
    x = mul.getLhs();
  else if (isFloatConstantNear(mul.getLhs(), -2.455492))
    x = mul.getRhs();
  else
    return false;

  // Fast-GELU 是 `x / (1 + 2^(c*x))`：分子必须是同一个 `x`，且就是 `%in`。
  return numer == x && x == blockInput;
}

/// 判断 `linalg.generic` 的 region 是否正好是 CUTLASS kernel 实现的
/// Fast-GELU，即 D = FastGELU(A @ B)。
///
/// 只接受 Fast-GELU：erf / tanh 形式的 GELU 不融合，否则会被替换成 kernel
/// 的 Fast-GELU 近似，改变数值结果。
///
/// 下面是 tensor 上的示例（test/fuse_pattern.mlir）。本 pass 运行在
/// One-Shot Bufferize 之后，届时 ins/outs 变成 memref、generic 没有返回值，
/// region 内容不变。#map = affine_map<(i, j) -> (i, j)>。
///
///   %r = linalg.generic {indexing_maps = [#map, #map],
///                        iterator_types = ["parallel", "parallel"]}
///       ins(%mm : tensor<128x128xf32>) outs(%empty : tensor<128x128xf32>) {
///   ^bb0(%in: f32, %out: f32):
///     %c = arith.constant -2.455492 : f32
///     %one = arith.constant 1.0 : f32
///     %t = arith.mulf %in, %c : f32
///     %e = math.exp2 %t : f32
///     %d = arith.addf %e, %one : f32
///     %y = arith.divf %in, %d : f32
///     linalg.yield %y : f32
///   } -> tensor<128x128xf32>
bool matchGeluBody(linalg::GenericOp generic) {
  Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return false;
  Block &block = region.front();
  // bufferize 后的一元逐元素 generic：block 参数为 (input, init/out)。
  if (block.getNumArguments() < 1)
    return false;

  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;

  return matchFastGeluAlgebraic(yield.getOperand(0), block.getArgument(0));
}

/// 融合 kernel 从零开始计算 `A @ B`，而 `linalg.matmul` 是在输出 buffer
/// 原有内容上累加。只有当该 buffer 是函数内的 `memref.alloc`、在 matmul 之前
/// 被 `linalg.fill 0.0` 清零、且没有其他观察者时，融合才等价。返回需要与
/// matmul 一起删除的 `linalg.fill`，不满足条件则返回空。
linalg::FillOp findZeroInitOfLocalAcc(linalg::MatmulOp matmul,
                                      linalg::GenericOp activation) {
  Value acc = matmul.getOutputs()[0];
  if (!acc.getDefiningOp<memref::AllocOp>())
    return linalg::FillOp();

  Block *block = matmul->getBlock();
  if (activation->getBlock() != block ||
      !matmul->isBeforeInBlock(activation) ||
      activation.getOutputs()[0] == acc)
    return linalg::FillOp();

  linalg::FillOp fill;
  for (Operation *user : acc.getUsers()) {
    if (user == matmul.getOperation() || user == activation.getOperation() ||
        isa<memref::DeallocOp>(user))
      continue;
    auto candidate = dyn_cast<linalg::FillOp>(user);
    if (!candidate || fill || candidate.getOutputs()[0] != acc ||
        candidate->getBlock() != block ||
        !candidate->isBeforeInBlock(matmul) ||
        !matchPattern(candidate.getInputs()[0], m_AnyZeroFloat()))
      return linalg::FillOp();
    fill = candidate;
  }
  return fill;
}

/// 结构检查 + GELU 语义识别。
/// 仅当 `generic` 是 bufferize 后、全 parallel、identity map 的一元逐元素
/// op，读取 `expectedInput`，且 region 是 Fast-GELU 时返回 true。
bool isElementwiseGeluOf(linalg::GenericOp generic, Value expectedInput) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;
  if (generic.getInputs()[0] != expectedInput)
    return false;
  // tensor 语义的 generic 仍有 SSA 结果；这里要求已 bufferize 的形式。
  if (generic.getNumResults() != 0)
    return false;

  for (utils::IteratorType iter : generic.getIteratorTypesArray())
    if (iter != utils::IteratorType::parallel)
      return false;

  for (AffineMap map : generic.getIndexingMapsArray())
    if (!map.isIdentity())
      return false;

  // 元素类型必须是 runtime 能接受的浮点类型（emit 路径目前只支持 f32
  // memref，这里提前拒绝）。
  auto inType = dyn_cast<MemRefType>(expectedInput.getType());
  if (!inType || !inType.getElementType().isF32())
    return false;
  auto outType = dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
  if (!outType || !outType.getElementType().isF32())
    return false;
  if (!inType.getLayout().isIdentity() || !outType.getLayout().isIdentity())
    return false;

  return matchGeluBody(generic);
}

class FuseGemmGeluPass
    : public PassWrapper<FuseGemmGeluPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseGemmGeluPass)

  StringRef getArgument() const override { return "fuse-gemm-gelu"; }
  StringRef getDescription() const override {
    return "Fuse a bufferized linalg.matmul + GELU linalg.generic into a "
           "call to the external CUTLASS fused GEMM+GELU kernel "
           "(@cutlass_fused_gemm_gelu)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    arith::ArithDialect, math::MathDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  // 作用在 module 上：改写时要往 module 里插入外部函数声明，函数级 pass
  // 不允许修改 module。
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<FusedGemmGeluMatch> matches;
    module.walk([&](linalg::MatmulOp matmulOp) {
      // 只处理 memref 形式（bufferize 之后）的 matmul：tensor 语义的 matmul
      // 还有返回值；bufferize 后的 matmul 原地写入 `outs`，没有返回值。
      if (matmulOp.getNumResults() != 0)
        return;
      if (matmulOp.getNumDpsInputs() != 2 || matmulOp.getNumDpsInits() != 1)
        return;

      // 提前检查 matmul 操作数的类型/布局，避免匹配成功后又被
      // emitExternalFusedCall 拒绝。
      for (Value operand :
           {matmulOp.getInputs()[0], matmulOp.getInputs()[1],
            matmulOp.getOutputs()[0]}) {
        auto ty = dyn_cast<MemRefType>(operand.getType());
        if (!ty || !ty.getElementType().isF32() || !ty.getLayout().isIdentity())
          return;
      }

      Value matmulOutput = matmulOp.getOutputs()[0];
      linalg::GenericOp matchedActivation;
      for (Operation *user : matmulOutput.getUsers()) {
        auto generic = dyn_cast<linalg::GenericOp>(user);
        if (!generic || !isElementwiseGeluOf(generic, matmulOutput))
          continue;
        if (matchedActivation)
          return;
        matchedActivation = generic;
      }
      if (!matchedActivation)
        return;

      // 累加 buffer 的其他使用者都在这里检查过，删除 matmul 不会改变任何
      // 其他 op 观察到的值。
      linalg::FillOp zeroInit =
          findZeroInitOfLocalAcc(matmulOp, matchedActivation);
      if (!zeroInit)
        return;
      matches.push_back({matmulOp, matchedActivation, zeroInit});
    });

    if (matches.empty())
      return;

    IRRewriter rewriter(&getContext());
    for (const FusedGemmGeluMatch &match : matches) {
      if (failed(emitExternalFusedCall(rewriter, match))) {
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

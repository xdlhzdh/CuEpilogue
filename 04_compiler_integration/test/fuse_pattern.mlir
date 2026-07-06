// Input for run_tests.sh: a tensor-level linalg.matmul immediately
// followed by an elementwise linalg.generic modeling a lowered Fast-GELU
// activation (spec 3.4: "识别到相连的 linalg.matmul 与对应的非线性激活 Ops").
// After -one-shot-bufferize + -fuse-gemm-gelu this pair should collapse
// into a single call to @cutlass_fused_gemm_gelu.
func.func @fused_gemm_gelu(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %matmul = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  %empty = tensor.empty() : tensor<128x128xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul : tensor<128x128xf32>) outs(%empty : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      // Fast-GELU algebraic approximation (spec 3.3), expressed with plain
      // arith/math ops as a StableHLO/Linalg lowering would emit it. The
      // fusion pass matches this structurally (elementwise, all-parallel)
      // without inspecting these ops.
      %c = arith.constant -2.455492 : f32
      %t = arith.mulf %in, %c : f32
      %e = math.exp2 %t : f32
      %one = arith.constant 1.0 : f32
      %r = arith.addf %e, %one : f32
      %p = arith.divf %in, %r : f32
      linalg.yield %p : f32
    } -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}

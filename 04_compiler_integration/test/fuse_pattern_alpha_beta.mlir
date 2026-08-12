// Alpha/beta epilogue folded into the activation generic (run by run_tests.sh):
//   yield FastGELU(2.0 * matmul_out + 0.5 * init)
// After bufferize + fuse-gemm-gelu the external call should pass alpha=2.0,
// beta=0.5 to @cutlass_fused_gemm_gelu (not the identity 1.0 / 0.0 defaults).
func.func @fused_gemm_gelu_alpha_beta(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %matmul = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %empty = tensor.empty() : tensor<64x64xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %alpha_cst = arith.constant 2.0 : f32
      %beta_cst = arith.constant 0.5 : f32
      %scaled_in = arith.mulf %in, %alpha_cst : f32
      %scaled_out = arith.mulf %out, %beta_cst : f32
      %linear = arith.addf %scaled_in, %scaled_out : f32
      %c = arith.constant -2.455492 : f32
      %t = arith.mulf %linear, %c : f32
      %e = math.exp2 %t : f32
      %one = arith.constant 1.0 : f32
      %r = arith.addf %e, %one : f32
      %p = arith.divf %linear, %r : f32
      linalg.yield %p : f32
    } -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

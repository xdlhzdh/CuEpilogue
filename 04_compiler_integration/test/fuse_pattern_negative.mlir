// Cases the fusion must reject (run by run_tests.sh); every matmul and
// generic here must survive -fuse-gemm-gelu unchanged.
#map = affine_map<(i, j) -> (i, j)>

// linalg.matmul accumulates into %init (a caller-visible buffer with unknown
// contents), but the fused kernel computes only A @ B.
func.func @nonzero_accumulator(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %mm = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %empty = tensor.empty() : tensor<64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.constant -2.455492 : f32
    %one = arith.constant 1.0 : f32
    %t = arith.mulf %in, %c : f32
    %e = math.exp2 %t : f32
    %d = arith.addf %e, %one : f32
    %y = arith.divf %in, %d : f32
    linalg.yield %y : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// FastGELU(2.0 * in): scaled GELU input; the external call always passes
// alpha = 1.0, so only plain FastGELU(A @ B) is fused.
func.func @scaled_input(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %acc_empty = tensor.empty() : tensor<64x64xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%acc_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
  %empty = tensor.empty() : tensor<64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %alpha = arith.constant 2.0 : f32
    %c = arith.constant -2.455492 : f32
    %one = arith.constant 1.0 : f32
    %x = arith.mulf %in, %alpha : f32
    %t = arith.mulf %x, %c : f32
    %e = math.exp2 %t : f32
    %d = arith.addf %e, %one : f32
    %y = arith.divf %x, %d : f32
    linalg.yield %y : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// FastGELU(2.0 * in + 0.5 * out): GELU input mixes in the old D value; the
// external call always passes beta = 0.0.
func.func @linear_with_out(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>, %D: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %acc_empty = tensor.empty() : tensor<64x64xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%acc_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%D : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %alpha = arith.constant 2.0 : f32
    %beta = arith.constant 0.5 : f32
    %scaled_in = arith.mulf %in, %alpha : f32
    %scaled_out = arith.mulf %out, %beta : f32
    %x = arith.addf %scaled_in, %scaled_out : f32
    %c = arith.constant -2.455492 : f32
    %t = arith.mulf %x, %c : f32
    %e = math.exp2 %t : f32
    %one = arith.constant 1.0 : f32
    %d = arith.addf %e, %one : f32
    %y = arith.divf %x, %d : f32
    linalg.yield %y : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// FastGELU(in) + 0.5 * out: the extra term sits outside the activation.
func.func @beta_outside_gelu(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>, %D: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %acc_empty = tensor.empty() : tensor<64x64xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%acc_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%D : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.constant -2.455492 : f32
    %one = arith.constant 1.0 : f32
    %half = arith.constant 0.5 : f32
    %t = arith.mulf %in, %c : f32
    %e = math.exp2 %t : f32
    %d = arith.addf %e, %one : f32
    %g = arith.divf %in, %d : f32
    %bo = arith.mulf %out, %half : f32
    %y = arith.addf %g, %bo : f32
    linalg.yield %y : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// Exact erf GELU: not fused, the kernel only implements Fast-GELU.
func.func @erf_gelu(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %acc_empty = tensor.empty() : tensor<64x64xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%acc_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
  %empty = tensor.empty() : tensor<64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %half = arith.constant 0.5 : f32
    %rsqrt2 = arith.constant 0.70710678 : f32
    %one = arith.constant 1.0 : f32
    %s = arith.mulf %in, %rsqrt2 : f32
    %e = math.erf %s : f32
    %p = arith.addf %e, %one : f32
    %h = arith.mulf %in, %half : f32
    %y = arith.mulf %h, %p : f32
    linalg.yield %y : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// tanh GELU approximation: not fused, the kernel only implements Fast-GELU.
func.func @tanh_gelu(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %acc_empty = tensor.empty() : tensor<64x64xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%acc_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
  %empty = tensor.empty() : tensor<64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %half = arith.constant 0.5 : f32
    %one = arith.constant 1.0 : f32
    %k0 = arith.constant 0.7978845608 : f32
    %k1 = arith.constant 0.044715 : f32
    %x2 = arith.mulf %in, %in : f32
    %x3 = arith.mulf %x2, %in : f32
    %cx3 = arith.mulf %x3, %k1 : f32
    %inner = arith.addf %in, %cx3 : f32
    %s = arith.mulf %inner, %k0 : f32
    %t = math.tanh %s : f32
    %p = arith.addf %t, %one : f32
    %h = arith.mulf %in, %half : f32
    %y = arith.mulf %h, %p : f32
    linalg.yield %y : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

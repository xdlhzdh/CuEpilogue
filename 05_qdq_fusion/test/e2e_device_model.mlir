// Model for the GPU end-to-end test (e2e_device_test.cu): qdq_pattern.mlir
// in destination-passing style. CMake compiles it ahead of time:
// qdq-opt (fuse-qdq-qgemm-bias-gelu + one-shot-bufferize +
// lower-cutlass-qgemm-to-call) -> LLVM dialect -> mlir-translate -> llc.
//
// Every buffer comes from the caller (the result is written into %dst), so
// the test - playing the inference framework - can hand in device memory for
// all of them. No linalg lowering is run, so the build fails if the pattern is
// not fused. `llvm.emit_c_interface` exports `_mlir_ciface_qdq_gemm_bias_gelu`.
func.func @qdq_gemm_bias_gelu(
    %A: tensor<64x64xi8>, %B: tensor<64x64xi8>, %bias: tensor<64xf32>,
    %sa: f32, %za: i32, %sb: f32, %zb: i32, %sd: f32, %zd: i32,
    %dst: tensor<64x64xi8>)
    -> tensor<64x64xi8> attributes {llvm.emit_c_interface} {
  %a_init = tensor.empty() : tensor<64x64xf32>
  %A_f = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : tensor<64x64xi8>) outs(%a_init : tensor<64x64xf32>) {
    ^bb0(%in: i8, %out: f32):
      %xf = arith.sitofp %in : i8 to f32
      %zaf = arith.sitofp %za : i32 to f32
      %sub = arith.subf %xf, %zaf : f32
      %m = arith.mulf %sub, %sa : f32
      linalg.yield %m : f32
    } -> tensor<64x64xf32>

  %b_init = tensor.empty() : tensor<64x64xf32>
  %B_f = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%B : tensor<64x64xi8>) outs(%b_init : tensor<64x64xf32>) {
    ^bb0(%in: i8, %out: f32):
      %xf = arith.sitofp %in : i8 to f32
      %zbf = arith.sitofp %zb : i32 to f32
      %sub = arith.subf %xf, %zbf : f32
      %m = arith.mulf %sub, %sb : f32
      linalg.yield %m : f32
    } -> tensor<64x64xf32>

  %c_init = tensor.empty() : tensor<64x64xf32>
  %acc = linalg.matmul ins(%A_f, %B_f : tensor<64x64xf32>, tensor<64x64xf32>)
                       outs(%c_init : tensor<64x64xf32>) -> tensor<64x64xf32>

  %bias_init = tensor.empty() : tensor<64x64xf32>
  %biased = linalg.generic {
      indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (j)>,
        affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%acc, %bias : tensor<64x64xf32>, tensor<64xf32>)
      outs(%bias_init : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>

  %gelu_init = tensor.empty() : tensor<64x64xf32>
  %gelu = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%biased : tensor<64x64xf32>) outs(%gelu_init : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %c = arith.constant -2.455492 : f32
      %t = arith.mulf %in, %c : f32
      %e = math.exp2 %t : f32
      %one = arith.constant 1.0 : f32
      %r = arith.addf %e, %one : f32
      %p = arith.divf %in, %r : f32
      linalg.yield %p : f32
    } -> tensor<64x64xf32>

  %result = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%gelu : tensor<64x64xf32>) outs(%dst : tensor<64x64xi8>) {
    ^bb0(%in: f32, %unused: i8):
      %q = arith.divf %in, %sd : f32
      %zdf = arith.sitofp %zd : i32 to f32
      %add = arith.addf %q, %zdf : f32
      %i = arith.fptosi %add : f32 to i8
      linalg.yield %i : i8
    } -> tensor<64x64xi8>
  return %result : tensor<64x64xi8>
}

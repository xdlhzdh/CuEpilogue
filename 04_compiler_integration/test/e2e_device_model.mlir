// Model for the GPU end-to-end test (e2e_device_test.cu). CMake compiles it
// ahead of time: fused-opt (one-shot-bufferize + fuse-gemm-gelu) -> LLVM
// dialect -> mlir-translate -> llc -> object file linked into the test.
//
// Every buffer comes from the caller (destination-passing style: the result
// is written into %out), so the test - playing the inference framework - can
// hand in device memory for all of them. No linalg lowering is run, so the
// build fails if the matmul + Fast-GELU pair is not fused into the external
// call. `llvm.emit_c_interface` exports `_mlir_ciface_gemm_gelu`, which takes
// memref descriptors (StridedMemRefType) by pointer.
func.func @gemm_gelu(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32>
    attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %m = tensor.dim %A, %c0 : tensor<?x?xf32>
  %n = tensor.dim %B, %c1 : tensor<?x?xf32>
  %acc_empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%acc_empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%acc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %c = arith.constant -2.455492 : f32
      %t = arith.mulf %in, %c : f32
      %e = math.exp2 %t : f32
      %one = arith.constant 1.0 : f32
      %r = arith.addf %e, %one : f32
      %p = arith.divf %in, %r : f32
      linalg.yield %p : f32
    } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

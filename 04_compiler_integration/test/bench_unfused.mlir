// Stage 4 end-to-end latency benchmark - "unfused" baseline (spec 3.4:
// "基础指令展开", i.e. what the IR would lower to WITHOUT the
// `--fuse-gemm-gelu` pattern match). Same matmul + elementwise Fast-GELU
// shape as fuse_pattern.mlir, but left as plain linalg ops so
// `--convert-linalg-to-loops` turns them into ordinary scalar loops
// instead of an external CUTLASS call. See bench_fused.mlir for the
// fused counterpart and why the call is hand-written rather than routed
// through the real fusion pass.
func.func private @rtclock() -> f64
func.func private @printF64(f64)
func.func private @printNewline()

func.func @gemm_gelu_unfused(%A: memref<128x128xf32>, %B: memref<128x128xf32>,
                              %Init: memref<128x128xf32>, %D: memref<128x128xf32>) {
  linalg.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>) outs(%Init : memref<128x128xf32>)
  %c = arith.constant -2.455492 : f32
  %one = arith.constant 1.0 : f32
  linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%Init : memref<128x128xf32>) outs(%D : memref<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %t = arith.mulf %in, %c : f32
      %e = math.exp2 %t : f32
      %r = arith.addf %e, %one : f32
      %p = arith.divf %in, %r : f32
      linalg.yield %p : f32
  }
  return
}

func.func @main() {
  %n_iters = arith.constant 20 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst0 = arith.constant 0.0 : f32
  %cst1 = arith.constant 1.0 : f32

  %A = memref.alloc() : memref<128x128xf32>
  %B = memref.alloc() : memref<128x128xf32>
  %Init = memref.alloc() : memref<128x128xf32>
  %D = memref.alloc() : memref<128x128xf32>
  linalg.fill ins(%cst1 : f32) outs(%A : memref<128x128xf32>)
  linalg.fill ins(%cst1 : f32) outs(%B : memref<128x128xf32>)
  linalg.fill ins(%cst0 : f32) outs(%Init : memref<128x128xf32>)
  linalg.fill ins(%cst0 : f32) outs(%D : memref<128x128xf32>)

  %t0 = func.call @rtclock() : () -> f64
  scf.for %i = %c0 to %n_iters step %c1 {
    %ic = arith.index_cast %i : index to i32
    %fc = arith.sitofp %ic : i32 to f32
    memref.store %fc, %A[%c0, %c0] : memref<128x128xf32>
    func.call @gemm_gelu_unfused(%A, %B, %Init, %D)
      : (memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>) -> ()
  }
  %t1 = func.call @rtclock() : () -> f64
  %dt = arith.subf %t1, %t0 : f64
  func.call @printF64(%dt) : (f64) -> ()
  func.call @printNewline() : () -> ()

  memref.dealloc %A : memref<128x128xf32>
  memref.dealloc %B : memref<128x128xf32>
  memref.dealloc %Init : memref<128x128xf32>
  memref.dealloc %D : memref<128x128xf32>
  return
}

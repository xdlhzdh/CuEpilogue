#pragma once
// Stage 8: Sm90 FlashAttention forward (hand-written CuTe mainloop).
//
//   O[b, i, h, :] = sum_j softmax_j(scale * Q[b, i, h, :] . K[b, j, h, :]) V[b, j, h, :]
//   LSE[b, h, i]  = log sum_j exp(scale * Q[b, i, h, :] . K[b, j, h, :])
//
// Q/O are [batch, seqlen_q, heads, head_dim], K/V are [batch, seqlen_k, heads,
// head_dim], all fp16 and contiguous. LSE is fp32 [batch, heads, seqlen_q].
// With causal = true, query i sees keys j <= i + seqlen_k - seqlen_q
// (bottom-right aligned); rows that see no key get O = 0 and LSE = +inf.
// Requires an sm_90a device; the kernel is compiled for sm_90a only.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cu_epilogue {

// How each consumer warpgroup orders GEMM1, softmax and GEMM2.
//   kSerial          per block: GEMM1 -> wait -> softmax -> GEMM2 -> wait.
//   kIntraWGOverlap  software-pipelined mainloop: GEMM1 of block n and GEMM2
//                    of block n-1 are in flight while the warpgroup runs the
//                    softmax of block n (prologue / steady state / drain).
//   kPingPong        kIntraWGOverlap plus named-barrier turn taking between
//                    the two consumer warpgroups, so one issues its GEMMs
//                    while the other runs softmax.
enum class FlashAttnSchedule { kSerial, kIntraWGOverlap, kPingPong };

struct FlashAttnParams {
  const __half *q = nullptr;
  const __half *k = nullptr;
  const __half *v = nullptr;
  __half *o = nullptr;
  float *lse = nullptr;  // optional, may be nullptr
  int batch = 0;
  int seqlen_q = 0;
  int seqlen_k = 0;
  int num_heads = 0;
  int head_dim = 0;  // 64 or 128
  float softmax_scale = 0.0f;
  bool causal = false;
  FlashAttnSchedule schedule = FlashAttnSchedule::kPingPong;
};

// Returns false when the arguments are unsupported or the launch fails.
bool FlashAttnFwdSm90(const FlashAttnParams &params, cudaStream_t stream = nullptr);

} // namespace cu_epilogue

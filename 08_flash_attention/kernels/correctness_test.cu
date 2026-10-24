// Stage 8 correctness: Sm90 FlashAttention forward vs a double-precision CPU
// reference, checking both O and LSE. Covers both head dims, causal and
// non-causal, sequence lengths that are not multiples of the 128 tile, and
// causal with seqlen_q > seqlen_k (fully masked rows: O = 0, LSE = +inf).
// Skips (exit 0) when the GPU compute capability is < 90.
#include "flash_attn_fwd_sm90.cuh"
#include "cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using namespace cu_epilogue;

namespace {

bool DeviceSupportsSm90() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return false;
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess)
    return false;
  return major >= 9;
}

struct Case {
  int batch, heads, seqlen_q, seqlen_k, head_dim;
  bool causal;
};

const char *ScheduleName(FlashAttnSchedule s) {
  switch (s) {
  case FlashAttnSchedule::kSerial:
    return "serial";
  case FlashAttnSchedule::kIntraWGOverlap:
    return "overlap";
  case FlashAttnSchedule::kPingPong:
    return "pingpong";
  }
  return "?";
}

// Inputs are [B, S, H, D] fp16 (passed here as their exact float values).
void Reference(const Case &c, float scale, const std::vector<float> &q,
               const std::vector<float> &k, const std::vector<float> &v,
               std::vector<float> &o, std::vector<float> &lse) {
  const int D = c.head_dim, H = c.heads;
  const int diag = c.seqlen_k - c.seqlen_q;
  std::vector<double> s(c.seqlen_k);
  for (int b = 0; b < c.batch; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < c.seqlen_q; ++i) {
        const float *qi = &q[((size_t(b) * c.seqlen_q + i) * H + h) * D];
        const int j_end = c.causal ? std::min(c.seqlen_k, i + diag + 1) : c.seqlen_k;
        double m = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < j_end; ++j) {
          const float *kj = &k[((size_t(b) * c.seqlen_k + j) * H + h) * D];
          double dot = 0.0;
          for (int d = 0; d < D; ++d)
            dot += double(qi[d]) * double(kj[d]);
          s[j] = dot * scale;
          m = std::max(m, s[j]);
        }
        float *oi = &o[((size_t(b) * c.seqlen_q + i) * H + h) * D];
        float &lse_i = lse[(size_t(b) * H + h) * c.seqlen_q + i];
        if (j_end <= 0) {
          for (int d = 0; d < D; ++d)
            oi[d] = 0.0f;
          lse_i = std::numeric_limits<float>::infinity();
          continue;
        }
        double sum = 0.0;
        std::vector<double> acc(D, 0.0);
        for (int j = 0; j < j_end; ++j) {
          const double p = std::exp(s[j] - m);
          sum += p;
          const float *vj = &v[((size_t(b) * c.seqlen_k + j) * H + h) * D];
          for (int d = 0; d < D; ++d)
            acc[d] += p * double(vj[d]);
        }
        for (int d = 0; d < D; ++d)
          oi[d] = float(acc[d] / sum);
        lse_i = float(m + std::log(sum));
      }
    }
  }
}

void FillRandom(std::vector<__half> &dev_vals, std::vector<float> &host_vals,
                uint32_t seed) {
  uint32_t state = seed;
  for (size_t i = 0; i < dev_vals.size(); ++i) {
    state = state * 1664525u + 1013904223u;
    const float x = float((state >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
    dev_vals[i] = __float2half(x);
    host_vals[i] = __half2float(dev_vals[i]);
  }
}

bool RunCase(const Case &c, FlashAttnSchedule schedule) {
  const size_t nq = size_t(c.batch) * c.seqlen_q * c.heads * c.head_dim;
  const size_t nk = size_t(c.batch) * c.seqlen_k * c.heads * c.head_dim;
  const size_t nlse = size_t(c.batch) * c.heads * c.seqlen_q;
  const float scale = 1.0f / std::sqrt(float(c.head_dim));

  std::vector<__half> hq(nq), hk(nk), hv(nk), ho(nq);
  std::vector<float> fq(nq), fk(nk), fv(nk), ref_o(nq), ref_lse(nlse), got_o(nq),
      got_lse(nlse);
  FillRandom(hq, fq, 1u);
  FillRandom(hk, fk, 2u);
  FillRandom(hv, fv, 3u);
  Reference(c, scale, fq, fk, fv, ref_o, ref_lse);

  __half *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
  float *dlse = nullptr;
  CUDA_CHECK(cudaMalloc(&dq, nq * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dk, nk * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dv, nk * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dout, nq * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dlse, nlse * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), nq * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), nk * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), nk * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dout, 0xFF, nq * sizeof(__half)));  // NaN poison
  CUDA_CHECK(cudaMemset(dlse, 0xFF, nlse * sizeof(float)));

  FlashAttnParams p;
  p.q = dq;
  p.k = dk;
  p.v = dv;
  p.o = dout;
  p.lse = dlse;
  p.batch = c.batch;
  p.seqlen_q = c.seqlen_q;
  p.seqlen_k = c.seqlen_k;
  p.num_heads = c.heads;
  p.head_dim = c.head_dim;
  p.softmax_scale = scale;
  p.causal = c.causal;
  p.schedule = schedule;
  const bool launched = FlashAttnFwdSm90(p, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(ho.data(), dout, nq * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_lse.data(), dlse, nlse * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nq; ++i)
    got_o[i] = __half2float(ho[i]);

  // P is rounded to fp16 before GEMM2 and O is stored as fp16.
  const float o_tol = 1e-2f, lse_tol = 1e-3f;
  float max_o = 0.0f, max_lse = 0.0f;
  size_t bad_o = 0, bad_lse = 0;
  for (size_t i = 0; i < nq; ++i) {
    const float err = std::fabs(got_o[i] - ref_o[i]);
    if (!(err <= o_tol))
      ++bad_o;
    if (err > max_o || err != err)
      max_o = err;
  }
  for (size_t i = 0; i < nlse; ++i) {
    if (std::isinf(ref_lse[i])) {
      if (!(std::isinf(got_lse[i]) && got_lse[i] > 0))
        ++bad_lse;
      continue;
    }
    const float err = std::fabs(got_lse[i] - ref_lse[i]);
    if (!(err <= lse_tol))
      ++bad_lse;
    if (err > max_lse || err != err)
      max_lse = err;
  }

  const bool pass = launched && bad_o == 0 && bad_lse == 0;
  std::printf("[flash_attn_fwd_sm90 %-8s] B=%d H=%d Sq=%d Sk=%d D=%d causal=%d "
              "launched=%d max_abs_O=%.3e max_abs_LSE=%.3e bad_O=%zu bad_LSE=%zu -> %s\n",
              ScheduleName(schedule), c.batch, c.heads, c.seqlen_q, c.seqlen_k,
              c.head_dim, int(c.causal), int(launched), max_o, max_lse, bad_o, bad_lse,
              pass ? "PASS" : "FAIL");
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dout);
  cudaFree(dlse);
  return pass;
}

} // namespace

int main() {
  if (!DeviceSupportsSm90()) {
    std::printf("SKIP: Sm90 FlashAttention correctness requires Hopper (cc >= 90); "
                "current device is below that or unavailable. Compile/SASS checks "
                "still apply.\n");
    return 0;
  }

  const Case cases[] = {
      {2, 3, 200, 333, 64, false},
      {2, 3, 200, 333, 64, true},
      {1, 2, 256, 256, 128, false},
      {1, 2, 256, 256, 128, true},
      {1, 2, 300, 129, 128, true},  // seqlen_q > seqlen_k: leading rows fully masked
      {1, 1, 77, 1, 64, false},
      {1, 1, 77, 1, 64, true},
      {1, 2, 640, 1000, 64, true},  // long enough for unmasked steady-state blocks
  };
  const FlashAttnSchedule schedules[] = {FlashAttnSchedule::kSerial,
                                         FlashAttnSchedule::kIntraWGOverlap,
                                         FlashAttnSchedule::kPingPong};
  bool all_ok = true;
  for (FlashAttnSchedule s : schedules)
    for (const Case &c : cases)
      all_ok &= RunCase(c, s);
  std::printf(all_ok ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
  return all_ok ? 0 : 1;
}

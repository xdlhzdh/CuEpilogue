#pragma once
// Hand-written Sm90 FlashAttention forward mainloop:
//   GEMM1  S = Q * K^T          (WGMMA, both operands from SMEM)
//   online softmax on S         (registers, quad shuffles)
//   GEMM2  O += P * V           (WGMMA, P from registers, V from SMEM)
// K and V tiles stream through two independent PipelineTmaAsync rings
// (Stages-deep), filled by TMA from a producer warpgroup. Q is loaded once by
// TMA and signalled through its own transaction barrier.
//
// Global tensors are [batch, seqlen, heads, head_dim] (head_dim contiguous) and
// are described to TMA as rank-4 (seqlen, head_dim, heads, batch). Out-of-range
// rows are zero-filled by TMA; the score mask handles the padded K columns.

#include "flash_attn_fwd_sm90.cuh"
#include "flash_attn_softmax.cuh"

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

namespace cu_epilogue {
namespace fmha {

using namespace cute;

// Issues one warpgroup-wide WGMMA batch (all k-blocks of the tile) and commits
// it. kWait >= 0 waits until at most kWait batches are still in flight;
// kWait == -1 returns with the batch in flight, and the caller must
// warpgroup_wait<>() before touching tCrC (or overwriting a register tCrA).
template <bool kZeroInit, int kWait, class TiledMma, class TensorA,
          class TensorB, class TensorC>
CUTE_DEVICE void wgmma_gemm(TiledMma &tiled_mma, TensorA const &tCrA,
                            TensorB const &tCrB, TensorC &tCrC) {
  constexpr bool kIsRS = !cute::is_base_of<GMMA::DescriptorIterator,
                                           typename TiledMma::FrgTypeA>::value;
  if constexpr (kIsRS) {
    warpgroup_fence_operand(const_cast<TensorA &>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  if constexpr (kZeroInit) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  }
  CUTE_UNROLL
  for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
    cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_commit_batch();
  if constexpr (kWait >= 0) {
    warpgroup_wait<kWait>();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (kIsRS) {
    warpgroup_fence_operand(const_cast<TensorA &>(tCrA));
  }
}

template <int kHeadDim_, int kBlockM_ = 128, int kBlockN_ = 128,
          int kStages_ = 2, bool kIsCausal_ = false,
          FlashAttnSchedule kSchedule_ = FlashAttnSchedule::kPingPong>
struct FusedAttentionMainloop {
  using Element = cutlass::half_t;
  using ElementAccum = float;

  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kStages = kStages_;
  static constexpr bool kIsCausal = kIsCausal_;
  static constexpr FlashAttnSchedule kSchedule = kSchedule_;
  static constexpr bool kIntraWGOverlap = kSchedule != FlashAttnSchedule::kSerial;
  static constexpr bool kPingPong = kSchedule == FlashAttnSchedule::kPingPong;

  static_assert(kHeadDim % 64 == 0, "head_dim must be a multiple of 64 (SW128 atom)");
  static_assert(kBlockM % 64 == 0, "one WGMMA M=64 tile per consumer warpgroup");
  static_assert(kBlockN % 16 == 0 && kBlockN <= 256, "GEMM1 N must fit one WGMMA atom");
  static_assert(kStages >= 2, "need at least double buffering");

  static constexpr int kNumMmaWarpGroups = kBlockM / 64;
  static constexpr int kNumMmaThreads = kNumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int kNumThreads = kNumMmaThreads + cutlass::NumThreadsPerWarpGroup;
  static_assert(!kPingPong || kNumMmaWarpGroups == 2,
                "ping-pong turn taking is written for two consumer warpgroups");

  // User named-barrier ids (CUTLASS offsets them past its reserved ids).
  // kBarrierEpilogue: all consumers before reusing sQ for O.
  // kBarrierTurnWG1/2: ping-pong "your turn to issue GEMMs" for WG1 / WG2.
  static constexpr uint32_t kBarrierEpilogue = 0;
  static constexpr uint32_t kBarrierTurnWG1 = 1;
  static constexpr uint32_t kBarrierTurnWG2 = 2;

  // GEMM1: (M, N, K) = (kBlockM, kBlockN, kHeadDim)
  using TileShapeQK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  // GEMM2: (M, N, K) = (kBlockM, kHeadDim, kBlockN)
  using TileShapePV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
  using AtomLayoutMNK = Layout<Shape<Int<kNumMmaWarpGroups>, _1, _1>>;

  using TiledMmaQK = decltype(make_tiled_mma(
      GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>(),
      AtomLayoutMNK{}));
  // V is stored [seq_k, head_dim]; as the B operand of P*V (N = head_dim,
  // K = seq_k) it is MN-major, which 16-bit WGMMA supports directly.
  using TiledMmaPV = decltype(make_tiled_mma(
      GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapePV,
                           GMMA::Major::K, GMMA::Major::MN>(),
      AtomLayoutMNK{}));

  // Accumulator rows owned by one thread: 2 per 64-row WGMMA tile.
  static constexpr int kRowsPerThread = 2;

  // 128B-swizzled K-major atom: 8 rows x 64 halves.
  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>;
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));
  using SmemLayoutV = SmemLayoutK;
  // Same bytes as sV, indexed (head_dim, seq_k, stage).
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{},
      make_ordered_layout(Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
                          Step<_2, _1, _3>{})));
  // The O tile is staged through the Q buffer once GEMM1 is finished with it.
  using SmemLayoutO = SmemLayoutQ;

  static constexpr uint32_t kTmaBytesQ =
      kBlockM * kHeadDim * uint32_t(sizeof(Element));
  static constexpr uint32_t kTmaBytesKV =
      kBlockN * kHeadDim * uint32_t(sizeof(Element));

  using PipelineKV = cutlass::PipelineTmaAsync<kStages>;
  using PipelineState = typename PipelineKV::PipelineState;
  using ClusterShape = Shape<_1, _1, _1>;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<1024> {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 1024> smem_q;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 1024> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 1024> smem_v;
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16> {
      typename PipelineKV::SharedStorage pipeline_k;
      typename PipelineKV::SharedStorage pipeline_v;
      uint64_t barrier_q;
    } pipelines;
  };

  // (seqlen, head_dim, heads, batch)
  using GmemShape = Shape<int32_t, int32_t, int32_t, int32_t>;
  using GmemStride = Stride<int64_t, _1, int64_t, int64_t>;
  using GmemLayout = Layout<GmemShape, GmemStride>;

  using TMA_Q = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const *>(nullptr)), GmemLayout{}),
      SmemLayoutQ{}, select<0, 2>(TileShapeQK{}), _1{}));
  using TMA_KV = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const *>(nullptr)), GmemLayout{}),
      SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShapeQK{}), _1{}));

  struct Params {
    TMA_Q tma_load_q;
    TMA_KV tma_load_k;
    TMA_KV tma_load_v;
    GmemLayout layout_q;
    GmemLayout layout_k;
    int seqlen_q;
    int seqlen_k;
    float softmax_scale;
    float softmax_scale_log2;
  };

  static GmemLayout make_gmem_layout(int batch, int seqlen, int heads) {
    int64_t const row_stride = int64_t(heads) * kHeadDim;
    return make_layout(
        make_shape(seqlen, int32_t(kHeadDim), heads, batch),
        make_stride(row_stride, _1{}, int64_t(kHeadDim), row_stride * seqlen));
  }

  static Params to_underlying_arguments(Element const *q, Element const *k,
                                        Element const *v, int batch,
                                        int seqlen_q, int seqlen_k, int heads,
                                        float softmax_scale) {
    GmemLayout layout_q = make_gmem_layout(batch, seqlen_q, heads);
    GmemLayout layout_k = make_gmem_layout(batch, seqlen_k, heads);
    TMA_Q tma_q = make_tma_copy(SM90_TMA_LOAD{},
                                make_tensor(make_gmem_ptr(q), layout_q),
                                SmemLayoutQ{}, select<0, 2>(TileShapeQK{}), _1{});
    TMA_KV tma_k = make_tma_copy(SM90_TMA_LOAD{},
                                 make_tensor(make_gmem_ptr(k), layout_k),
                                 SmemLayoutK{}(_, _, _0{}),
                                 select<1, 2>(TileShapeQK{}), _1{});
    TMA_KV tma_v = make_tma_copy(SM90_TMA_LOAD{},
                                 make_tensor(make_gmem_ptr(v), layout_k),
                                 SmemLayoutV{}(_, _, _0{}),
                                 select<1, 2>(TileShapeQK{}), _1{});
    return {tma_q,    tma_k,    tma_v,         layout_q,
            layout_k, seqlen_q, seqlen_k,      softmax_scale,
            softmax_scale * 1.44269504088896340736f};
  }

  CUTE_DEVICE static void prefetch_tma_descriptors(Params const &params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
  }

  // Number of K/V blocks this Q tile has to visit. With a causal mask, blocks
  // entirely above the (bottom-right aligned) diagonal are skipped; it can be
  // 0 when seqlen_q > seqlen_k and every row of the tile is fully masked.
  CUTE_HOST_DEVICE static int n_block_max(Params const &params, int m_block) {
    int n_blocks = (params.seqlen_k + kBlockN - 1) / kBlockN;
    if constexpr (kIsCausal) {
      int const col_end =
          (m_block + 1) * kBlockM + params.seqlen_k - params.seqlen_q;
      n_blocks = col_end <= 0 ? 0 : min(n_blocks, (col_end + kBlockN - 1) / kBlockN);
    }
    return n_blocks;
  }

  // First K/V block that needs score masking. Every later block needs it
  // too: the seqlen_k tail is the last block, and with a causal mask the
  // diagonal only moves further left of the remaining blocks. A block needs
  // the causal mask iff (n + 1) * kBlockN > m_block * kBlockM + causal_diag + 1.
  CUTE_HOST_DEVICE static int first_masked_block(Params const &params, int m_block) {
    int n = params.seqlen_k / kBlockN;
    if constexpr (kIsCausal) {
      int const lim = m_block * kBlockM + params.seqlen_k - params.seqlen_q + 1;
      n = min(n, lim <= 0 ? 0 : lim / kBlockN);
    }
    return n;
  }

  // Producer: executed by a single thread (the pipeline leader). Issues the Q
  // load once, then one K and one V tile per block, each into the next free
  // stage of its ring.
  CUTE_DEVICE void load(Params const &params, PipelineKV &pipeline_k,
                        PipelineKV &pipeline_v, PipelineState &smem_pipe_write_k,
                        PipelineState &smem_pipe_write_v, SharedStorage &storage,
                        int m_block, int head, int batch, int n_blocks) const {
    Tensor sQ = make_tensor(make_smem_ptr(storage.tensors.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.tensors.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.tensors.smem_v.data()), SmemLayoutV{});

    Tensor mQ = params.tma_load_q.get_tma_tensor(params.layout_q.shape());
    Tensor mK = params.tma_load_k.get_tma_tensor(params.layout_k.shape());
    Tensor mV = params.tma_load_v.get_tma_tensor(params.layout_k.shape());

    // (kBlockM, D) and (kBlockN, D, n_block)
    Tensor gQ = local_tile(mQ(_, _, head, batch), select<0, 2>(TileShapeQK{}),
                           make_coord(m_block, _0{}));
    Tensor gK = local_tile(mK(_, _, head, batch), select<1, 2>(TileShapeQK{}),
                           make_coord(_, _0{}));
    Tensor gV = local_tile(mV(_, _, head, batch), select<1, 2>(TileShapeQK{}),
                           make_coord(_, _0{}));

    auto block_tma_q = params.tma_load_q.get_slice(_0{});
    auto block_tma_k = params.tma_load_k.get_slice(_0{});
    auto block_tma_v = params.tma_load_v.get_slice(_0{});
    Tensor tQgQ = block_tma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K)
    Tensor tQsQ = block_tma_q.partition_D(sQ);
    Tensor tKgK = block_tma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, n_block)
    Tensor tKsK = block_tma_k.partition_D(sK);  // (TMA, TMA_N, TMA_K, stage)
    Tensor tVgV = block_tma_v.partition_S(gV);
    Tensor tVsV = block_tma_v.partition_D(sV);

    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        &storage.pipelines.barrier_q, kTmaBytesQ);
    copy(params.tma_load_q.with(storage.pipelines.barrier_q), tQgQ, tQsQ);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int n_block = 0; n_block < n_blocks; ++n_block) {
      pipeline_k.producer_acquire(smem_pipe_write_k);
      copy(params.tma_load_k.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k)),
           tKgK(_, _, _, n_block), tKsK(_, _, _, smem_pipe_write_k.index()));
      ++smem_pipe_write_k;

      pipeline_v.producer_acquire(smem_pipe_write_v);
      copy(params.tma_load_v.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v)),
           tVgV(_, _, _, n_block), tVsV(_, _, _, smem_pipe_write_v.index()));
      ++smem_pipe_write_v;
    }
  }

  // Consumers: both MMA warpgroups. thread_idx is in [0, kNumMmaThreads).
  // On return tOrO holds the unnormalized output and softmax the running
  // max / sum for every row this thread owns.
  template <class FrgTensorO, class Softmax>
  CUTE_DEVICE void mma(Params const &params, PipelineKV &pipeline_k,
                       PipelineKV &pipeline_v, PipelineState &smem_pipe_read_k,
                       PipelineState &smem_pipe_read_v, FrgTensorO &tOrO,
                       Softmax &softmax, int thread_idx, int m_block,
                       int n_blocks, SharedStorage &storage) const {
    Tensor sQ = make_tensor(make_smem_ptr(storage.tensors.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.tensors.smem_k.data()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(storage.tensors.smem_v.data()), SmemLayoutVt{});

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);   // (MMA, MMA_M, MMA_K)
    Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);   // (MMA, MMA_N, MMA_K, stage)
    Tensor tOrV = thr_mma_pv.partition_fragment_B(sVt);  // (MMA, MMA_N, MMA_K, stage)

    Tensor cS = make_identity_tensor(select<0, 1>(TileShapeQK{}));
    Tensor tScS = thr_mma_qk.partition_C(cS);

    int const seqlen_k = params.seqlen_k;
    int const causal_diag = params.seqlen_k - params.seqlen_q;
    int const row_offset = m_block * kBlockM;

    auto mask_scores = [&](auto &tSrS, int n_block) {
      int const col_offset = n_block * kBlockN;
      bool const need_mask =
          col_offset + kBlockN > seqlen_k ||
          (kIsCausal && col_offset + kBlockN > row_offset + causal_diag + 1);
      if (need_mask) {
        apply_mask<kIsCausal>(tSrS, tScS, row_offset, col_offset, seqlen_k,
                              causal_diag);
      }
    };

    cutlass::arch::ClusterTransactionBarrier::wait(&storage.pipelines.barrier_q, 0);
    if (n_blocks == 0) {
      return;
    }

    float scale_o[Softmax::kRows];

    if constexpr (!kIntraWGOverlap) {
      // Serial: every block runs GEMM1 -> softmax -> GEMM2 back to back, and
      // the warpgroup idles on each warpgroup_wait.
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n_block = 0; n_block < n_blocks; ++n_block) {
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShapeQK{}));
        pipeline_k.consumer_wait(smem_pipe_read_k);
        wgmma_gemm</*kZeroInit=*/true, /*kWait=*/0>(
            tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;

        mask_scores(tSrS, n_block);
        softmax.online_softmax(tSrS, scale_o);
        Softmax::rescale_o(tOrO, scale_o);

        Tensor rP = convert_type<Element>(tSrS);
        Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs(rP.layout()));
        pipeline_v.consumer_wait(smem_pipe_read_v);
        wgmma_gemm</*kZeroInit=*/false, /*kWait=*/0>(
            tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
      }
    } else {
      // Software-pipelined mainloop. Iteration n works on two blocks at once:
      //   S-stage of block n   : GEMM1 S(n) = Q K(n)^T, then softmax -> P(n)
      //   O-stage of block n-1 : GEMM2 O += P(n-1) V(n-1)
      // Both GEMMs are in flight while the warpgroup runs the softmax of
      // block n, so the tensor cores and the MUFU/FMA pipes overlap.
      //
      // O may not be touched while GEMM2 accumulates into it, so the
      // exp(m(n-1) - m(n)) rescale of O computed by softmax(n) is deferred
      // to the next iteration, right before GEMM2(n) is issued.
      //
      //   prologue : S-stage of block 0
      //   mainloop : S-stage of n, O-stage of n-1     for n = 1 .. N-1
      //   drain    : O-stage of block N-1
      Tensor tOrP = make_tensor<Element>(convert_layout_acc_Aregs(
          partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShapeQK{})).layout()));

      turn_init();

      // ---- prologue ----
      {
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShapeQK{}));
        pipeline_k.consumer_wait(smem_pipe_read_k);
        turn_wait();
        wgmma_gemm</*kZeroInit=*/true, /*kWait=*/-1>(
            tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        turn_pass(/*last_turn=*/n_blocks == 1);
        warpgroup_wait<0>();
        warpgroup_fence_operand(tSrS);
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;

        mask_scores(tSrS, 0);
        softmax.online_softmax(tSrS, scale_o);
        convert_into(tSrS, tOrP);
      }

      // ---- steady state ----
      // No branch may sit between warpgroup_wait<1> and <0>: at a control-flow
      // join ptxas drains every in-flight WGMMA, which would serialize GEMM2
      // with the softmax again. So the mask is a compile-time property of the
      // loop body, and masked blocks (always the trailing ones) get their own
      // loop.
      auto steady_state_iteration = [&](int n_block, auto mask_tag) {
        constexpr bool kMask = decltype(mask_tag)::value;
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShapeQK{}));
        pipeline_k.consumer_wait(smem_pipe_read_k);
        turn_wait();
        wgmma_gemm</*kZeroInit=*/true, /*kWait=*/-1>(
            tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        // GEMM1(n) only reads Q/K and writes tSrS, so O can be rescaled
        // to m(n-1) while it runs; GEMM2(n-1) then accumulates on top.
        Softmax::rescale_o(tOrO, scale_o);
        pipeline_v.consumer_wait(smem_pipe_read_v);
        wgmma_gemm</*kZeroInit=*/false, /*kWait=*/-1>(
            tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        turn_pass(/*last_turn=*/n_block == n_blocks - 1);

        // Oldest batch first: GEMM1(n) is done, GEMM2(n-1) may still run.
        warpgroup_wait<1>();
        warpgroup_fence_operand(tSrS);
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;

        if constexpr (kMask) {
          apply_mask<kIsCausal>(tSrS, tScS, row_offset, n_block * kBlockN,
                                seqlen_k, causal_diag);
        }
        softmax.online_softmax(tSrS, scale_o);
        // Pin the softmax results here: otherwise the compiler is free to
        // sink this register-only math below the wait and nothing overlaps.
        // The register fences only bind NVVM; ptxas still hoists the wait
        // over the masked softmax (seen with head_dim 64) unless a warp
        // barrier sits in between.
        warpgroup_fence_operand(tSrS);
        fence_registers(scale_o);
        fence_registers(softmax.row_sum);
        __syncwarp();

        // GEMM2(n-1) must retire before its P registers are overwritten.
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
        convert_into(tSrS, tOrP);
      };

      int const n_unmasked_end = min(first_masked_block(params, m_block), n_blocks);
      int n_block = 1;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; n_block < n_unmasked_end; ++n_block) {
        steady_state_iteration(n_block, cute::false_type{});
      }
      CUTLASS_PRAGMA_NO_UNROLL
      for (; n_block < n_blocks; ++n_block) {
        steady_state_iteration(n_block, cute::true_type{});
      }

      // ---- drain ----
      Softmax::rescale_o(tOrO, scale_o);
      pipeline_v.consumer_wait(smem_pipe_read_v);
      wgmma_gemm</*kZeroInit=*/false, /*kWait=*/0>(
          tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
      pipeline_v.consumer_release(smem_pipe_read_v);
      ++smem_pipe_read_v;
    }
  }

  // Ping-pong turn taking between the two consumer warpgroups (canonical
  // warpgroup 1 and 2). A warpgroup waits for its turn before issuing its
  // GEMMs and hands the turn over right after issuing them, so the other
  // warpgroup's GEMMs queue behind its own while it runs softmax.
  // Turn barrier k completes when warpgroup k syncs on it (128 threads) and
  // the other warpgroup has arrived on it (128 threads).
  // Every warpgroup takes exactly n_blocks turns; WG2's initial arrive gives
  // WG1 the first turn and WG2 skips its last pass so no arrive is left
  // pending when the CTA exits.
  CUTE_DEVICE static void turn_init() {
    if constexpr (kPingPong) {
      if (cutlass::canonical_warp_group_idx() == 2) {
        cutlass::arch::NamedBarrier::arrive(kNumMmaThreads, kBarrierTurnWG1);
      }
    }
  }

  CUTE_DEVICE static void turn_wait() {
    if constexpr (kPingPong) {
      // Literal ids per branch keep ptxas from reserving all 16 barriers.
      if (cutlass::canonical_warp_group_idx() == 1) {
        cutlass::arch::NamedBarrier::sync(kNumMmaThreads, kBarrierTurnWG1);
      } else {
        cutlass::arch::NamedBarrier::sync(kNumMmaThreads, kBarrierTurnWG2);
      }
    }
  }

  CUTE_DEVICE static void turn_pass(bool last_turn) {
    if constexpr (kPingPong) {
      if (cutlass::canonical_warp_group_idx() == 1) {
        cutlass::arch::NamedBarrier::arrive(kNumMmaThreads, kBarrierTurnWG2);
      } else if (!last_turn) {
        cutlass::arch::NamedBarrier::arrive(kNumMmaThreads, kBarrierTurnWG1);
      }
    }
  }
};

} // namespace fmha
} // namespace cu_epilogue

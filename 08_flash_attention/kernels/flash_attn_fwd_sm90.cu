// Stage 8: Sm90 FlashAttention forward kernel.
//
// One CTA = one 128-row Q tile of one (batch, head). Three warpgroups:
//   WG0      producer: one thread issues every TMA load (Q once, then K/V
//            tiles into PipelineTmaAsync rings); setmaxnreg down to 24.
//   WG1, WG2 consumers: each owns 64 Q rows, runs GEMM1 -> online softmax ->
//            GEMM2 per K/V block (serial, software-pipelined, or
//            software-pipelined + ping-pong, see FlashAttnSchedule), then the
//            epilogue; setmaxnreg up to 240.
// Epilogue: O /= row_sum in registers, fp16 O staged into the (now idle) Q
// buffer and written by one TMA store; LSE written directly per row.
#include "flash_attn_fwd_sm90.cuh"
#include "flash_attn_mainloop_sm90.cuh"

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cutlass.h>

#include <cstdint>
#include <cstdio>

namespace cu_epilogue {
namespace fmha {

using namespace cute;

template <class Mainloop_> struct FlashAttnFwdKernel {
  using Mainloop = Mainloop_;
  using Element = typename Mainloop::Element;
  using SharedStorage = typename Mainloop::SharedStorage;
  using GmemLayout = typename Mainloop::GmemLayout;
  using TileShapeQK = typename Mainloop::TileShapeQK;
  using TileShapePV = typename Mainloop::TileShapePV;
  using SmemLayoutO = typename Mainloop::SmemLayoutO;

  static constexpr int kNumThreads = Mainloop::kNumThreads;
  static constexpr int kNumMmaThreads = Mainloop::kNumMmaThreads;
  static constexpr int kBlockM = Mainloop::kBlockM;
  static constexpr int kSharedStorageSize = int(sizeof(SharedStorage));

  static constexpr uint32_t kLoadRegisters = 24;
  static constexpr uint32_t kMmaRegisters = 240;

  using TMA_O = decltype(make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(make_gmem_ptr(static_cast<Element *>(nullptr)), GmemLayout{}),
      SmemLayoutO{}, select<0, 2>(TileShapeQK{}), _1{}));

  struct Params {
    typename Mainloop::Params mainloop;
    TMA_O tma_store_o;
    GmemLayout layout_o;
    float *lse;
    int num_heads;
  };

  static Params to_underlying_arguments(FlashAttnParams const &args) {
    auto const *q = reinterpret_cast<Element const *>(args.q);
    auto const *k = reinterpret_cast<Element const *>(args.k);
    auto const *v = reinterpret_cast<Element const *>(args.v);
    auto *o = reinterpret_cast<Element *>(args.o);
    GmemLayout layout_o =
        Mainloop::make_gmem_layout(args.batch, args.seqlen_q, args.num_heads);
    TMA_O tma_o = make_tma_copy(SM90_TMA_STORE{}, make_tensor(make_gmem_ptr(o), layout_o),
                                SmemLayoutO{}, select<0, 2>(TileShapeQK{}), _1{});
    return {Mainloop::to_underlying_arguments(q, k, v, args.batch, args.seqlen_q,
                                              args.seqlen_k, args.num_heads,
                                              args.softmax_scale),
            tma_o, layout_o, args.lse, args.num_heads};
  }
};

template <class Kernel>
__global__ void __launch_bounds__(Kernel::kNumThreads, 1)
    flash_attn_fwd_sm90_kernel(CUTE_GRID_CONSTANT typename Kernel::Params const params) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && defined(CUTE_ARCH_TMA_SM90_ENABLED)
  using Mainloop = typename Kernel::Mainloop;
  using Element = typename Kernel::Element;
  using SharedStorage = typename Kernel::SharedStorage;
  using PipelineKV = typename Mainloop::PipelineKV;
  using PipelineState = typename Mainloop::PipelineState;
  using TileShapeQK = typename Kernel::TileShapeQK;
  using TileShapePV = typename Kernel::TileShapePV;
  constexpr int kNumMmaThreads = Kernel::kNumMmaThreads;
  constexpr int kBlockM = Kernel::kBlockM;

  extern __shared__ char smem_buf[];
  SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

  int const m_block = blockIdx.x;
  int const head = blockIdx.y;
  int const batch = blockIdx.z;

  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const warp_group_idx = cutlass::canonical_warp_group_idx();
  int const lane_predicate = cute::elect_one_sync();

  if (warp_idx == 0 && lane_predicate) {
    Mainloop::prefetch_tma_descriptors(params.mainloop);
    cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
    cutlass::arch::ClusterTransactionBarrier::init(&storage.pipelines.barrier_q, 1);
  }

  typename PipelineKV::Params pipeline_params;
  pipeline_params.role = warp_group_idx == 0 ? PipelineKV::ThreadCategory::Producer
                                             : PipelineKV::ThreadCategory::Consumer;
  pipeline_params.is_leader = threadIdx.x == 0;
  pipeline_params.num_consumers = kNumMmaThreads;
  pipeline_params.transaction_bytes = Mainloop::kTmaBytesKV;
  // The constructors initialise the full/empty mbarriers (warp 0) and fence
  // the inits; the CTA barrier below publishes them (and barrier_q).
  PipelineKV pipeline_k(storage.pipelines.pipeline_k, pipeline_params,
                        typename Mainloop::ClusterShape{});
  PipelineKV pipeline_v(storage.pipelines.pipeline_v, pipeline_params,
                        typename Mainloop::ClusterShape{});
  __syncthreads();

  Mainloop mainloop;
  int const n_blocks = Mainloop::n_block_max(params.mainloop, m_block);

  if (warp_group_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<Kernel::kLoadRegisters>();
    if (threadIdx.x == 0) {
      PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<PipelineKV>();
      PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<PipelineKV>();
      mainloop.load(params.mainloop, pipeline_k, pipeline_v, smem_pipe_write_k,
                    smem_pipe_write_v, storage, m_block, head, batch, n_blocks);
      // Do not retire the producer while consumers may still read the rings.
      pipeline_k.producer_tail(smem_pipe_write_k);
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<Kernel::kMmaRegisters>();
    int const mma_thread_idx = int(threadIdx.x) - cutlass::NumThreadsPerWarpGroup;

    typename Mainloop::TiledMmaPV tiled_mma_pv;
    Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShapePV{}));
    clear(tOrO);
    OnlineSoftmax<Mainloop::kRowsPerThread> softmax(params.mainloop.softmax_scale_log2);

    PipelineState smem_pipe_read_k;
    PipelineState smem_pipe_read_v;
    mainloop.mma(params.mainloop, pipeline_k, pipeline_v, smem_pipe_read_k,
                 smem_pipe_read_v, tOrO, softmax, mma_thread_idx, m_block,
                 n_blocks, storage);

    // ---- Epilogue -------------------------------------------------------
    float lse[Mainloop::kRowsPerThread];
    softmax.finalize(tOrO, params.mainloop.softmax_scale, lse);
    Tensor rO = convert_type<Element>(tOrO);

    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(mma_thread_idx);
    Tensor sO = make_tensor(make_smem_ptr(storage.tensors.smem_q.data()),
                            typename Kernel::SmemLayoutO{});
    Tensor tOsO = thr_mma_pv.partition_C(sO);

    // sQ is reused for O: every consumer must be past its last GEMM1.
    cutlass::arch::NamedBarrier::sync(kNumMmaThreads, Mainloop::kBarrierEpilogue);
    copy(rO, tOsO);
    // Make the generic-proxy smem writes visible to the TMA (async proxy).
    cute::tma_store_fence();
    cutlass::arch::NamedBarrier::sync(kNumMmaThreads, Mainloop::kBarrierEpilogue);

    if (mma_thread_idx == 0) {
      Tensor mO = params.tma_store_o.get_tma_tensor(params.layout_o.shape());
      Tensor gO = local_tile(mO(_, _, head, batch), select<0, 2>(TileShapeQK{}),
                             make_coord(m_block, _0{}));
      auto block_tma_o = params.tma_store_o.get_slice(_0{});
      copy(params.tma_store_o, block_tma_o.partition_S(sO), block_tma_o.partition_D(gO));
      cute::tma_store_arrive();
      cute::tma_store_wait<0>();
    }

    if (params.lse != nullptr && (mma_thread_idx % 4) == 0) {
      Tensor cO = make_identity_tensor(select<0, 1>(TileShapePV{}));
      Tensor tOcO = thr_mma_pv.partition_C(cO);
      Tensor tOcO_rc = make_tensor(tOcO.data(), convert_layout_acc_rowcol(tOcO.layout()));
      int const seqlen_q = params.mainloop.seqlen_q;
      float *lse_bh = params.lse + (int64_t(batch) * params.num_heads + head) * seqlen_q;
      CUTE_UNROLL
      for (int r = 0; r < Mainloop::kRowsPerThread; ++r) {
        int const row = m_block * kBlockM + get<0>(tOcO_rc(r, 0));
        if (row < seqlen_q) {
          lse_bh[row] = lse[r];
        }
      }
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("flash_attn_fwd_sm90_kernel requires sm_90a\n");
  }
#endif
}

template <int kHeadDim, bool kIsCausal, FlashAttnSchedule kSchedule>
bool Launch(FlashAttnParams const &args, cudaStream_t stream) {
  using Mainloop = FusedAttentionMainloop<kHeadDim, /*kBlockM=*/128, /*kBlockN=*/128,
                                          /*kStages=*/2, kIsCausal, kSchedule>;
  using Kernel = FlashAttnFwdKernel<Mainloop>;

  typename Kernel::Params params = Kernel::to_underlying_arguments(args);
  auto kernel = flash_attn_fwd_sm90_kernel<Kernel>;
  constexpr int smem_bytes = Kernel::kSharedStorageSize;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_bytes) != cudaSuccess) {
    return false;
  }
  dim3 grid((args.seqlen_q + Kernel::kBlockM - 1) / Kernel::kBlockM, args.num_heads,
            args.batch);
  kernel<<<grid, Kernel::kNumThreads, smem_bytes, stream>>>(params);
  return cudaGetLastError() == cudaSuccess;
}

template <int kHeadDim, bool kIsCausal>
bool DispatchSchedule(FlashAttnParams const &args, cudaStream_t stream) {
  switch (args.schedule) {
  case FlashAttnSchedule::kSerial:
    return Launch<kHeadDim, kIsCausal, FlashAttnSchedule::kSerial>(args, stream);
  case FlashAttnSchedule::kIntraWGOverlap:
    return Launch<kHeadDim, kIsCausal, FlashAttnSchedule::kIntraWGOverlap>(args, stream);
  case FlashAttnSchedule::kPingPong:
    return Launch<kHeadDim, kIsCausal, FlashAttnSchedule::kPingPong>(args, stream);
  }
  return false;
}

template <int kHeadDim>
bool DispatchCausal(FlashAttnParams const &args, cudaStream_t stream) {
  return args.causal ? DispatchSchedule<kHeadDim, true>(args, stream)
                     : DispatchSchedule<kHeadDim, false>(args, stream);
}

} // namespace fmha

bool FlashAttnFwdSm90(const FlashAttnParams &args, cudaStream_t stream) {
  if (args.q == nullptr || args.k == nullptr || args.v == nullptr || args.o == nullptr) {
    return false;
  }
  if (args.batch <= 0 || args.num_heads <= 0 || args.seqlen_q <= 0 || args.seqlen_k <= 0) {
    return false;
  }
  if (args.batch > 65535 || args.num_heads > 65535) {
    return false;
  }
  // TMA: 16-byte aligned base addresses (row strides are heads * head_dim * 2
  // bytes, already a multiple of 16 for the supported head dims).
  auto misaligned = [](const void *p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) != 0;
  };
  if (misaligned(args.q) || misaligned(args.k) || misaligned(args.v) || misaligned(args.o)) {
    return false;
  }
  switch (args.head_dim) {
  case 64:
    return fmha::DispatchCausal<64>(args, stream);
  case 128:
    return fmha::DispatchCausal<128>(args, stream);
  default:
    return false;
  }
}

} // namespace cu_epilogue

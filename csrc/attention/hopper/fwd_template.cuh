#pragma once

#include "attention/hopper/epilogue_fwd_fp32.hpp"

#include "attention/hopper/fwd_params.h"

#include <ATen/cuda/Exceptions.h>
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>
#include <cutlass/numeric_types.h>
#include <torch/types.h>

#include <cstdint>
#include <tuple>
#include <type_traits>

#include "cute/tensor.hpp"
#include "attention/semantics/visibility.h"
#include "attention/hopper/epilogue_fwd.hpp"
#include "attention/hopper/tile_scheduler.hpp"
#include "attention/hopper/fwd_kernel_sm90.h"
#include "attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp"

namespace xattn {
namespace ops {

using namespace cute;

template <typename Mainloop>
struct AttentionDefaultMainloop {
  using Type = Mainloop;
};

template <typename Visibility, bool kVarlen, bool kHasSegment>
constexpr std::tuple<int, int, bool, bool> AttentionTileSizeFwdSm90(
    int headdim, int headdim_v) {
  // Bound segment-mask and QK/PV fragment register lifetimes for wide V.
  if constexpr (kHasSegment && Visibility::kKind !=
      attention::semantics::VisibilityKind::kCausalFull) {
    if (headdim <= 64 && headdim_v == 192) {
      return {128, 96, true, true};
    }
    if ((headdim == 128 && headdim_v == 192) ||
        (headdim == 160 && headdim_v == 256)) {
      return {128, 64, true, false};
    }
  }
  if constexpr (
      Visibility::kKind ==
          attention::semantics::VisibilityKind::kCausalFull &&
      !kHasSegment) {
    // Serialize QK/PV register lifetimes for this rectangular tile.
    if (!kVarlen && headdim == 128 && headdim_v == 256) {
      return {128, 128, true, false};
    }
    if (headdim == headdim_v) {
      if (headdim <= 64) {
        return {192, 128, true, true};
      }
      if (headdim <= 96) {
        return {192, 144, false, true};
      }
      if (headdim <= 128) {
        return {128, 128, true, true};
      }
      if (headdim <= 192) {
        return {128, headdim_v <= 128 ? 128 : 112, true, true};
      }
      return {128, 80, true, true};
    }
  }
  if constexpr (
      Visibility::kKind ==
          attention::semantics::VisibilityKind::kSlidingChunk &&
      !kVarlen && !kHasSegment) {
    if (headdim <= 64 && headdim_v <= 64) {
      return {192, 128, true, true};
    }
    if (headdim <= 96 && headdim_v <= 96 && headdim > 64) {
      return {192, 128, false, true};
    }
  }
  if constexpr (
      Visibility::kKind ==
          attention::semantics::VisibilityKind::kCausalSlidingWindow &&
      !kVarlen && !kHasSegment) {
    if ((headdim <= 32 && headdim_v <= 128) ||
        (headdim <= 64 && headdim_v <= 96)) {
      return {192, 128, true, true};
    }
    if (headdim <= 96 && headdim_v <= 96) {
      return {192, 128, false, true};
    }
  }
  if (headdim <= 64) {
    if (headdim_v == 256) {
      if constexpr (kHasSegment) {
        return {128, 64, true, false};
      }
      return {128, 96, true, false};
    }
    if constexpr (!kHasSegment) {
      if (headdim_v <= 128) {
        return {128, 128, true, true};
      }
      return {128, 96, true, true};
    }
    return {192, 128, true, true};
  }
  if (headdim <= 96) {
    if (headdim_v == 256) {
      if constexpr (kVarlen || kHasSegment) {
        return {128, 64, true, false};
      }
      return {128, 96, true, false};
    }
    if constexpr (kHasSegment) {
      if (headdim_v == 192) {
        return {128, 96, false, true};
      }
    }
    if constexpr (!kHasSegment) {
      if (headdim_v <= 96) {
        return {128, 128, true, true};
      }
      if (headdim_v <= 192) {
        return {128, 96, false, true};
      }
    }
    return {192, 128, false, true};
  }
  if (headdim <= 128) {
    if (headdim_v == 256 && kHasSegment) {
      return {128, 64, true, false};
    }
    return {128, 128, true, true};
  }
  if (headdim <= 192) {
    return {128, 96, true, true};
  }
  return {128, 64, true, true};
}

struct AttentionFwdEpilogueArguments {
  template <typename CollectiveEpilogue, typename Params>
  static typename CollectiveEpilogue::Arguments Make(Params& params) {
    return {
        static_cast<typename CollectiveEpilogue::Element*>(params.o_ptr),
        {params.seqlen_q, params.dv, params.h, params.b, 1},
        {params.o_row_stride, _1{}, params.o_head_stride,
         params.o_batch_stride, 0},
        static_cast<float*>(params.oaccum_ptr),
        {params.oaccum_row_stride, _1{}, params.oaccum_head_stride,
         params.oaccum_batch_stride, params.oaccum_split_stride},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, params.seqlen_q, params.h * params.seqlen_q, 0},
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {_1{}, params.lseaccum_head_stride, params.lseaccum_batch_stride,
         params.lseaccum_split_stride},
        params.h_k,
        CollectiveEpilogue::Varlen ? params.cu_seqlens_q : nullptr,
        nullptr,
        static_cast<float*>(params.o_state_ptr),
        {params.o_state_row_stride, _1{}, params.o_state_head_stride,
         params.o_state_batch_stride, 0}};
  }
};

template <
    int kHeadDim, int kHeadDimV, typename Element, typename ElementOut,
    bool kVarlen, bool kHasSegment, bool kUsePersistentScheduler,
    int kBlockM, int kBlockN,
    bool MmaPVIsRS, bool IntraWGOverlap, bool kUseFastExp2=false,
    typename Visibility = attention::semantics::SlidingChunkVisibility,
    template <class, class, class> class FwdKernel =
        flash::AttentionFwdSm90,
    int kClusterM = 1, bool kSoftDeltaPair = false,
    bool kSoftDeltaSingleCTA = false,
    typename SchedulerOverride = void,
    template <class, class, class, class, int, bool, bool, bool, bool>
    class FwdEpilogue = flash::CollectiveEpilogueFwd,
    typename EpilogueArguments = AttentionFwdEpilogueArguments,
    template <typename> class MainloopAdapter =
        AttentionDefaultMainloop,
    int kStages = 2,
    bool kSoftDeltaSplitKVMulticast = false,
    bool kSoftDeltaClusteredRowPair = false,
    bool kInitializeTileCounterOnlyForMultipleWaves = false>
void RunAttentionFwdSm90KernelTile(
    AttentionFwdParams& params, cudaStream_t stream) {
  static constexpr bool IsCausal = Visibility::kUseCausalMask;
  static constexpr bool IsLocal = Visibility::kUseLocalMask;
  static constexpr bool IsSlidingChunk = Visibility::kIsSlidingChunk;
  static constexpr bool HasSoftcap = false;
  static constexpr bool Varlen = kVarlen;
  static constexpr bool PagedKVNonTMA = false;
  static constexpr bool AppendKV = false;
  static constexpr bool HasQv = false;
  static constexpr bool PackGQA = false;
  static constexpr bool Split = false;
  static constexpr bool VColmajor = false;
  static constexpr bool FP8TransposeV = false;
  static constexpr int ClusterM = kClusterM;
  static_assert(!kSoftDeltaSingleCTA || kSoftDeltaPair);
  static_assert(
      !kSoftDeltaClusteredRowPair || kSoftDeltaSingleCTA);
  static_assert(
      !kSoftDeltaClusteredRowPair || ClusterM == 2);
  static_assert(
      !kInitializeTileCounterOnlyForMultipleWaves ||
      (kUsePersistentScheduler && ClusterM == 1));
  static_assert(
      !kSoftDeltaSplitKVMulticast || kSoftDeltaPair);
  static_assert(
      !kSoftDeltaSplitKVMulticast || !kSoftDeltaSingleCTA ||
      kSoftDeltaClusteredRowPair);
  static_assert(
      !kSoftDeltaPair ||
      (kSoftDeltaSingleCTA
           ? ClusterM == (kSoftDeltaClusteredRowPair ? 2 : 1)
           : (ClusterM == 1 || ClusterM == 2)));
  static_assert(kBlockM % 64 == 0,
                "attention SM90 FWD BM must be a multiple of 64");
  static_assert(kBlockN % 8 == 0,
                "attention SM90 FWD BN must be a multiple of 8");
  static_assert(kHeadDim % 8 == 0,
                "attention SM90 FWD requires D to be a multiple of 8");
  static_assert(kHeadDimV % 8 == 0,
                "attention SM90 FWD requires V to be a multiple of 8");
  using TileShapeMNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using TileShapeMNKPV =
      cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
  using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;
  using ArchTag = cutlass::arch::Sm90;
  using DefaultCollectiveMainloop = flash::CollectiveMainloopFwdSm90<
      kStages, ClusterShape, TileShapeMNK, kHeadDimV, Element, float,
      ArchTag, IsCausal, IsLocal, HasSoftcap, Varlen, PagedKVNonTMA,
      AppendKV, HasQv, kHasSegment, MmaPVIsRS, IntraWGOverlap, PackGQA,
      Split, VColmajor, kUseFastExp2, IsSlidingChunk, kSoftDeltaPair,
      kSoftDeltaSingleCTA, kSoftDeltaSplitKVMulticast,
      kSoftDeltaClusteredRowPair>;
  using CollectiveMainloop =
      typename MainloopAdapter<DefaultCollectiveMainloop>::Type;
  using CollectiveEpilogue = FwdEpilogue<
      TileShapeMNKPV, ClusterShape, ElementOut, ArchTag,
      CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, Split,
      FP8TransposeV>;
  using DensePersistentScheduler = flash::DynamicPersistentTileScheduler<
      CollectiveMainloop::NumMmaThreads,
      CollectiveMainloop::NumProducerThreads, Split, PackGQA,
      true /*WarpSpecialized*/>;
  using VarlenPersistentScheduler =
      flash::VarlenDynamicPersistentTileScheduler<
          kBlockM, kBlockN, CollectiveMainloop::NumMmaThreads,
          CollectiveMainloop::NumProducerThreads, false /*Split*/,
          false /*PackGQA*/, true /*WarpSpecialized*/, false /*LPT*/,
          false /*Sort*/, false /*Prepared*/>;
  using PersistentScheduler = std::conditional_t<
      kVarlen, VarlenPersistentScheduler, DensePersistentScheduler>;
  using SingleTileScheduler =
      flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;
  using DefaultScheduler = std::conditional_t<
      kUsePersistentScheduler, PersistentScheduler, SingleTileScheduler>;
  using Scheduler = std::conditional_t<
      std::is_void_v<SchedulerOverride>, DefaultScheduler,
      SchedulerOverride>;
  using AttnKernel =
      flash::enable_sm90<FwdKernel<
          CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

  using StrideV = typename CollectiveMainloop::StrideV;
  StrideV v_strides = make_stride(
      params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride);

  typename CollectiveMainloop::Arguments mainloop_args{
      static_cast<Element const*>(params.q_ptr),
      {params.seqlen_q, params.d, params.h, params.b},
      {params.q_row_stride, _1{}, params.q_head_stride,
       params.q_batch_stride},
      static_cast<Element*>(params.k_ptr),
      {params.seqlen_k, params.d, params.h_k, params.b},
      {params.k_row_stride, _1{}, params.k_head_stride,
       params.k_batch_stride},
      static_cast<Element*>(params.v_ptr),
      params.dv,
      v_strides,
      nullptr,
      {0, params.d, params.h_k, params.b},
      {params.k_row_stride, _1{}, params.k_head_stride,
       params.k_batch_stride},
      nullptr,
      v_strides,
      nullptr,
      {params.q_row_stride, _1{}, params.q_head_stride,
       params.q_batch_stride},
      nullptr,
      {params.seqlen_k, 0},
      {0, _1{}},
      nullptr,
      {0, _1{}},
      false,
      nullptr,
      {params.b, 0},
      {0, _1{}},
      params.scale_softmax,
      nullptr,
      nullptr,
      nullptr,
      {0, 0},
      {0, 0},
      {0, 0},
      params.window_size_left,
      params.window_size_right,
      params.attention_chunk,
      0.0f,
      1,
      kHasSegment ? params.q_segment_idx : nullptr,
      kHasSegment ? params.k_segment_idx : nullptr,
      kHasSegment ? params.k_segment_len : 0,
      nullptr,
      kVarlen ? params.cu_seqlens_q : nullptr,
      kVarlen ? params.cu_seqlens_k : nullptr,
      nullptr,
      nullptr,
      kVarlen ? params.seqused_k : nullptr,
      nullptr,
      nullptr,
      kVarlen ? params.q_position_offsets : nullptr,
      kHasSegment ? params.q_chunk_positions : nullptr,
      kHasSegment ? params.reset_attention_chunk : 0};

  typename CollectiveEpilogue::Arguments epilogue_args =
      EpilogueArguments::template Make<CollectiveEpilogue>(params);

  int qhead_per_khead = params.h / params.h_k;
  static constexpr int kSchedulerBlockM =
      kSoftDeltaSingleCTA ? kBlockM / 2 : kBlockM;
  int num_blocks_m = cutlass::ceil_div(
      kVarlen ? params.max_seqlen_q : params.seqlen_q,
      kSchedulerBlockM);
  int scheduler_heads = kSoftDeltaPair ? params.h / 2 : params.h;
  int scheduler_batches = kVarlen ? params.num_sequences : params.b;
  std::int64_t scheduler_work_tiles =
      std::int64_t(num_blocks_m) * scheduler_heads * scheduler_batches;
  torch::Tensor tile_count;
  int* tile_count_ptr = nullptr;
  if constexpr (kUsePersistentScheduler) {
    tile_count = torch::empty(
        {1},
        torch::TensorOptions()
            .device(torch::kCUDA)
            .dtype(at::kInt)
            .memory_format(at::MemoryFormat::Contiguous));
    tile_count_ptr = tile_count.data_ptr<int>();
    bool initialize_tile_counter = true;
    if constexpr (kInitializeTileCounterOnlyForMultipleWaves) {
      initialize_tile_counter = scheduler_work_tiles > params.num_sm;
    }
    if (initialize_tile_counter) {
      C10_CUDA_CHECK(cudaMemsetAsync(
          tile_count.data_ptr<int>(), 0, sizeof(int), stream));
    }
  }
  typename flash::TileSchedulerArguments scheduler_args{
      num_blocks_m,
      scheduler_heads,
      scheduler_batches,
      1,
      qhead_per_khead,
      kVarlen ? params.max_seqlen_q : params.seqlen_q,
      kVarlen ? params.max_seqlen_k : params.seqlen_k,
      params.d,
      params.dv,
      sizeof(Element),
      tile_count_ptr,
      kVarlen ? params.cu_seqlens_q : nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      params.seqlen_q};

  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  typename AttnKernel::Params kernel_params =
      AttnKernel::to_underlying_arguments(
          {mainloop_args, epilogue_args, {device, params.num_sm},
           scheduler_args});

  dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
  dim3 block_dims = AttnKernel::get_block_shape();
  int smem_size = AttnKernel::SharedStorageSize;
  auto kernel = cutlass::device_kernel<AttnKernel>;
  if (smem_size >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  cutlass::Status status;
  if constexpr (ClusterM == 1) {
    status = cutlass::kernel_launch<AttnKernel>(
        grid_dims, block_dims, smem_size, stream, kernel_params, false);
  } else {
    void const* kernel_ptr =
        const_cast<void const*>(reinterpret_cast<void*>(kernel));
    status = cutlass::launch_kernel_on_cluster(
        {grid_dims, block_dims, dim3(ClusterM, 1, 1), smem_size, stream},
        kernel_ptr, kernel_params);
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "attention SM90 FWD CUTLASS launch failed: ",
              cutlass::cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
    int kHeadDim, int kHeadDimV, typename Element, typename ElementOut,
    bool kVarlen, bool kHasSegment, bool kUsePersistentScheduler,
    bool kUseFastExp2=false,
    typename Visibility = attention::semantics::SlidingChunkVisibility,
    template <class, class, class> class FwdKernel =
        flash::AttentionFwdSm90>
void RunAttentionFwdSm90Kernel(
    AttentionFwdParams& params, cudaStream_t stream) {
  static constexpr std::tuple<int, int, bool, bool> kCurrentBlockMN =
      AttentionTileSizeFwdSm90<Visibility, kVarlen, kHasSegment>(
          kHeadDim, kHeadDimV);

  if constexpr (!kVarlen && !kHasSegment && kHeadDim == 96 &&
                (kHeadDimV == 64 || kHeadDimV == 128) &&
                Visibility::kKind == attention::semantics::VisibilityKind::kCausalFull) {
    if (params.d == 96 && params.dv == kHeadDimV &&
        params.b == 1 && params.h == 8 &&
        (params.h_k == 1 || params.h_k == 2 || params.h_k == 8) &&
        params.seqlen_q == params.seqlen_k &&
        params.window_size_right == 0 && !params.output_fp32 &&
        params.seqlen_q >= (kHeadDimV == 64 ? 32768 : 4096)) {
      if (params.seqlen_q >= 32768) {
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, ElementOut, false, false,
            kUsePersistentScheduler, 192, 128, false, true,
            kUseFastExp2, Visibility, FwdKernel>(params, stream);
      } else {
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, ElementOut, false, false,
            kUsePersistentScheduler, 128, 128, true, true,
            kUseFastExp2, Visibility, FwdKernel>(params, stream);
      }
      return;
    }
  }

  // Dense equal-length tile specializations.
  static constexpr bool kWindowTile =
      Visibility::kKind == attention::semantics::VisibilityKind::kCausalSlidingWindow &&
      ((kHeadDim == 32 && kHeadDimV <= 128) ||
       (kHeadDim == 64 && kHeadDimV <= 96) ||
       (kHeadDim == 96 && kHeadDimV == 96));
  static constexpr bool kFullTile =
      Visibility::kKind == attention::semantics::VisibilityKind::kCausalFull &&
      ((kHeadDim == 32 && kHeadDimV == 64) ||
       (kHeadDim == 64 && kHeadDimV == 32) ||
       (kHeadDim == 192 && kHeadDimV == 128) ||
       (kHeadDim == 256 && kHeadDimV == 32));
  if constexpr (!kVarlen && !kHasSegment && (kWindowTile || kFullTile)) {
    static constexpr bool kLongFull = kFullTile && kHeadDim <= 64;
    static constexpr bool kWideWindow = kWindowTile &&
        ((kHeadDim == 32 && kHeadDimV >= 96) ||
         (kHeadDim == 64 && kHeadDimV == 96));
    const bool use_tile = params.d == kHeadDim && params.dv == kHeadDimV &&
        params.seqlen_q == params.seqlen_k &&
        (!kLongFull || params.seqlen_q >= 16384) &&
        (!kWindowTile || kWideWindow ||
         (params.window_size_left >= 0 && params.window_size_left <= 511));
    if (use_tile) {
      if (params.output_fp32) {
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, ElementOut, false, false,
            kUsePersistentScheduler, kLongFull ? 192 : 128, 128,
            std::get<2>(kCurrentBlockMN), std::get<3>(kCurrentBlockMN),
            kUseFastExp2, Visibility, FwdKernel, 1, false, false, void,
            flash::CollectiveEpilogueFwdFp32>(params, stream);
      } else {
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, ElementOut, false, false,
            kUsePersistentScheduler, kLongFull ? 192 : 128, 128,
            std::get<2>(kCurrentBlockMN), std::get<3>(kCurrentBlockMN),
            kUseFastExp2, Visibility, FwdKernel>(params, stream);
      }
      return;
    }
  }

  if (params.output_fp32) {
    RunAttentionFwdSm90KernelTile<
        kHeadDim, kHeadDimV, Element, ElementOut, kVarlen, kHasSegment,
        kUsePersistentScheduler, std::get<0>(kCurrentBlockMN),
        std::get<1>(kCurrentBlockMN), std::get<2>(kCurrentBlockMN),
        std::get<3>(kCurrentBlockMN), kUseFastExp2, Visibility,
        FwdKernel, 1, false, false, void,
        flash::CollectiveEpilogueFwdFp32>(params, stream);
    return;
  }

  // Low-precision epilogue specialization for selected dense full shapes.
  if constexpr (
      Visibility::kKind ==
          attention::semantics::VisibilityKind::kCausalFull &&
      !kVarlen && !kHasSegment && kUseFastExp2 &&
      kHeadDim == kHeadDimV && (kHeadDim == 64 || kHeadDim == 128)) {
    if (params.o_state_ptr == nullptr && params.use_fast_reciprocal) {
      RunAttentionFwdSm90KernelTile<
          kHeadDim, kHeadDimV, Element, ElementOut, kVarlen, kHasSegment,
          kUsePersistentScheduler, std::get<0>(kCurrentBlockMN),
          std::get<1>(kCurrentBlockMN), std::get<2>(kCurrentBlockMN),
          std::get<3>(kCurrentBlockMN), kUseFastExp2, Visibility,
          FwdKernel, 1 /*kClusterM*/, false /*kSoftDeltaPair*/,
          false /*kSoftDeltaSingleCTA*/, void /*SchedulerOverride*/,
          flash::CollectiveEpilogueFwdNoState>(params, stream);
      return;
    }
  }

  RunAttentionFwdSm90KernelTile<
      kHeadDim, kHeadDimV, Element, ElementOut, kVarlen, kHasSegment,
      kUsePersistentScheduler, std::get<0>(kCurrentBlockMN),
      std::get<1>(kCurrentBlockMN), std::get<2>(kCurrentBlockMN),
      std::get<3>(kCurrentBlockMN), kUseFastExp2, Visibility,
      FwdKernel>(
      params, stream);
}

}  // namespace ops
}  // namespace xattn

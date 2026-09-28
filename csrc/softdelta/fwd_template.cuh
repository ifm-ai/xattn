#pragma once

#include "softdelta/fwd.h"

#include <tuple>

#include "attention/hopper/fwd_template.cuh"
#include "softdelta/fwd_epilogue_single_cta_sm90.h"
#include "softdelta/fwd_kernel_sm90.h"
#include "softdelta/mainloop_fwd_sm90.h"
#include "softdelta/fwd_policy.h"
#include "softdelta/fwd_tile_scheduler.h"

namespace xattn {
namespace ops {

struct FlashSoftDeltaOutputEpilogueArguments {
  template <typename CollectiveEpilogue>
  static typename CollectiveEpilogue::Arguments Make(
      AttentionFwdParams& params) {
    auto& softdelta_params =
        static_cast<FlashSoftDeltaFwdParams&>(params);
    return {
        static_cast<typename CollectiveEpilogue::Element*>(
            params.o_ptr),
        {params.seqlen_q, params.dv, params.h / 2, params.b, 1},
        {params.o_row_stride, cute::_1{}, params.o_head_stride,
         params.o_batch_stride, 0},
        static_cast<const typename CollectiveEpilogue::Element*>(
            softdelta_params.gate_ptr),
        softdelta_params.gate_batch_stride,
        softdelta_params.gate_row_stride,
        softdelta_params.gate_head_stride,
        softdelta_params.gate_group_stride};
  }
};

template <typename Visibility>
constexpr std::tuple<int, int, bool, bool>
FlashSoftDeltaTileSizeFwdSm90(int headdim, int headdim_v) {
  const auto tile = AttentionTileSizeFwdSm90<Visibility, false, false>(
      headdim, headdim_v);
  return {
      std::get<0>(tile), std::get<1>(tile),
      std::get<2>(tile), std::get<3>(tile)};
}

template <
    typename Element, typename ElementOut, int kHeadDim,
    int kHeadDimV, typename Visibility>
bool MaybeRunFlashSoftDeltaInterleavedReaderFwdSm90VD(
    AttentionFwdParams& params, cudaStream_t stream) {
  auto& softdelta_params =
      static_cast<FlashSoftDeltaFwdParams&>(params);
  // Specialized direct-output epilogues assume four gate groups. Other
  // group counts need interleaved reads followed by the generic gate kernel.
  if (!softdelta_params.emit_ungated_pair &&
      softdelta_params.gate_group_dim == kHeadDimV / 4) {
    return false;
  }
  static constexpr std::tuple<int, int, bool, bool> kTile =
      FlashSoftDeltaTileSizeFwdSm90<Visibility>(
          kHeadDim, kHeadDimV);
  if constexpr (Visibility::kKind ==
                    attention::semantics::VisibilityKind::kCausalSlidingWindow &&
                (((kHeadDim == 32 || kHeadDim == 64) &&
                  (kHeadDimV == 32 || kHeadDimV == 64)) ||
                 (kHeadDim == 96 && kHeadDimV == 96))) {
    if (softdelta_params.emit_ungated_pair &&
        softdelta_params.gate_group_dim == kHeadDimV / 4 &&
        params.b == 1 && params.h == 16 &&
        (params.h_k == 1 || params.h_k == 2 || params.h_k == 8) &&
        params.seqlen_q == params.seqlen_k && params.seqlen_q >= 4096 &&
        params.window_size_left == 255 && params.window_size_right == 0 &&
        params.o_state_ptr == nullptr) {
      RunAttentionFwdSm90KernelTile<
          kHeadDim, kHeadDimV, Element, ElementOut, false, false,
          false, 128, 128, std::get<2>(kTile), std::get<3>(kTile),
          true, Visibility, flash::FlashSoftDeltaFwdSm90, 2, true, false,
          FlashSoftDeltaPairPersistentTileScheduler,
          flash::CollectiveEpilogueFwd, AttentionFwdEpilogueArguments,
          FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);
      return true;
    }
  }
  RunAttentionFwdSm90KernelTile<
      kHeadDim, kHeadDimV, Element, ElementOut, false, false,
      false, std::get<0>(kTile), std::get<1>(kTile),
      std::get<2>(kTile), std::get<3>(kTile), true, Visibility,
      flash::FlashSoftDeltaFwdSm90, 2, true, false,
      FlashSoftDeltaPairPersistentTileScheduler,
      flash::CollectiveEpilogueFwd,
      AttentionFwdEpilogueArguments,
      FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);
  return true;
}

template <
    typename Element, typename ElementOut, int kHeadDim,
    int kHeadDimV, typename Visibility>
void RunFlashSoftDeltaFwdSm90VD(
    AttentionFwdParams& params, cudaStream_t stream) {
  static constexpr std::tuple<int, int, bool, bool> kTile =
      FlashSoftDeltaTileSizeFwdSm90<Visibility>(
          kHeadDim, kHeadDimV);
  if (MaybeRunFlashSoftDeltaInterleavedReaderFwdSm90VD<
          Element, ElementOut, kHeadDim, kHeadDimV, Visibility>(
          params, stream)) {
    return;
  }
  auto& softdelta_params =
      static_cast<FlashSoftDeltaFwdParams&>(params);
  if constexpr (
      FlashSoftDeltaUseClusteredRowPairFwdSm90(
          kHeadDim, kHeadDimV, 4, Visibility::kKind)) {
    if (softdelta_params.gate_group_dim == kHeadDimV / 4) {
      RunAttentionFwdSm90KernelTile<
          kHeadDim, kHeadDimV, Element, ElementOut, false, false,
          true, 128, 80, true, true, true, Visibility,
          flash::FlashSoftDeltaFwdSm90, 2, true, true,
          FlashSoftDeltaClusteredRowPairTileScheduler<
              256, 32, FlashSoftDeltaAllHeadTileGrouping,
              FlashSoftDeltaWaveTailTileOrder>,
          FlashSoftDeltaSingleCTAEpilogueFwd,
          FlashSoftDeltaOutputEpilogueArguments,
          FlashSoftDeltaMainloopAdapterFwdSm90,
          2, false, true>(params, stream);
      return;
    }
  }
  if constexpr (
      FlashSoftDeltaUseSingleCTAFwdSm90(
          kHeadDim, kHeadDimV, 4, Visibility::kKind)) {
    if (softdelta_params.gate_group_dim == kHeadDimV / 4 &&
        !FlashSoftDeltaUsePairCTAFwdSm90(
            kHeadDim, kHeadDimV, 4, Visibility::kKind)) {
      static constexpr int kSingleCTABlockN =
          FlashSoftDeltaSingleCTABlockNFwdSm90(
              kHeadDim, kHeadDimV, Visibility::kKind);
      static constexpr bool kSingleCTAIntraWGOverlap =
          FlashSoftDeltaSingleCTAIntraWGOverlapFwdSm90(
              kHeadDim, kHeadDimV, Visibility::kKind);
      if constexpr (
          Visibility::kKind ==
              attention::semantics::VisibilityKind::kCausalFull &&
          ((kHeadDim == 256 && kHeadDimV == 256) ||
           (kHeadDim == 128 && kHeadDimV == 128))) {
        static constexpr int kDynamicBlockN =
            kHeadDim == 128 ? 176 : kSingleCTABlockN;
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, ElementOut, false, false,
            true, 128, kDynamicBlockN, true,
            kSingleCTAIntraWGOverlap, true, Visibility,
            flash::FlashSoftDeltaFwdSm90, 1, true, true,
            FlashSoftDeltaSingleCTADynamicTileScheduler<
                256, 32, FlashSoftDeltaKVHeadTileGrouping<2>>,
            FlashSoftDeltaSingleCTAEpilogueFwd,
            FlashSoftDeltaOutputEpilogueArguments,
            FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);
        return;
      }
      RunAttentionFwdSm90KernelTile<
          kHeadDim, kHeadDimV, Element, ElementOut, false, false,
          false, 128, kSingleCTABlockN, true,
          kSingleCTAIntraWGOverlap, true, Visibility,
          flash::FlashSoftDeltaFwdSm90, 1, true, true,
          FlashSoftDeltaSingleCTAPersistentTileScheduler,
          FlashSoftDeltaSingleCTAEpilogueFwd,
          FlashSoftDeltaOutputEpilogueArguments,
          FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);
      return;
    }
  }
  if constexpr (
      FlashSoftDeltaUseSingleCTARowPairFwdSm90(
          kHeadDim, kHeadDimV, 4, Visibility::kKind)) {
    if (softdelta_params.gate_group_dim == kHeadDimV / 4) {
      if constexpr (std::is_same_v<Element, cutlass::half_t>) {
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, ElementOut, false, false,
            true, 192, 128, true, true, true, Visibility,
            flash::FlashSoftDeltaFwdSm90, 1, true, true,
            FlashSoftDeltaSingleCTADynamicTileScheduler<
                384, 32, FlashSoftDeltaKVHeadTileGrouping<2>>,
            FlashSoftDeltaSingleCTAEpilogueFwd,
            FlashSoftDeltaOutputEpilogueArguments,
            FlashSoftDeltaMainloopAdapterFwdSm90,
            3>(
                params, stream);
      } else {
        using RowPairTileScheduler = std::conditional_t<
            kHeadDim == 64 && kHeadDimV == 64,
            FlashSoftDeltaClusteredRowPairL2TileScheduler<
                384, 32, FlashSoftDeltaKVHeadTileGrouping<2>>,
            FlashSoftDeltaClusteredRowPairTileScheduler<
                384, 32, FlashSoftDeltaKVHeadTileGrouping<2>>>;
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, ElementOut, false, false,
            true, 192, 128, true, true, true, Visibility,
            flash::FlashSoftDeltaFwdSm90, 2, true, true,
            RowPairTileScheduler,
            FlashSoftDeltaSingleCTAEpilogueFwd,
            FlashSoftDeltaOutputEpilogueArguments,
            FlashSoftDeltaMainloopAdapterFwdSm90,
            3, true, true>(
                params, stream);
      }
      return;
    }
  }
  if constexpr (
      Visibility::kKind ==
          attention::semantics::VisibilityKind::kCausalFull &&
      ((kHeadDim == 128 && kHeadDimV == 128) ||
       (kHeadDim == 256 && kHeadDimV == 256))) {
    if (softdelta_params.gate_group_dim == kHeadDimV / 4 &&
        FlashSoftDeltaUsePairCTAFwdSm90(
            kHeadDim, kHeadDimV, 4, Visibility::kKind)) {
      RunAttentionFwdSm90KernelTile<
          kHeadDim, kHeadDimV, Element, ElementOut, false, false,
          true, std::get<0>(kTile),
          kHeadDim == 128 ? 176 : std::get<1>(kTile),
          std::get<2>(kTile), std::get<3>(kTile), true, Visibility,
          flash::FlashSoftDeltaFwdSm90, 2, true, false,
          FlashSoftDeltaPairDynamicTileScheduler<256, 32>,
          flash::CollectiveEpilogueFwd,
          AttentionFwdEpilogueArguments,
          FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);
      return;
    }
  }
  if constexpr (
      Visibility::kKind ==
          attention::semantics::VisibilityKind::kCausalFull &&
      kHeadDim == 96 && kHeadDimV == 96) {
    RunAttentionFwdSm90KernelTile<
        kHeadDim, kHeadDimV, Element, ElementOut, false, false,
        false, 192, 144, false, true, true, Visibility,
        flash::FlashSoftDeltaFwdSm90, 1, true, false,
        FlashSoftDeltaPairPersistentTileScheduler,
        flash::CollectiveEpilogueFwd,
        AttentionFwdEpilogueArguments,
        FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);
    return;
  }
  RunAttentionFwdSm90KernelTile<
      kHeadDim, kHeadDimV, Element, ElementOut, false, false,
      false, std::get<0>(kTile), std::get<1>(kTile),
      std::get<2>(kTile), std::get<3>(kTile), true, Visibility,
      flash::FlashSoftDeltaFwdSm90, 2, true, false,
      FlashSoftDeltaPairPersistentTileScheduler,
      flash::CollectiveEpilogueFwd,
      AttentionFwdEpilogueArguments,
      FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);
}

}  // namespace ops
}  // namespace xattn

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE(                  \
    Element, HeadDim, HeadDimV)                                     \
  namespace xattn {                                                  \
  namespace ops {                                                    \
  template void RunFlashSoftDeltaFwdSm90VD<                          \
      Element, Element, HeadDim, HeadDimV,                           \
      attention::semantics::CausalFullVisibility>(                   \
      AttentionFwdParams&, cudaStream_t);                             \
  template void RunFlashSoftDeltaFwdSm90VD<                          \
      Element, Element, HeadDim, HeadDimV,                           \
      attention::semantics::CausalSlidingWindowVisibility>(          \
      AttentionFwdParams&, cudaStream_t);                             \
  template void RunFlashSoftDeltaFwdSm90VD<                          \
      Element, Element, HeadDim, HeadDimV,                           \
      attention::semantics::SlidingChunkVisibility>(                 \
      AttentionFwdParams&, cudaStream_t);                             \
  }                                                                  \
  }

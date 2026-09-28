#pragma once

#include "softdelta/fwd_template.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_SINGLE_CTA_GROUPED_FULL( \
    Element, HeadDim, HeadDimV, BlockM, BlockN, IntraWGOverlap,             \
    HeadGrouping, Stages)                                                  \
  namespace xattn {                                                        \
  namespace ops {                                                          \
  template <>                                                              \
  void RunFlashSoftDeltaFwdSm90VD<                                         \
      Element, Element, HeadDim, HeadDimV,                                 \
      attention::semantics::CausalFullVisibility>(                         \
      AttentionFwdParams& params, cudaStream_t stream) {                    \
    using Visibility = attention::semantics::CausalFullVisibility;         \
    auto& softdelta_params =                                               \
        static_cast<FlashSoftDeltaFwdParams&>(params);                     \
    if (MaybeRunFlashSoftDeltaInterleavedReaderFwdSm90VD<                  \
            Element, Element, HeadDim, HeadDimV, Visibility>(              \
            params, stream)) {                                            \
      return;                                                             \
    }                                                                     \
    if (softdelta_params.gate_group_dim == HeadDimV / 4) {                 \
      RunAttentionFwdSm90KernelTile<                                        \
          HeadDim, HeadDimV, Element, Element, false, false,               \
          true, BlockM, BlockN, true, IntraWGOverlap, true,                \
          Visibility, flash::FlashSoftDeltaFwdSm90, 1, true, true,         \
          FlashSoftDeltaSingleCTADynamicTileScheduler<                     \
              256, 32, HeadGrouping>,                                      \
          FlashSoftDeltaSingleCTAEpilogueFwd,                     \
          FlashSoftDeltaOutputEpilogueArguments,                           \
          FlashSoftDeltaMainloopAdapterFwdSm90, Stages>(params, stream);   \
      return;                                                              \
    }                                                                      \
    static constexpr std::tuple<int, int, bool, bool> kTile =              \
        FlashSoftDeltaTileSizeFwdSm90<Visibility>(HeadDim, HeadDimV);       \
    RunAttentionFwdSm90KernelTile<                                          \
        HeadDim, HeadDimV, Element, Element, false, false,                 \
        false, std::get<0>(kTile), std::get<1>(kTile),                     \
        std::get<2>(kTile), std::get<3>(kTile), true, Visibility,          \
        flash::FlashSoftDeltaFwdSm90, 2, true, false,                      \
        FlashSoftDeltaPairPersistentTileScheduler,                         \
        flash::CollectiveEpilogueFwd, AttentionFwdEpilogueArguments,        \
        FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);             \
  }                                                                        \
  template void RunFlashSoftDeltaFwdSm90VD<                                \
      Element, Element, HeadDim, HeadDimV,                                 \
      attention::semantics::CausalSlidingWindowVisibility>(                \
      AttentionFwdParams&, cudaStream_t);                                   \
  template void RunFlashSoftDeltaFwdSm90VD<                                \
      Element, Element, HeadDim, HeadDimV,                                 \
      attention::semantics::SlidingChunkVisibility>(                       \
      AttentionFwdParams&, cudaStream_t);                                   \
  }                                                                        \
  }

#pragma once

#include "softdelta/fwd_template.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_ORDERED_DYNAMIC_FULL_TILE( \
    Element, HeadDim, HeadDimV, BlockM, BlockN, Stages, HeadGrouping,   \
    TileOrder)                                                         \
  namespace xattn {                                                   \
  namespace ops {                                                     \
  template <>                                                         \
  void RunFlashSoftDeltaFwdSm90VD<                                    \
      Element, Element, HeadDim, HeadDimV,                            \
      attention::semantics::CausalFullVisibility>(                    \
      AttentionFwdParams& params, cudaStream_t stream) {                \
    using Visibility =                                                \
        attention::semantics::CausalFullVisibility;                    \
    if (MaybeRunFlashSoftDeltaInterleavedReaderFwdSm90VD<              \
            Element, Element, HeadDim, HeadDimV, Visibility>(          \
            params, stream)) {                                        \
      return;                                                         \
    }                                                                 \
    auto& softdelta_params =                                          \
        static_cast<FlashSoftDeltaFwdParams&>(params);                 \
    if (softdelta_params.gate_group_dim == HeadDimV / 4) {             \
      RunAttentionFwdSm90KernelTile<                                    \
          HeadDim, HeadDimV, Element, Element, false, false,           \
          true, BlockM, BlockN, true, true, true, Visibility,          \
          flash::FlashSoftDeltaFwdSm90, 1, true, true,                 \
          FlashSoftDeltaSingleCTADynamicTileScheduler<                 \
              BlockM * 2, 32, HeadGrouping, TileOrder>,                \
          FlashSoftDeltaSingleCTAEpilogueFwd,                 \
          FlashSoftDeltaOutputEpilogueArguments,                       \
          FlashSoftDeltaMainloopAdapterFwdSm90, Stages>(               \
          params, stream);                                             \
      return;                                                          \
    }                                                                  \
    static constexpr std::tuple<int, int, bool, bool> kTile =          \
        FlashSoftDeltaTileSizeFwdSm90<Visibility>(                     \
            HeadDim, HeadDimV);                                        \
    RunAttentionFwdSm90KernelTile<                                      \
        HeadDim, HeadDimV, Element, Element, false, false,             \
        false, std::get<0>(kTile), std::get<1>(kTile),                 \
        std::get<2>(kTile), std::get<3>(kTile), true, Visibility,      \
        flash::FlashSoftDeltaFwdSm90, 2, true, false,                  \
        FlashSoftDeltaPairPersistentTileScheduler,                     \
        flash::CollectiveEpilogueFwd, AttentionFwdEpilogueArguments,    \
        FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);         \
  }                                                                    \
  }                                                                    \
  }

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_FULL_TILE(       \
    Element, HeadDim, HeadDimV, BlockM, BlockN, Stages, HeadGrouping) \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_ORDERED_DYNAMIC_FULL_TILE(     \
      Element, HeadDim, HeadDimV, BlockM, BlockN, Stages,              \
      HeadGrouping, xattn::ops::FlashSoftDeltaL2TileOrder)

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_FULL_TILE( \
    Element, HeadDim, HeadDimV, BlockM, BlockN, Stages, HeadGrouping) \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_FULL_TILE(             \
      Element, HeadDim, HeadDimV, BlockM, BlockN, Stages,              \
      HeadGrouping)                                                    \
  namespace xattn {                                                    \
  namespace ops {                                                      \
  template void RunFlashSoftDeltaFwdSm90VD<                            \
      Element, Element, HeadDim, HeadDimV,                             \
      attention::semantics::CausalSlidingWindowVisibility>(            \
      AttentionFwdParams&, cudaStream_t);                               \
  template void RunFlashSoftDeltaFwdSm90VD<                            \
      Element, Element, HeadDim, HeadDimV,                             \
      attention::semantics::SlidingChunkVisibility>(                   \
      AttentionFwdParams&, cudaStream_t);                               \
  }                                                                    \
  }

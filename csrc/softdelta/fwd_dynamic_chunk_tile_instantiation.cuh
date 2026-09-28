#pragma once

#include "softdelta/fwd_branchless_tile_scheduler.h"
#include "softdelta/fwd_template.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_CHUNK_TILE(  \
    Element, HeadDim, HeadDimV, BlockM, BlockN, Stages)            \
  namespace xattn {                                                \
  namespace ops {                                                  \
  template <>                                                      \
  void RunFlashSoftDeltaFwdSm90VD<                                 \
      Element, Element, HeadDim, HeadDimV,                         \
      attention::semantics::SlidingChunkVisibility>(               \
      AttentionFwdParams& params, cudaStream_t stream) {            \
    using Visibility =                                             \
        attention::semantics::SlidingChunkVisibility;              \
    if (MaybeRunFlashSoftDeltaInterleavedReaderFwdSm90VD<           \
            Element, Element, HeadDim, HeadDimV, Visibility>(       \
            params, stream)) {                                     \
      return;                                                      \
    }                                                              \
    if (params.attention_chunk <= 128) {                           \
      RunAttentionFwdSm90KernelTile<                                \
          HeadDim, HeadDimV, Element, Element, false, false,       \
          false, BlockM, BlockN, true, true, true, Visibility,     \
          flash::FlashSoftDeltaFwdSm90, 1, true, true,             \
          FlashSoftDeltaSingleCTAPersistentTileScheduler,          \
          FlashSoftDeltaSingleCTAEpilogueFwd,             \
          FlashSoftDeltaOutputEpilogueArguments,                   \
          FlashSoftDeltaMainloopAdapterFwdSm90, Stages>(           \
          params, stream);                                         \
      return;                                                      \
    }                                                              \
    RunAttentionFwdSm90KernelTile<                                  \
        HeadDim, HeadDimV, Element, Element, false, false,         \
        true, BlockM, BlockN, true, true, true, Visibility,        \
        flash::FlashSoftDeltaFwdSm90, 1, true, true,               \
        FlashSoftDeltaSingleCTABranchlessTileScheduler<            \
            BlockM * 2, 32, FlashSoftDeltaAllHeadTileGrouping>,    \
        FlashSoftDeltaSingleCTAEpilogueFwd,               \
        FlashSoftDeltaOutputEpilogueArguments,                     \
        FlashSoftDeltaMainloopAdapterFwdSm90, Stages, false,       \
        false, true>(params, stream);                              \
  }                                                                \
  }                                                                \
  }

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_CHUNK_TILE( \
    Element, HeadDim, HeadDimV, BlockM, BlockN, Stages)                \
  namespace xattn {                                                    \
  namespace ops {                                                      \
  template void RunFlashSoftDeltaFwdSm90VD<                            \
      Element, Element, HeadDim, HeadDimV,                             \
      attention::semantics::CausalFullVisibility>(                     \
      AttentionFwdParams&, cudaStream_t);                               \
  template void RunFlashSoftDeltaFwdSm90VD<                            \
      Element, Element, HeadDim, HeadDimV,                             \
      attention::semantics::CausalSlidingWindowVisibility>(            \
      AttentionFwdParams&, cudaStream_t);                               \
  }                                                                    \
  }                                                                    \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_CHUNK_TILE(            \
      Element, HeadDim, HeadDimV, BlockM, BlockN, Stages)

#pragma once

#include "softdelta/fwd_dynamic_chunk_tile_instantiation.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_WINDOW_TILE( \
    Element, HeadDim, HeadDimV, BlockM, BlockN, Stages,            \
    HeadGrouping)                                                   \
  namespace xattn {                                                \
  namespace ops {                                                  \
  template <>                                                      \
  void RunFlashSoftDeltaFwdSm90VD<                                 \
      Element, Element, HeadDim, HeadDimV,                         \
      attention::semantics::CausalSlidingWindowVisibility>(        \
      AttentionFwdParams& params, cudaStream_t stream) {            \
    using Visibility =                                             \
        attention::semantics::CausalSlidingWindowVisibility;       \
    if (MaybeRunFlashSoftDeltaInterleavedReaderFwdSm90VD<           \
            Element, Element, HeadDim, HeadDimV, Visibility>(       \
            params, stream)) {                                     \
      return;                                                      \
    }                                                              \
    RunAttentionFwdSm90KernelTile<                                  \
        HeadDim, HeadDimV, Element, Element, false, false,         \
        true, BlockM, BlockN, true, true, true, Visibility,        \
        flash::FlashSoftDeltaFwdSm90, 1, true, true,               \
        FlashSoftDeltaSingleCTADynamicTileScheduler<               \
            BlockM * 2, 32, HeadGrouping>,                         \
        FlashSoftDeltaSingleCTAEpilogueFwd,               \
        FlashSoftDeltaOutputEpilogueArguments,                     \
        FlashSoftDeltaMainloopAdapterFwdSm90, Stages>(             \
        params, stream);                                           \
  }                                                                \
  }                                                                \
  }

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_WINDOW_CHUNK( \
    Element, HeadDim, HeadDimV, WindowBlockM, WindowBlockN, WindowStages,        \
    WindowHeadGrouping, ChunkBlockM, ChunkBlockN, ChunkStages)                   \
  namespace xattn {                                                              \
  namespace ops {                                                                \
  template void RunFlashSoftDeltaFwdSm90VD<                                      \
      Element, Element, HeadDim, HeadDimV,                                       \
      attention::semantics::CausalFullVisibility>(                               \
      AttentionFwdParams&, cudaStream_t);                                         \
  }                                                                              \
  }                                                                              \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_WINDOW_TILE(                     \
      Element, HeadDim, HeadDimV, WindowBlockM, WindowBlockN,                    \
      WindowStages, WindowHeadGrouping)                                          \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_CHUNK_TILE(                      \
      Element, HeadDim, HeadDimV, ChunkBlockM, ChunkBlockN, ChunkStages)

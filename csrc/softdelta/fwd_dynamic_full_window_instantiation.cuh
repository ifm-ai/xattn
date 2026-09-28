#pragma once

#include "softdelta/fwd_dynamic_full_tile_instantiation.cuh"
#include "softdelta/fwd_dynamic_window_chunk_instantiation.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_FULL_WINDOW( \
    Element, HeadDim, HeadDimV, FullBlockM, FullBlockN, FullStages,             \
    FullHeadGrouping, WindowBlockM, WindowBlockN, WindowStages,                 \
    WindowHeadGrouping)                                                         \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_FULL_TILE(                      \
      Element, HeadDim, HeadDimV, FullBlockM, FullBlockN, FullStages,           \
      FullHeadGrouping)                                                         \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_WINDOW_TILE(                    \
      Element, HeadDim, HeadDimV, WindowBlockM, WindowBlockN,                   \
      WindowStages, WindowHeadGrouping)                                         \
  namespace xattn {                                                             \
  namespace ops {                                                               \
  template void RunFlashSoftDeltaFwdSm90VD<                                     \
      Element, Element, HeadDim, HeadDimV,                                      \
      attention::semantics::SlidingChunkVisibility>(                            \
      AttentionFwdParams&, cudaStream_t);                                        \
  }                                                                             \
  }

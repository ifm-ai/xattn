#pragma once

#include "softdelta/fwd_dynamic_chunk_tile_instantiation.cuh"
#include "softdelta/fwd_dynamic_full_tile_instantiation.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_ORDERED_DYNAMIC_FULL_CHUNK( \
    Element, HeadDim, HeadDimV, FullBlockM, FullBlockN, FullStages,                    \
    FullHeadGrouping, FullTileOrder, ChunkBlockM, ChunkBlockN, ChunkStages)            \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_ORDERED_DYNAMIC_FULL_TILE(                     \
      Element, HeadDim, HeadDimV, FullBlockM, FullBlockN, FullStages,          \
      FullHeadGrouping, FullTileOrder)                                         \
  namespace xattn {                                                            \
  namespace ops {                                                              \
  template void RunFlashSoftDeltaFwdSm90VD<                                    \
      Element, Element, HeadDim, HeadDimV,                                     \
      attention::semantics::CausalSlidingWindowVisibility>(                    \
      AttentionFwdParams&, cudaStream_t);                                       \
  }                                                                            \
  }                                                                            \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_CHUNK_TILE(                    \
      Element, HeadDim, HeadDimV, ChunkBlockM, ChunkBlockN, ChunkStages)

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_FULL_CHUNK( \
    Element, HeadDim, HeadDimV, FullBlockM, FullBlockN, FullStages,            \
    FullHeadGrouping, ChunkBlockM, ChunkBlockN, ChunkStages)                   \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_ORDERED_DYNAMIC_FULL_CHUNK( \
      Element, HeadDim, HeadDimV, FullBlockM, FullBlockN, FullStages,            \
      FullHeadGrouping, xattn::ops::FlashSoftDeltaL2TileOrder,              \
      ChunkBlockM, ChunkBlockN, ChunkStages)

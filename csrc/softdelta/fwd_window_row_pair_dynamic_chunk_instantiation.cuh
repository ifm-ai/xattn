#pragma once

#include "softdelta/fwd_dynamic_chunk_tile_instantiation.cuh"
#include "softdelta/fwd_row_pair_specialization.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_WINDOW_ROW_PAIR_DYNAMIC_CHUNK( \
    Element, HeadDim, HeadDimV, BlockM, BlockN, Stages)                       \
  namespace xattn {                                                           \
  namespace ops {                                                             \
  template void RunFlashSoftDeltaFwdSm90VD<                                   \
      Element, Element, HeadDim, HeadDimV,                                    \
      attention::semantics::CausalFullVisibility>(                            \
      AttentionFwdParams&, cudaStream_t);                                      \
  }                                                                           \
  }                                                                           \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_ROW_PAIR(                              \
      Element, HeadDim, HeadDimV,                                             \
      attention::semantics::CausalSlidingWindowVisibility)                    \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_DYNAMIC_CHUNK_TILE(                    \
      Element, HeadDim, HeadDimV, BlockM, BlockN, Stages)

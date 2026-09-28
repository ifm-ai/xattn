#pragma once

#include "softdelta/fwd_row_pair_specialization.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_WINDOW_ROW_PAIR_TILE( \
    Element, HeadDim, HeadDimV, BlockM, BlockN, Stages)                       \
  XATTN_FLASH_SOFTDELTA_FWD_SM90_DEFINE_ROW_PAIR_TILE(                         \
      Element, HeadDim, HeadDimV,                                             \
      attention::semantics::CausalSlidingWindowVisibility,                    \
      BlockM, BlockN, Stages)                                                  \
  namespace xattn {                                                           \
  namespace ops {                                                             \
  template void RunFlashSoftDeltaFwdSm90VD<                                   \
      Element, Element, HeadDim, HeadDimV,                                    \
      attention::semantics::CausalFullVisibility>(                            \
      AttentionFwdParams&, cudaStream_t);                                      \
  template void RunFlashSoftDeltaFwdSm90VD<                                   \
      Element, Element, HeadDim, HeadDimV,                                    \
      attention::semantics::SlidingChunkVisibility>(                          \
      AttentionFwdParams&, cudaStream_t);                                      \
  }                                                                           \
  }

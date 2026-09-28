#pragma once

#include "softdelta/fwd_chunk_ordered_kernel_sm90.h"
#include "softdelta/fwd_chunk_ordered_tile_scheduler.h"
#include "softdelta/fwd_template.cuh"

#define XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_CHUNK_ORDERED(  \
    Element, HeadDim, HeadDimV)                                    \
  namespace xattn {                                                 \
  namespace ops {                                                   \
  template void RunFlashSoftDeltaFwdSm90VD<                         \
      Element, Element, HeadDim, HeadDimV,                          \
      attention::semantics::CausalFullVisibility>(                  \
      AttentionFwdParams&, cudaStream_t);                            \
  template void RunFlashSoftDeltaFwdSm90VD<                         \
      Element, Element, HeadDim, HeadDimV,                          \
      attention::semantics::CausalSlidingWindowVisibility>(         \
      AttentionFwdParams&, cudaStream_t);                            \
  template <>                                                       \
  void RunFlashSoftDeltaFwdSm90VD<                                  \
      Element, Element, HeadDim, HeadDimV,                          \
      attention::semantics::SlidingChunkVisibility>(                \
      AttentionFwdParams& params, cudaStream_t stream) {              \
    using Visibility =                                              \
        attention::semantics::SlidingChunkVisibility;               \
    if (MaybeRunFlashSoftDeltaInterleavedReaderFwdSm90VD<           \
            Element, Element, HeadDim, HeadDimV, Visibility>(       \
            params, stream)) {                                     \
      return;                                                      \
    }                                                              \
    static constexpr int kBlockN =                                  \
        FlashSoftDeltaSingleCTABlockNFwdSm90(                       \
            HeadDim, HeadDimV, Visibility::kKind);                  \
    static constexpr bool kIntraWGOverlap =                         \
        FlashSoftDeltaSingleCTAIntraWGOverlapFwdSm90(               \
            HeadDim, HeadDimV, Visibility::kKind);                  \
    static_assert(kBlockN == 128);                                  \
    static_assert(kIntraWGOverlap);                                 \
    if (params.attention_chunk <= 128) {                            \
      RunAttentionFwdSm90KernelTile<                                 \
          HeadDim, HeadDimV, Element, Element, false, false,        \
          false, 128, kBlockN, true, kIntraWGOverlap, true,         \
          Visibility, flash::FlashSoftDeltaFwdSm90, 1, true, true, \
          FlashSoftDeltaSingleCTAPersistentTileScheduler,           \
          FlashSoftDeltaSingleCTAEpilogueFwd,              \
          FlashSoftDeltaOutputEpilogueArguments,                    \
          FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);    \
      return;                                                       \
    }                                                               \
    RunAttentionFwdSm90KernelTile<                                   \
        HeadDim, HeadDimV, Element, Element, false, false,          \
        false, 128, kBlockN, true, kIntraWGOverlap, true,           \
        Visibility, flash::FlashSoftDeltaChunkOrderedFwdSm90,       \
        1, true, true,                                              \
        FlashSoftDeltaSingleCTAChunkOrderedTileScheduler,           \
        FlashSoftDeltaSingleCTAEpilogueFwd,                \
        FlashSoftDeltaOutputEpilogueArguments,                      \
        FlashSoftDeltaMainloopAdapterFwdSm90>(params, stream);      \
  }                                                                 \
  }                                                                 \
  }

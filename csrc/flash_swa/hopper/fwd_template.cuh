#pragma once

#include "flash_swa/hopper/fwd.h"

#include "attention/hopper/fwd_template.cuh"
#include "flash_swa/hopper/fwd_kernel_sm90.h"

namespace xattn {
namespace ops {
namespace flash_swa {

template <
    typename Element, typename ElementOut, bool kVarlen,
    bool kHasSegment, int kHeadDim, int kHeadDimV,
    typename Visibility>
void RunFlashSWAFwdSm90VD(
    AttentionFwdParams& params, cudaStream_t stream) {
  static constexpr bool kUseFastExp2 =
      !kVarlen && !kHasSegment;
  RunAttentionFwdSm90Kernel<
      kHeadDim, kHeadDimV, Element, ElementOut,
      kVarlen, kHasSegment, true /*kUsePersistentScheduler*/,
      kUseFastExp2, Visibility,
      flash::FlashSWAFwdSm90>(params, stream);
}

}  // namespace flash_swa
}  // namespace ops
}  // namespace xattn

#define XATTN_FLASH_SWA_FWD_SM90_INSTANTIATE_DENSE_SEGMENT(            \
    Element, HeadDim, HeadDimV)                                       \
  namespace xattn {                                                    \
  namespace ops {                                                      \
  namespace flash_swa {                                                \
  template void RunFlashSWAFwdSm90VD<                                 \
      Element, Element, false, false, HeadDim, HeadDimV,               \
      attention::semantics::CausalSlidingWindowVisibility>(           \
      AttentionFwdParams&, cudaStream_t);                              \
  template void RunFlashSWAFwdSm90VD<                                 \
      Element, Element, false, true, HeadDim, HeadDimV,                \
      attention::semantics::CausalSlidingWindowVisibility>(           \
      AttentionFwdParams&, cudaStream_t);                              \
  }                                                                    \
  }                                                                    \
  }

#define XATTN_FLASH_SWA_FWD_SM90_INSTANTIATE_VARLEN(                   \
    Element, HeadDim, HeadDimV)                                       \
  namespace xattn {                                                    \
  namespace ops {                                                      \
  namespace flash_swa {                                                \
  template void RunFlashSWAFwdSm90VD<                                 \
      Element, Element, true, false, HeadDim, HeadDimV,                \
      attention::semantics::CausalSlidingWindowVisibility>(           \
      AttentionFwdParams&, cudaStream_t);                              \
  }                                                                    \
  }                                                                    \
  }

#define XATTN_FLASH_SWA_FWD_SM90_INSTANTIATE(                          \
    Element, HeadDim, HeadDimV)                                       \
  XATTN_FLASH_SWA_FWD_SM90_INSTANTIATE_DENSE_SEGMENT(                  \
      Element, HeadDim, HeadDimV)                                     \
  XATTN_FLASH_SWA_FWD_SM90_INSTANTIATE_VARLEN(                         \
      Element, HeadDim, HeadDimV)                                     \
  namespace xattn {                                                    \
  namespace ops {                                                      \
  namespace flash_swa {                                                \
  template void RunFlashSWAFwdSm90VD<                                 \
      Element, Element, false, false, HeadDim, HeadDimV,               \
      attention::semantics::CausalFullVisibility>(                    \
      AttentionFwdParams&, cudaStream_t);                              \
  }                                                                    \
  }                                                                    \
  }

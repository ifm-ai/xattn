#pragma once

#include "flash_sca/hopper/fwd.h"

#include "attention/hopper/fwd_template.cuh"

namespace xattn {
namespace ops {

template <
    typename Element, typename ElementOut, bool kVarlen, bool kHasSegment,
    int kHeadDim, int kHeadDimV>
void RunFlashSCAFwdSm90VD(
    AttentionFwdParams& params, cudaStream_t stream) {
  static constexpr bool kUseFastExp2 = !kVarlen && !kHasSegment;
  RunAttentionFwdSm90Kernel<
      kHeadDim, kHeadDimV, Element, ElementOut, kVarlen, kHasSegment,
      true /*kUsePersistentScheduler*/, kUseFastExp2>(params, stream);
}

}  // namespace ops
}  // namespace xattn

#define XATTN_FLASH_SCA_FWD_SM90_INSTANTIATE_CHUNK_GLOBAL(              \
    Element, HeadDim, HeadDimV)                                        \
  namespace xattn {                                                     \
  namespace ops {                                                      \
  template void RunFlashSCAFwdSm90VD<                                  \
      Element, Element, false, false, HeadDim, HeadDimV>(              \
      AttentionFwdParams&, cudaStream_t);                              \
  template void RunFlashSCAFwdSm90VD<                                  \
      Element, Element, false, true, HeadDim, HeadDimV>(               \
      AttentionFwdParams&, cudaStream_t);                              \
  }                                                                    \
  }

#define XATTN_FLASH_SCA_FWD_SM90_INSTANTIATE_CHUNK_RESET(               \
    Element, HeadDim, HeadDimV)                                        \
  namespace xattn {                                                     \
  namespace ops {                                                      \
  template void RunFlashSCAFwdSm90VD<                                  \
      Element, Element, true, false, HeadDim, HeadDimV>(               \
      AttentionFwdParams&, cudaStream_t);                              \
  }                                                                    \
  }

#define XATTN_FLASH_SCA_FWD_SM90_INSTANTIATE(Element, HeadDim, HeadDimV) \
  XATTN_FLASH_SCA_FWD_SM90_INSTANTIATE_CHUNK_GLOBAL(                     \
      Element, HeadDim, HeadDimV)                                       \
  XATTN_FLASH_SCA_FWD_SM90_INSTANTIATE_CHUNK_RESET(                      \
      Element, HeadDim, HeadDimV)

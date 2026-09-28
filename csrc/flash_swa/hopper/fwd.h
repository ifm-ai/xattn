#pragma once

#include <cuda_runtime_api.h>

#include "attention/hopper/fwd_params.h"
#include "attention/semantics/visibility.h"

namespace xattn {
namespace ops {

namespace flash_swa {

template <
    typename Element, typename ElementOut, bool kVarlen,
    bool kHasSegment, int kHeadDim, int kHeadDimV,
    typename Visibility>
void RunFlashSWAFwdSm90VD(
    AttentionFwdParams& params, cudaStream_t stream);

}  // namespace flash_swa
}  // namespace ops
}  // namespace xattn

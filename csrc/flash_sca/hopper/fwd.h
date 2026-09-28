#pragma once

#include <cuda_runtime_api.h>

#include "attention/hopper/fwd_params.h"

namespace xattn {
namespace ops {

template <
    typename Element, typename ElementOut, bool kVarlen, bool kHasSegment,
    int kHeadDim, int kHeadDimV>
void RunFlashSCAFwdSm90VD(
    AttentionFwdParams& params, cudaStream_t stream);

}  // namespace ops
}  // namespace xattn

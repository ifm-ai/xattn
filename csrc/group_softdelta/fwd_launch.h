#pragma once

#include <cuda_runtime_api.h>

#include "attention/hopper/fwd_params.h"

namespace xattn {
namespace ops {

// Explicitly instantiated per (dtype, D, V) in group_softdelta/instantiations.
template <typename Element, int kHeadDim, int kHeadDimV, bool kHasSegment>
void RunGroupSoftDeltaFwdSm90(AttentionFwdParams& params, cudaStream_t stream);

}  // namespace ops
}  // namespace xattn

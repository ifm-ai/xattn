#pragma once

#include <cuda_runtime_api.h>

#include "attention/hopper/fwd_params.h"

namespace xattn {
namespace ops {

// Explicitly instantiated per (dtype, D, V) in softdelta/instantiations: causal
// (CausalFullVisibility dense rows, the CausalSlidingWindowVisibility segment instance
// also serving sliding windows with segments) and, in the *_local files, sliding
// windows on dense rows and sliding chunks (SlidingChunkVisibility).
template <typename Element, int kHeadDim, int kHeadDimV, typename Visibility, bool kHasSegment>
void RunSoftDeltaFwdSm90(AttentionFwdParams& params, cudaStream_t stream);

}  // namespace ops
}  // namespace xattn

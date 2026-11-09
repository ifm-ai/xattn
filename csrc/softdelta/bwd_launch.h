#pragma once

#include <cuda_runtime_api.h>

#include "attention/hopper/bwd_params.h"

namespace xattn {
namespace ops {

// Explicitly instantiated per (dtype, D, V) in softdelta/instantiations
// (kReaderGroups: the softdelta_bwd_groups_* files) with the forward's
// visibility instances (fwd_launch.h).
template <typename Element, int kHeadDim, int kHeadDimV, typename Visibility, bool kHasSegment,
          bool kReaderGroups>
void RunSoftDeltaBwdSm90(AttentionBwdParams& params, cudaStream_t stream);

}  // namespace ops
}  // namespace xattn

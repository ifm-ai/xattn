#pragma once

#include <cuda_runtime_api.h>

#include "attention/hopper/bwd_params.h"

namespace xattn {
namespace ops {

// Explicitly instantiated per (dtype, D, V) in group_softdelta/instantiations
// (kReaderGroups: the group_softdelta_bwd_groups_* files).
template <typename Element, int kHeadDim, int kHeadDimV, bool kHasSegment, bool kReaderGroups>
void RunGroupSoftDeltaBwdSm90(AttentionBwdParams& params, cudaStream_t stream);

}  // namespace ops
}  // namespace xattn

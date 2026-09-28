#pragma once

#include "attention/partition/segment_index.h"

namespace xattn {
namespace ops {
namespace attention {
namespace hopper {
namespace semantics {

template <int kBlockM, int kBlockN>
using SegmentPartitionAdapter =
    attention::partition::SegmentPartitionPolicy<kBlockM, kBlockN>;

template <bool HasSegment>
using SegmentPartitionStorage =
    attention::partition::SegmentMaskStorage<HasSegment>;

}  // namespace semantics
}  // namespace hopper
}  // namespace attention
}  // namespace ops
}  // namespace xattn

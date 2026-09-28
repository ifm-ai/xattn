#pragma once

#include "attention/partition/segment_index.h"
#include "attention/semantics/identity.h"

namespace xattn {
namespace ops {
namespace attention {
namespace partition {

template <typename PartitionSemantic, int kBlockM, int kBlockN>
struct PolicyFor;

template <int kBlockM, int kBlockN>
struct PolicyFor<semantics::NoPartition, kBlockM, kBlockN> {
  using Type = NoPartitionPolicy;
};

template <int kBlockM, int kBlockN>
struct PolicyFor<semantics::SegmentIndexPartition, kBlockM, kBlockN> {
  using Type = SegmentPartitionPolicy<kBlockM, kBlockN>;
};

template <int kBlockM, int kBlockN>
struct DenseOrSegmentPolicyBundle {
  using Dense = typename PolicyFor<
      semantics::NoPartition, kBlockM, kBlockN>::Type;
  using Segment = typename PolicyFor<
      semantics::SegmentIndexPartition, kBlockM, kBlockN>::Type;
};

}  // namespace partition
}  // namespace attention
}  // namespace ops
}  // namespace xattn

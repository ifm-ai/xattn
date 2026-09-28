// Author: Shicheng Wen

#pragma once

#include "attention/semantics/identity.h"
#include "attention/semantics/visibility.h"

namespace xattn {
namespace ops {
namespace softdelta {
namespace semantics {

template <typename Direction, typename Visibility, typename Position,
          typename Partition, typename Route, typename Determinism>
using SoftDeltaSemantic =
    attention::semantics::AttentionSemanticIdentity<
        Direction, attention::semantics::SoftDeltaFormula, Visibility,
        attention::semantics::InclusiveStrictPastBoundaryPair, Position,
        Partition, Route, attention::semantics::RuntimeHeadMapping,
        Determinism>;

template <typename Direction, typename Determinism>
using SoftDeltaFullSemantic = SoftDeltaSemantic<
    Direction, attention::semantics::CausalFullVisibility,
    attention::semantics::GlobalPosition,
    attention::semantics::DenseOrSegmentPartitionBundle,
    attention::semantics::DenseOrSegmentRouteBundle, Determinism>;

template <typename Direction, typename Determinism>
using SoftDeltaSlidingWindowSemantic = SoftDeltaSemantic<
    Direction, attention::semantics::CausalSlidingWindowVisibility,
    attention::semantics::GlobalPosition,
    attention::semantics::DenseOrSegmentPartitionBundle,
    attention::semantics::DenseOrSegmentRouteBundle, Determinism>;

template <typename Direction, typename Determinism>
using SoftDeltaSlidingChunkSemantic = SoftDeltaSemantic<
    Direction, attention::semantics::SlidingChunkVisibility,
    attention::semantics::GlobalOrResetPositionBundle,
    attention::semantics::DenseOrSegmentPartitionBundle,
    attention::semantics::DenseOrSegmentRouteBundle, Determinism>;

using SoftDeltaFullFwdSemantic = SoftDeltaFullSemantic<
    attention::semantics::ForwardDirection,
    attention::semantics::NoDeterminism>;
using SoftDeltaFullBwdSemantic = SoftDeltaFullSemantic<
    attention::semantics::BackwardDirection,
    attention::semantics::RuntimeBackwardDeterminism>;

using SoftDeltaSlidingWindowFwdSemantic =
    SoftDeltaSlidingWindowSemantic<
        attention::semantics::ForwardDirection,
        attention::semantics::NoDeterminism>;
using SoftDeltaSlidingWindowBwdSemantic =
    SoftDeltaSlidingWindowSemantic<
        attention::semantics::BackwardDirection,
        attention::semantics::RuntimeBackwardDeterminism>;

using SoftDeltaSlidingChunkFwdSemantic =
    SoftDeltaSlidingChunkSemantic<
        attention::semantics::ForwardDirection,
        attention::semantics::NoDeterminism>;
using SoftDeltaSlidingChunkBwdSemantic =
    SoftDeltaSlidingChunkSemantic<
        attention::semantics::BackwardDirection,
        attention::semantics::RuntimeBackwardDeterminism>;

}  // namespace semantics
}  // namespace softdelta
}  // namespace ops
}  // namespace xattn

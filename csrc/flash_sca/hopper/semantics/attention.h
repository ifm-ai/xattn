// Author: Shicheng Wen

#pragma once

#include "attention/semantics/identity.h"
#include "attention/semantics/visibility.h"

namespace xattn {
namespace ops {
namespace flash_sca {
namespace semantics {

using attention::semantics::DirectionKind;
using attention::semantics::RouteKind;

using ScaDenseFwdSemantic =
    attention::semantics::AttentionSemanticIdentity<
        attention::semantics::ForwardDirection,
        attention::semantics::StandardFormula,
        attention::semantics::SlidingChunkVisibility,
        attention::semantics::InclusiveBoundary,
        attention::semantics::GlobalOrResetPositionBundle,
        attention::semantics::DenseOrSegmentPartitionBundle,
        attention::semantics::DenseOrSegmentRouteBundle,
        attention::semantics::RuntimeHeadMapping,
        attention::semantics::NoDeterminism>;

using ScaVarlenFwdSemantic =
    attention::semantics::AttentionSemanticIdentity<
        attention::semantics::ForwardDirection,
        attention::semantics::StandardFormula,
        attention::semantics::SlidingChunkVisibility,
        attention::semantics::InclusiveBoundary,
        attention::semantics::GlobalOrResetPositionBundle,
        attention::semantics::VarlenPartition,
        attention::semantics::VarlenRoute,
        attention::semantics::RuntimeHeadMapping,
        attention::semantics::NoDeterminism>;

using ScaDenseBwdSemantic =
    attention::semantics::AttentionSemanticIdentity<
        attention::semantics::BackwardDirection,
        attention::semantics::StandardFormula,
        attention::semantics::SlidingChunkVisibility,
        attention::semantics::InclusiveBoundary,
        attention::semantics::GlobalOrResetPositionBundle,
        attention::semantics::DenseOrSegmentPartitionBundle,
        attention::semantics::DenseOrSegmentRouteBundle,
        attention::semantics::RuntimeHeadMapping,
        attention::semantics::RuntimeBackwardDeterminism>;

using ScaVarlenBwdSemantic =
    attention::semantics::AttentionSemanticIdentity<
        attention::semantics::BackwardDirection,
        attention::semantics::StandardFormula,
        attention::semantics::SlidingChunkVisibility,
        attention::semantics::InclusiveBoundary,
        attention::semantics::GlobalOrResetPositionBundle,
        attention::semantics::VarlenPartition,
        attention::semantics::VarlenRoute,
        attention::semantics::RuntimeHeadMapping,
        attention::semantics::RuntimeBackwardDeterminism>;

}  // namespace semantics
}  // namespace flash_sca
}  // namespace ops
}  // namespace xattn

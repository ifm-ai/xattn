// Author: Shicheng Wen

#pragma once

#include "attention/semantics/identity.h"
#include "attention/semantics/visibility.h"

namespace xattn {
namespace ops {
namespace flash_swa {
namespace semantics {

using FlashSWADenseFwdSemantic =
    attention::semantics::AttentionSemanticIdentity<
        attention::semantics::ForwardDirection,
        attention::semantics::StandardFormula,
        attention::semantics::CausalSlidingWindowVisibility,
        attention::semantics::InclusiveBoundary,
        attention::semantics::GlobalPosition,
        attention::semantics::DenseOrSegmentPartitionBundle,
        attention::semantics::DenseOrSegmentRouteBundle,
        attention::semantics::RuntimeHeadMapping,
        attention::semantics::NoDeterminism>;

using FlashSWAVarlenFwdSemantic =
    attention::semantics::AttentionSemanticIdentity<
        attention::semantics::ForwardDirection,
        attention::semantics::StandardFormula,
        attention::semantics::CausalSlidingWindowVisibility,
        attention::semantics::InclusiveBoundary,
        attention::semantics::GlobalPosition,
        attention::semantics::VarlenPartition,
        attention::semantics::VarlenRoute,
        attention::semantics::RuntimeHeadMapping,
        attention::semantics::NoDeterminism>;

using FlashSWADenseBwdSemantic =
    attention::semantics::AttentionSemanticIdentity<
        attention::semantics::BackwardDirection,
        attention::semantics::StandardFormula,
        attention::semantics::CausalSlidingWindowVisibility,
        attention::semantics::InclusiveBoundary,
        attention::semantics::GlobalPosition,
        attention::semantics::DenseOrSegmentPartitionBundle,
        attention::semantics::DenseOrSegmentRouteBundle,
        attention::semantics::RuntimeHeadMapping,
        attention::semantics::RuntimeBackwardDeterminism>;

using FlashSWAVarlenBwdSemantic =
    attention::semantics::AttentionSemanticIdentity<
        attention::semantics::BackwardDirection,
        attention::semantics::StandardFormula,
        attention::semantics::CausalSlidingWindowVisibility,
        attention::semantics::InclusiveBoundary,
        attention::semantics::GlobalPosition,
        attention::semantics::VarlenPartition,
        attention::semantics::VarlenRoute,
        attention::semantics::RuntimeHeadMapping,
        attention::semantics::RuntimeBackwardDeterminism>;

}  // namespace semantics
}  // namespace flash_swa
}  // namespace ops
}  // namespace xattn

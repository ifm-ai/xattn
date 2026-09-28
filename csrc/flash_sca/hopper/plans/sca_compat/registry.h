#pragma once

#include "flash_sca/hopper/plans/sca_compat/plan.h"

namespace xattn {
namespace ops {
namespace flash_sca {
namespace plans {

template <
    semantics::DirectionKind Direction, semantics::RouteKind Route>
struct ScaCompatRegistryEntrySm90 {
  static_assert(
      Direction != Direction,
      "unsupported FlashSCA SM90 compatibility registry entry");
};

template <>
struct ScaCompatRegistryEntrySm90<
    semantics::DirectionKind::kForward,
    semantics::RouteKind::kDenseOrSegmentBundle> {
  using Semantic = semantics::ScaDenseFwdSemantic;
  using Plan = ScaDenseFwdCompatPlanSm90;
  static constexpr bool kVarlen = Plan::kVarlen;
  static constexpr bool kAllowsSegment = Plan::kAllowsSegment;
  static constexpr bool kRuntimeDeterminism =
      Plan::kRuntimeDeterminism;
  static constexpr const char* kStablePlanId =
      "sca_compat_fwd_dense_segment";
  static constexpr const char* kStableInstanceRouteBundleId =
      "sca_compat_fwd_dense_segment_varlen";
};

template <>
struct ScaCompatRegistryEntrySm90<
    semantics::DirectionKind::kForward,
    semantics::RouteKind::kVarlen> {
  using Semantic = semantics::ScaVarlenFwdSemantic;
  using Plan = ScaVarlenFwdCompatPlanSm90;
  static constexpr bool kVarlen = Plan::kVarlen;
  static constexpr bool kAllowsSegment = Plan::kAllowsSegment;
  static constexpr bool kRuntimeDeterminism =
      Plan::kRuntimeDeterminism;
  static constexpr const char* kStablePlanId =
      "sca_compat_fwd_varlen";
  static constexpr const char* kStableInstanceRouteBundleId =
      "sca_compat_fwd_dense_segment_varlen";
};

template <>
struct ScaCompatRegistryEntrySm90<
    semantics::DirectionKind::kBackward,
    semantics::RouteKind::kDenseOrSegmentBundle> {
  using Semantic = semantics::ScaDenseBwdSemantic;
  using Plan = ScaDenseBwdCompatPlanSm90;
  static constexpr bool kVarlen = Plan::kVarlen;
  static constexpr bool kAllowsSegment = Plan::kAllowsSegment;
  static constexpr bool kRuntimeDeterminism =
      Plan::kRuntimeDeterminism;
  static constexpr const char* kStablePlanId =
      "sca_compat_bwd_dense_segment";
  static constexpr const char* kStableInstanceRouteBundleId =
      "sca_compat_bwd_dense_segment_varlen";
};

template <>
struct ScaCompatRegistryEntrySm90<
    semantics::DirectionKind::kBackward,
    semantics::RouteKind::kVarlen> {
  using Semantic = semantics::ScaVarlenBwdSemantic;
  using Plan = ScaVarlenBwdCompatPlanSm90;
  static constexpr bool kVarlen = Plan::kVarlen;
  static constexpr bool kAllowsSegment = Plan::kAllowsSegment;
  static constexpr bool kRuntimeDeterminism =
      Plan::kRuntimeDeterminism;
  static constexpr const char* kStablePlanId =
      "sca_compat_bwd_varlen";
  static constexpr const char* kStableInstanceRouteBundleId =
      "sca_compat_bwd_dense_segment_varlen";
};

using ScaDenseFwdCompatRegistryEntrySm90 =
    ScaCompatRegistryEntrySm90<
        semantics::DirectionKind::kForward,
        semantics::RouteKind::kDenseOrSegmentBundle>;
using ScaVarlenFwdCompatRegistryEntrySm90 =
    ScaCompatRegistryEntrySm90<
        semantics::DirectionKind::kForward,
        semantics::RouteKind::kVarlen>;
using ScaDenseBwdCompatRegistryEntrySm90 =
    ScaCompatRegistryEntrySm90<
        semantics::DirectionKind::kBackward,
        semantics::RouteKind::kDenseOrSegmentBundle>;
using ScaVarlenBwdCompatRegistryEntrySm90 =
    ScaCompatRegistryEntrySm90<
        semantics::DirectionKind::kBackward,
        semantics::RouteKind::kVarlen>;

}  // namespace plans
}  // namespace flash_sca
}  // namespace ops
}  // namespace xattn

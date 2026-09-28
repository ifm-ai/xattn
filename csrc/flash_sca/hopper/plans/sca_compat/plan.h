#pragma once

#include "attention/partition/composition.h"
#include "flash_sca/hopper/semantics/attention.h"

namespace xattn {
namespace ops {
namespace flash_sca {
namespace plans {

template <typename Semantic>
struct ScaCompatPlanSm90 {
  static_assert(
      sizeof(Semantic) == 0,
      "unsupported FlashSCA SM90 compatibility semantic combination");
};

template <>
struct ScaCompatPlanSm90<semantics::ScaDenseFwdSemantic> {
  using Semantic = semantics::ScaDenseFwdSemantic;
  static constexpr bool kVarlen = false;
  static constexpr bool kAllowsSegment = true;
  static constexpr bool kRuntimeDeterminism = false;

  template <int kBlockM, int kBlockN>
  using PartitionLowering =
      attention::partition::DenseOrSegmentPolicyBundle<
          kBlockM, kBlockN>;
};

template <>
struct ScaCompatPlanSm90<semantics::ScaVarlenFwdSemantic> {
  static constexpr bool kVarlen = true;
  static constexpr bool kAllowsSegment = false;
  static constexpr bool kRuntimeDeterminism = false;
};

template <>
struct ScaCompatPlanSm90<semantics::ScaDenseBwdSemantic> {
  using Semantic = semantics::ScaDenseBwdSemantic;
  static constexpr bool kVarlen = false;
  static constexpr bool kAllowsSegment = true;
  static constexpr bool kRuntimeDeterminism = true;

  template <int kBlockM, int kBlockN>
  using PartitionLowering =
      attention::partition::DenseOrSegmentPolicyBundle<
          kBlockM, kBlockN>;
};

template <>
struct ScaCompatPlanSm90<semantics::ScaVarlenBwdSemantic> {
  static constexpr bool kVarlen = true;
  static constexpr bool kAllowsSegment = false;
  static constexpr bool kRuntimeDeterminism = true;
};

using ScaDenseFwdCompatPlanSm90 =
    ScaCompatPlanSm90<semantics::ScaDenseFwdSemantic>;
using ScaVarlenFwdCompatPlanSm90 =
    ScaCompatPlanSm90<semantics::ScaVarlenFwdSemantic>;
using ScaDenseBwdCompatPlanSm90 =
    ScaCompatPlanSm90<semantics::ScaDenseBwdSemantic>;
using ScaVarlenBwdCompatPlanSm90 =
    ScaCompatPlanSm90<semantics::ScaVarlenBwdSemantic>;

}  // namespace plans
}  // namespace flash_sca
}  // namespace ops
}  // namespace xattn

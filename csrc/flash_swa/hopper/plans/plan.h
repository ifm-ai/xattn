#pragma once

#include "attention/partition/composition.h"
#include "flash_swa/hopper/semantics/attention.h"

namespace xattn {
namespace ops {
namespace flash_swa {
namespace plans {

template <typename Semantic>
struct FlashSWAPlanSm90 {
  static_assert(
      sizeof(Semantic) == 0,
      "unsupported FlashSWA SM90 semantic");
};

template <>
struct FlashSWAPlanSm90<semantics::FlashSWADenseFwdSemantic> {
  using Semantic = semantics::FlashSWADenseFwdSemantic;
  static constexpr bool kVarlen = false;
  static constexpr bool kAllowsSegment = true;
  static constexpr bool kRuntimeDeterminism = false;

  template <int kBlockM, int kBlockN>
  using PartitionLowering =
      attention::partition::DenseOrSegmentPolicyBundle<
          kBlockM, kBlockN>;
};

template <>
struct FlashSWAPlanSm90<semantics::FlashSWAVarlenFwdSemantic> {
  using Semantic = semantics::FlashSWAVarlenFwdSemantic;
  static constexpr bool kVarlen = true;
  static constexpr bool kAllowsSegment = false;
  static constexpr bool kRuntimeDeterminism = false;
};

template <>
struct FlashSWAPlanSm90<semantics::FlashSWADenseBwdSemantic> {
  using Semantic = semantics::FlashSWADenseBwdSemantic;
  static constexpr bool kVarlen = false;
  static constexpr bool kAllowsSegment = true;
  static constexpr bool kRuntimeDeterminism = true;

  template <int kBlockM, int kBlockN>
  using PartitionLowering =
      attention::partition::DenseOrSegmentPolicyBundle<
          kBlockM, kBlockN>;
};

template <>
struct FlashSWAPlanSm90<semantics::FlashSWAVarlenBwdSemantic> {
  using Semantic = semantics::FlashSWAVarlenBwdSemantic;
  static constexpr bool kVarlen = true;
  static constexpr bool kAllowsSegment = false;
  static constexpr bool kRuntimeDeterminism = true;
};

using FlashSWADenseFwdPlanSm90 =
    FlashSWAPlanSm90<semantics::FlashSWADenseFwdSemantic>;
using FlashSWAVarlenFwdPlanSm90 =
    FlashSWAPlanSm90<semantics::FlashSWAVarlenFwdSemantic>;
using FlashSWADenseBwdPlanSm90 =
    FlashSWAPlanSm90<semantics::FlashSWADenseBwdSemantic>;
using FlashSWAVarlenBwdPlanSm90 =
    FlashSWAPlanSm90<semantics::FlashSWAVarlenBwdSemantic>;

}  // namespace plans
}  // namespace flash_swa
}  // namespace ops
}  // namespace xattn

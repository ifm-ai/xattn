#pragma once

#include "flash_swa/hopper/plans/plan.h"

namespace xattn {
namespace ops {
namespace flash_swa {
namespace plans {

struct FlashSWADenseFwdRegistrySm90 {
  using Semantic = semantics::FlashSWADenseFwdSemantic;
  using Plan = FlashSWADenseFwdPlanSm90;
};

struct FlashSWAVarlenFwdRegistrySm90 {
  using Semantic = semantics::FlashSWAVarlenFwdSemantic;
  using Plan = FlashSWAVarlenFwdPlanSm90;
};

struct FlashSWADenseBwdRegistrySm90 {
  using Semantic = semantics::FlashSWADenseBwdSemantic;
  using Plan = FlashSWADenseBwdPlanSm90;
};

struct FlashSWAVarlenBwdRegistrySm90 {
  using Semantic = semantics::FlashSWAVarlenBwdSemantic;
  using Plan = FlashSWAVarlenBwdPlanSm90;
};

}  // namespace plans
}  // namespace flash_swa
}  // namespace ops
}  // namespace xattn

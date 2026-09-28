#pragma once

#include "attention/semantics/visibility.h"

namespace xattn {
namespace ops {

constexpr bool FlashSoftDeltaUseSingleCTAFwdSm90(
    int qk_dim, int value_dim, int gate_groups,
    attention::semantics::VisibilityKind visibility) {
  return gate_groups == 4 &&
      ((qk_dim <= 64 && value_dim >= 96) ||
       ((qk_dim == 96 || qk_dim == 128) && value_dim >= 128) ||
       (qk_dim == 160 &&
        (value_dim == 32 || value_dim >= 128)) ||
       (qk_dim == 192 &&
        (value_dim == 32 || value_dim == 128 ||
         value_dim >= 192)) ||
       (qk_dim == 256 && value_dim == 256) ||
       (visibility ==
            attention::semantics::VisibilityKind::kCausalFull &&
        qk_dim == 128 && value_dim == 96) ||
       (visibility ==
            attention::semantics::VisibilityKind::kSlidingChunk &&
        ((qk_dim == 64 && value_dim == 64) ||
         (qk_dim == 128 && value_dim <= 128) ||
         (qk_dim == 160 &&
          (value_dim == 64 || value_dim == 96)) ||
         (qk_dim == 192 &&
          (value_dim == 64 || value_dim == 96)) ||
         (qk_dim == 256 && value_dim <= 192))));
}

constexpr bool FlashSoftDeltaUsePairCTAFwdSm90(
    int qk_dim, int value_dim, int gate_groups,
    attention::semantics::VisibilityKind visibility) {
  return gate_groups == 4 &&
      visibility ==
          attention::semantics::VisibilityKind::kCausalFull &&
      ((qk_dim == 128 && value_dim == 128) ||
       (qk_dim == 256 && value_dim == 256));
}

constexpr bool FlashSoftDeltaUseClusteredRowPairFwdSm90(
    int qk_dim, int value_dim, int gate_groups,
    attention::semantics::VisibilityKind visibility) {
  return gate_groups == 4 && qk_dim == 256 && value_dim == 256 &&
      visibility ==
          attention::semantics::VisibilityKind::kCausalFull;
}

constexpr bool FlashSoftDeltaUseSingleCTARowPairFwdSm90(
    int qk_dim, int value_dim, int gate_groups,
    attention::semantics::VisibilityKind visibility) {
  return gate_groups == 4 &&
      ((((qk_dim == 32 && value_dim == 64) ||
         (qk_dim == 64 && (value_dim == 32 || value_dim == 64))) &&
        visibility ==
            attention::semantics::VisibilityKind::kCausalFull) ||
       ((((qk_dim == 32 || qk_dim == 64) &&
          (value_dim == 32 || value_dim == 64)) ||
         (qk_dim == 96 && value_dim == 96)) &&
        visibility ==
            attention::semantics::VisibilityKind::
                kCausalSlidingWindow) ||
       (qk_dim == 32 && value_dim == 64 &&
        visibility ==
            attention::semantics::VisibilityKind::kSlidingChunk));
}

constexpr bool FlashSoftDeltaUseDirectOutputFwdSm90(
    int qk_dim, int value_dim, int gate_groups,
    attention::semantics::VisibilityKind visibility) {
  return
      (FlashSoftDeltaUseSingleCTAFwdSm90(
           qk_dim, value_dim, gate_groups, visibility) &&
       !FlashSoftDeltaUsePairCTAFwdSm90(
           qk_dim, value_dim, gate_groups, visibility)) ||
      FlashSoftDeltaUseSingleCTARowPairFwdSm90(
          qk_dim, value_dim, gate_groups, visibility) ||
      FlashSoftDeltaUseClusteredRowPairFwdSm90(
          qk_dim, value_dim, gate_groups, visibility);
}

constexpr int FlashSoftDeltaSingleCTABlockNFwdSm90(
    int qk_dim, int value_dim,
    attention::semantics::VisibilityKind visibility) {
  if (visibility !=
      attention::semantics::VisibilityKind::kCausalFull) {
    if (qk_dim == 256 && value_dim == 256) {
      return 64;
    }
    return value_dim <= 128 ? 128 : 96;
  }
  if ((qk_dim == 160 || qk_dim == 192) && value_dim == 32) {
    return 192;
  }
  if (qk_dim == 192 && value_dim == 128) {
    return 96;
  }
  if (qk_dim == 160 && value_dim == 192) {
    return 96;
  }
  if (qk_dim == 192 && value_dim == 192) {
    return 112;
  }
  if (qk_dim == 160 && value_dim == 256) {
    return 112;
  }
  if (qk_dim == 192 && value_dim == 256) {
    return 96;
  }
  if (qk_dim == 256 && value_dim == 256) {
    return 80;
  }
  return 128;
}

constexpr bool FlashSoftDeltaSingleCTAIntraWGOverlapFwdSm90(
    int qk_dim, int value_dim,
    attention::semantics::VisibilityKind visibility) {
  return (((qk_dim == 192 || qk_dim == 256) && value_dim == 256) &&
          visibility ==
              attention::semantics::VisibilityKind::kCausalFull) ||
      visibility !=
          attention::semantics::VisibilityKind::kCausalFull ||
      value_dim < 256;
}

}  // namespace ops
}  // namespace xattn

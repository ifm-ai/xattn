#pragma once

namespace xattn::ops::attention::precision {
// Values are part of the public Python API. Do not reinterpret this as bool.
enum class Level : int {
  kLow = 0,
  kOutputState = 1,
  kStable = 2,
  kConservative = 3,
  kProtected = 4,
};
constexpr bool IsValidLevel(int value) { return value >= 0 && value <= 4; }
template <int Value> struct Policy {
  static_assert(IsValidLevel(Value), "high_precision_level must be in [0, 4]");
  static constexpr bool kSaveFp32State = Value != 0;
  static constexpr bool kSubtractBeforeScale = Value != 0;
  static constexpr bool kSplitStatistics = Value >= 2;
  static constexpr bool kAnchoredDelta = Value >= 2;
  static constexpr bool kRepairScoreMass = Value >= 3;
  static constexpr bool kDualOperands = Value >= 4;
  static constexpr bool kRepairProbability = Value >= 4;
  static constexpr bool kProtectScoreRange = Value >= 3;
  static constexpr bool kProtectRange = Value >= 4;
};
}  // namespace xattn::ops::attention::precision

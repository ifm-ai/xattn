// Author: Shicheng Wen

#pragma once

#include <cstdint>

namespace xattn {
namespace ops {
namespace attention {
namespace semantics {

enum class VisibilityKind : std::uint8_t {
  kSlidingChunk,
  kCausalSlidingWindow,
  kCausalFull,
};

struct SlidingChunkVisibility {
  static constexpr VisibilityKind kKind = VisibilityKind::kSlidingChunk;
  static constexpr bool kIsCausal = false;
  static constexpr bool kIsLocal = true;
  static constexpr bool kIsSlidingChunk = true;
  static constexpr bool kUseCausalMask = false;
  static constexpr bool kUseLocalMask = true;
};

struct CausalSlidingWindowVisibility {
  static constexpr VisibilityKind kKind =
      VisibilityKind::kCausalSlidingWindow;
  static constexpr bool kIsCausal = true;
  static constexpr bool kIsLocal = true;
  static constexpr bool kIsSlidingChunk = false;
  static constexpr bool kUseCausalMask = false;
  static constexpr bool kUseLocalMask = true;
};

struct CausalFullVisibility {
  static constexpr VisibilityKind kKind = VisibilityKind::kCausalFull;
  static constexpr bool kIsCausal = true;
  static constexpr bool kIsLocal = false;
  static constexpr bool kIsSlidingChunk = false;
  static constexpr bool kUseCausalMask = true;
  static constexpr bool kUseLocalMask = false;
};

}  // namespace semantics
}  // namespace attention
}  // namespace ops
}  // namespace xattn

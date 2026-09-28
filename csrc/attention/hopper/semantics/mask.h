#pragma once

#include <algorithm>

#include "cutlass/cutlass.h"

#include "attention/hopper/mask.h"

namespace xattn {
namespace ops {
namespace attention {
namespace hopper {
namespace semantics {

struct SlidingChunkExactMaskPredicate {
  CUTLASS_HOST_DEVICE
  static int floor_div(int dividend, int divisor) {
    return dividend >= 0
        ? dividend / divisor
        : -1 - (-1 - dividend) / divisor;
  }

  CUTLASS_HOST_DEVICE
  static int round_down(int dividend, int divisor) {
    return floor_div(dividend, divisor) * divisor;
  }

  CUTLASS_HOST_DEVICE
  static int round_up(int dividend, int divisor) {
    return floor_div(dividend - 1, divisor) * divisor + divisor;
  }

  CUTLASS_HOST_DEVICE
  static bool allows(
      int q_pos, int k_pos, int seqlen_q, int seqlen_k,
      int window_size_left, int window_size_right,
      int sink_token_length, int attention_chunk,
      int chunk_position_offset) {
    if (q_pos < 0 || q_pos >= seqlen_q ||
        k_pos < 0 || k_pos >= seqlen_k) {
      return false;
    }

    int const q_aligned_to_k = q_pos + seqlen_k - seqlen_q;
    int right_inclusive = q_aligned_to_k + window_size_right;
    int left_inclusive = q_aligned_to_k - window_size_left;
    if (attention_chunk > 0) {
      right_inclusive = std::min(
          right_inclusive,
          round_up(
              q_aligned_to_k + 1 + chunk_position_offset,
              attention_chunk) -
              chunk_position_offset - 1);
      left_inclusive = std::max(
          left_inclusive,
          round_down(
              q_aligned_to_k + chunk_position_offset,
              attention_chunk) -
              chunk_position_offset - attention_chunk);
    }
    return k_pos <= right_inclusive &&
        (k_pos < sink_token_length || k_pos >= left_inclusive);
  }

  CUTLASS_HOST_DEVICE
  static bool allows_with_reset(
      int q_pos, int k_pos, int seqlen_q, int seqlen_k,
      int window_size_left, int window_size_right,
      int sink_token_length, int attention_chunk,
      int chunk_position_offset, int reset_chunk_left) {
    return allows(
               q_pos, k_pos, seqlen_q, seqlen_k,
               window_size_left, window_size_right, sink_token_length,
               attention_chunk, chunk_position_offset) &&
        k_pos >= reset_chunk_left;
  }
};

template <
    bool IsSlidingChunk,
    int kBlockM, int kBlockN, bool PackGQA, typename TiledMma,
    bool SwapAB = false, bool HasSegment = false,
    bool DenseTileFastPath = false,
    bool DirectChunkTileFastPath = IsSlidingChunk,
    bool AllowReset = HasSegment,
    int BranchPairLogicalBlockM = 0>
using AttentionMaskAdapter =
    flash::Mask<
        kBlockM, kBlockN, PackGQA, TiledMma, SwapAB, HasSegment,
        DenseTileFastPath, IsSlidingChunk, DirectChunkTileFastPath,
        AllowReset, BranchPairLogicalBlockM>;

}  // namespace semantics
}  // namespace hopper
}  // namespace attention
}  // namespace ops
}  // namespace xattn

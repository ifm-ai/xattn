#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include "attention/cuda/detail/cute_compat.h"

namespace xattn {
namespace ops {
namespace attention {
namespace hopper {
namespace semantics {

struct SlidingChunkPositionAdapter {
  CUTLASS_DEVICE
  static int qk_offset(int seqlen_q, int seqlen_k) {
    return seqlen_k - seqlen_q;
  }

  CUTLASS_DEVICE
  static int chunk_position_offset(
      int const* q_position_offsets, int bidb, int seqlen_q,
      int seqlen_k) {
    return q_position_offsets == nullptr
        ? 0
        : q_position_offsets[bidb] - qk_offset(seqlen_q, seqlen_k);
  }

  CUTLASS_DEVICE
  static int reset_chunk_left(
      int q_pos, int seqlen_q, int seqlen_k, int q_chunk_pos,
      cutlass::FastDivmod const& reset_attention_chunk_divmod) {
    return q_pos + qk_offset(seqlen_q, seqlen_k) - q_chunk_pos +
        flash::round_down(
               reset_attention_chunk_divmod, q_chunk_pos) -
        reset_attention_chunk_divmod.divisor;
  }
};

}  // namespace semantics
}  // namespace hopper
}  // namespace attention
}  // namespace ops
}  // namespace xattn

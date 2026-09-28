#pragma once

#include <cstdint>

namespace xattn {
namespace ops {

// Shared parameter transport for SM90 attention backward kernels. Visibility,
// boundary, and formula semantics are supplied separately by template policy.
struct AttentionBwdParams {
  using index_t = int64_t;

  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;
  void* __restrict__ dy_ptr;
  void* __restrict__ y_ptr;
  void* __restrict__ lse_ptr;
  void* __restrict__ lse_log2_ptr;
  void* __restrict__ dpsum_ptr;
  void* __restrict__ dq_accum_ptr;
  void* __restrict__ dq_ptr;
  void* __restrict__ dk_ptr;
  void* __restrict__ dv_ptr;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t dy_batch_stride;
  index_t y_batch_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t dy_row_stride;
  index_t y_row_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t dy_head_stride;
  index_t y_head_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;

  int b;
  int num_sequences;
  int seqlen_q;
  int seqlen_k;
  int max_seqlen_q;
  int max_seqlen_k;
  int seqlen_q_padded;
  int d;
  int dv;
  int h;
  int h_k;
  float scale_softmax;
  int window_size_left;
  int window_size_right;
  int attention_chunk;
  const int64_t* __restrict__ q_segment_idx;
  const int64_t* __restrict__ k_segment_idx;
  int k_segment_len;
  const int* __restrict__ q_position_offsets;
  const int* __restrict__ q_chunk_positions;
  int reset_attention_chunk;
  const int* __restrict__ cu_seqlens_q;
  const int* __restrict__ cu_seqlens_k;
  const int* __restrict__ seqused_k;
  int* __restrict__ dq_semaphore;
  int* __restrict__ tile_count_semaphore;
  int num_sm;

  // FP32 backward-state flag; preceding field offsets are fixed.
  bool y_is_fp32;
  // Optional per-head boundary adjustment for interleaved paired readers.
  // Even heads use window_size_right; odd heads add this delta.
  int odd_head_window_right_delta;
};

}  // namespace ops
}  // namespace xattn

#pragma once

#include <cstdint>

namespace xattn {
namespace ops {

// Shared parameter transport for SM90 attention forward kernels. Visibility,
// boundary, and formula semantics are supplied separately by template policy.
struct AttentionFwdParams {
  using index_t = int64_t;

  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;
  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ softmax_lseaccum_ptr;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  index_t oaccum_split_stride;
  index_t oaccum_batch_stride;
  index_t oaccum_row_stride;
  index_t oaccum_head_stride;
  index_t lseaccum_split_stride;
  index_t lseaccum_batch_stride;
  index_t lseaccum_head_stride;

  int b;
  int num_sequences;
  int seqlen_q;
  int seqlen_k;
  int max_seqlen_q;
  int max_seqlen_k;
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
  int* __restrict__ tile_count_semaphore;
  int num_sm;

  // Optional FP32 backward state; preceding field offsets are fixed.
  void* __restrict__ o_state_ptr;
  index_t o_state_batch_stride;
  index_t o_state_row_stride;
  index_t o_state_head_stride;

  // Opt-in approximate normalization; composed readers use precise normalization.
  bool use_fast_reciprocal;
  // Internal FP32-only output mode.
  bool output_fp32;
};

}  // namespace ops
}  // namespace xattn

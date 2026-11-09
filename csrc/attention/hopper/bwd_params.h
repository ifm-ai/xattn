#pragma once

#include <cstdint>

namespace at {
class Tensor;
}  // namespace at

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

  // Precision-repair workspace.
  const float* hp_delta = nullptr;
  float* hp_rho = nullptr;  // four planes of FP64 statistics (internal transport)
  float* hp_mass = nullptr; // max, anchored dP, scale, anchor, left, right
  // Deterministic repair: per-row exponent e with sum_j |dS_ij| < 2^e; the
  // statistics planes then hold int64 sums on a fixed 2^(e-62) grid.
  const int* hp_rho_exponent = nullptr;

  // Negative-query readers (softdelta): h counts logical heads
  // l = g * R + r; r = 0 reads negative query g (neg_q, [B, Lq, Hkv, D]) with
  // output gradient neg_dy ([B, Lq, Hkv, V]) and writes its query gradient
  // to dneg_q; r > 0 is query head g * (R - 1) + r - 1 of q / dy / dq. y and
  // lse are the logical forward output [B, Lq, H + Hkv, V] and log-sum-exp
  // [B, H + Hkv, Lq].
  void* neg_q_ptr = nullptr;
  index_t neg_q_batch_stride = 0;
  index_t neg_q_row_stride = 0;
  index_t neg_q_head_stride = 0;
  void* neg_dy_ptr = nullptr;
  index_t neg_dy_batch_stride = 0;
  index_t neg_dy_row_stride = 0;
  index_t neg_dy_head_stride = 0;
  void* dneg_q_ptr = nullptr;
  index_t dneg_q_batch_stride = 0;
  index_t dneg_q_row_stride = 0;
  index_t dneg_q_head_stride = 0;
  // Negative-query readers: add the negative-query gradient into dk (FP32,
  // before the dK cast; neg_q is the aligned keys) instead of writing dneg_q.
  bool fold_neg_q = false;
  // Caller-zeroed grouped FP32 dK | dV accumulators ([B, Lk, Hkv, D] then
  // [B, Lk, Hkv, V], instance head dimensions) for the GQA backward; null:
  // allocated and zeroed by the launch.
  float* dkv_accum_ptr = nullptr;
  // Grouped heads only: receive the FP32 dK / dV accumulators ([B, Lk, Hkv,
  // instance head dimension]) in place of the cast into dk / dv (null: cast).
  at::Tensor* dk_accum_out = nullptr;
  at::Tensor* dv_accum_out = nullptr;
};

}  // namespace ops
}  // namespace xattn

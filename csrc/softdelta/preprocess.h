#pragma once

#include <torch/types.h>

namespace xattn {
namespace ops {

// SoftDelta backward preprocess over logical heads (one pass; it also zeroes dQaccum
// [B, H + Hkv, padded_length * D], D the backward instance's head dimension, and
// the dK / dV accumulators dkv):
// the negative-query output gradient neg_dy [B, L, Hkv, V], the gate gradient
// [B, L, H, C], and dPsum / LSE_log2 [B, H + Hkv, padded_length] of every
// logical head (g R + r, r = 0 the negative query). dy [B, L, H, V] is the gated
// output's gradient, gate [B, L, H, C] the gate logits. Forward state in dy's dtype:
// neg_read_state [B, L, Hkv, V] (negative-query reads) and read_state [B, L, H, V]
// (the gated output), row-major over (batch, row) with any row stride (head slices of
// the class-major readers qualify); neg_read_lse [B, Hkv, L] and read_lse [B, H, L]
// FP32 with dense rows.
// Grouped FP32 dK / dV accumulators to zero ([B, L, Hkv, dk_row], [B, L, Hkv,
// dv_row]); dk null: none.
struct DkvAccum {
  float* dk = nullptr;
  int dk_row = 0;
  float* dv = nullptr;
  int dv_row = 0;
};

void SoftDeltaBwdPreprocess(
    const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& neg_read_state,
    const torch::Tensor& read_state, const torch::Tensor& neg_read_lse, const torch::Tensor& read_lse,
    torch::Tensor& neg_dy, torch::Tensor& gate_grad, torch::Tensor& dpsum, torch::Tensor& lse_log2,
    torch::Tensor& dq_accum, int padded_length, const DkvAccum& dkv = {});

}  // namespace ops
}  // namespace xattn

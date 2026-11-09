#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

// Both SoftDelta reads' backward from the forward's gated output y [B, Lq, H, V] and
// negative reads neg_read [B, Lq, Hkv, V] (q's dtype; head slices of the class-major
// readers qualify) with the class-major LSE [B, Hkv + H, Lq]: one preprocess pass
// (gate gradient, negative-query output gradient, dPsum / LSE_log2 of every logical
// head) and one attention launch over the logical heads. dy [B, Lq, H, C, Dg] is the
// gated output's gradient, gate [B, Lq, H, C, 1] its logits. Returns (dq, dk, dv,
// dneg_q, gate_grad), gate_grad in the gate's shape. fold_neg_q (neg_q is the aligned
// keys): the negative-query gradient is added to dk in FP32 and dneg_q is empty;
// otherwise dneg_q is the neg_q gradient. reader_groups: -1 picks the launch layout
// (per logical head or reader groups, see bwd.cu), 0 / 1 force per logical head /
// reader groups. visibility / span / reset: the forward's support (SoftDeltaFwd).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> SoftDeltaBwd(
    const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& neg_q,
    const torch::Tensor& y, const torch::Tensor& neg_read, const torch::Tensor& lse, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool fold_neg_q = false,
    int64_t reader_groups = -1, int64_t visibility = 0, int64_t span = 0, bool reset = false);

}  // namespace ops
}  // namespace xattn

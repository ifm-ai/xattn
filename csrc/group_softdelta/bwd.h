#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

// Both group SoftDelta readers' backward from the forward state: neg_read_state [B, Lq, Hkv, V]
// (negative-query reads) and read_state [B, Lq, H, V] (query reads, or the gated output
// with gated_output), e.g. views of the GroupSoftDeltaFwd class-major readers (FP32
// state allowed), with their LSEs [B, Hkv, Lq] / [B, H, Lq]: one preprocess pass (gate
// gradient, negative-query output gradient, dPsum / LSE_log2 of every logical head)
// and one attention launch over the logical heads. dy [B, Lq, H, V] is the gated
// output's gradient, gate [B, Lq, H, C] its logits. Returns (dq, dk, dv, dneg_q,
// gate_grad). fold_neg_q (neg_q is the aligned keys): the negative-query gradient
// is added to dk in FP32 and dneg_q is empty; otherwise dneg_q is the neg_q gradient.
// reader_groups: -1 picks the launch layout (per logical head or reader
// groups, see bwd.cu), 0 / 1 force per logical head / reader groups.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GroupSoftDeltaBwdStates(
    const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& neg_q,
    const torch::Tensor& neg_read_state, const torch::Tensor& read_state, const torch::Tensor& neg_read_lse,
    const torch::Tensor& read_lse, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool gated_output, bool fold_neg_q = false,
    int64_t reader_groups = -1);

// GroupSoftDeltaBwdStates from the forward's class-major readers [B, Lq, Hkv + H, V] and LSE
// [B, Hkv + H, Lq], the output gradient [B, Lq, H, C, Dg] and gate logits
// [B, Lq, H, C, 1]; gate_grad comes back in the gate's shape.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GroupSoftDeltaBwd(
    const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& neg_q,
    const torch::Tensor& readers, const torch::Tensor& lse, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool fold_neg_q, int64_t reader_groups = -1);

}  // namespace ops
}  // namespace xattn

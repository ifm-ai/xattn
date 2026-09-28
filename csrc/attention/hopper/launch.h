#pragma once

#include <tuple>

#include <torch/types.h>

namespace xattn {
namespace ops {
namespace attention {
namespace hopper {

bool CausalAttentionSM90Available(const torch::Tensor& q);

std::tuple<torch::Tensor, torch::Tensor> CausalAttentionSM90Fwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool strict_past = false,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt,
    bool causal_flash_attn = false, bool use_fast_reciprocal = false, bool output_fp32 = false);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
CausalAttentionSM90Bwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, bool strict_past = false, bool causal_flash_attn = false);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
CausalAttentionSM90BwdWithHeadBoundary(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, bool strict_past,
    int odd_head_window_right_delta, bool causal_flash_attn = false);

std::tuple<torch::Tensor, torch::Tensor> CausalAttentionSM90VarlenFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale,
    bool strict_past = false,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt, bool output_fp32 = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CausalAttentionSM90VarlenBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale, bool deterministic,
    bool strict_past = false);

void AttentionSM90FwdInto(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool strict_past,
    const c10::optional<torch::Tensor>& output_state,
    torch::Tensor& y,
    torch::Tensor& lse, bool causal_flash_attn);

}  // namespace hopper
}  // namespace attention
}  // namespace ops
}  // namespace xattn

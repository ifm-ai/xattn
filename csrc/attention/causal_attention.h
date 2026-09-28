#pragma once

#include <torch/types.h>

namespace xattn {
namespace ops {
namespace attention {

std::tuple<torch::Tensor, torch::Tensor> CausalAttentionFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend, bool strict_past,
    const c10::optional<torch::Tensor>& output_state,
    bool causal_flash_attn = false, bool use_fast_reciprocal = false, bool output_fp32 = false);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
CausalAttentionBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend, bool strict_past,
    bool causal_flash_attn = false);

std::tuple<torch::Tensor, torch::Tensor> CausalAttentionVarlenFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale,
    const std::string& backend, bool strict_past,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32 = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CausalAttentionVarlenBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale, bool deterministic,
    const std::string& backend, bool strict_past);

}  // namespace attention
}  // namespace ops
}  // namespace xattn

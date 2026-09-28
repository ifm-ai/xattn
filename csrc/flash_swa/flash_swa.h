#pragma once

#include <string>
#include <tuple>

#include <torch/types.h>

namespace xattn {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> FlashSWAFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt, bool output_fp32 = false);

std::tuple<torch::Tensor, torch::Tensor> FlashSWAStrictPastFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt, bool output_fp32 = false);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSWABwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSWAStrictPastBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend);

std::tuple<torch::Tensor, torch::Tensor> FlashSWAVarlenFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale,
    const std::string& backend, bool strict_past,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt, bool output_fp32 = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FlashSWAVarlenBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale, bool deterministic,
    const std::string& backend, bool strict_past);


}  // namespace ops
}  // namespace xattn

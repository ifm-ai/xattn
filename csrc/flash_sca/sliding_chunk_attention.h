#pragma once

#include <c10/util/Optional.h>
#include <torch/types.h>

#include <string>
#include <tuple>

namespace xattn {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> FlashSCAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    bool reset_chunk_pos_per_seq = false,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt, bool output_fp32 = false);

std::tuple<torch::Tensor, torch::Tensor> FlashSCAStrictPastFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    bool reset_chunk_pos_per_seq = false,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt, bool output_fp32 = false);

std::tuple<torch::Tensor, torch::Tensor> FlashSCACUDAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    bool reset_chunk_pos_per_seq = false,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt, bool output_fp32 = false);

bool FlashSCASM90Available(const torch::Tensor& q);

std::tuple<torch::Tensor, torch::Tensor> FlashSCASM90Fwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool reset_chunk_pos_per_seq = false,
    bool strict_past = false,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt, bool output_fp32 = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCASM90Bwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, bool reset_chunk_pos_per_seq = false,
    bool strict_past = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCASM90BwdWithHeadBoundary(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, bool reset_chunk_pos_per_seq,
    bool strict_past, int odd_head_window_right_delta);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCABwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend,
    bool reset_chunk_pos_per_seq = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCAStrictPastBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend,
    bool reset_chunk_pos_per_seq = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCACUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend,
    bool reset_chunk_pos_per_seq = false);


}  // namespace ops
}  // namespace xattn

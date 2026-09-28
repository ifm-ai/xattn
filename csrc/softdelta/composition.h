#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSoftDeltaBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    const torch::Tensor& read,
    const torch::Tensor& read_lse,
    const torch::Tensor& correction,
    const torch::Tensor& correction_lse,
    int64_t span,
    double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    int visibility,
    bool reset_chunk_pos_per_seq,
    bool deterministic);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
FlashSoftDeltaPairedBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    const torch::Tensor& pair_gate_state,
    const torch::Tensor& attention_output_state,
    const torch::Tensor& pair_lse,
    int64_t span,
    double scale,
    int visibility,
    bool deterministic);


}  // namespace ops
}  // namespace xattn

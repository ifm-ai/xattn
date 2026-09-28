#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSoftDeltaMergeGradients(
    const torch::Tensor& read_q_grad,
    const torch::Tensor& correction_q_grad,
    const torch::Tensor& read_k_grad,
    const torch::Tensor& correction_k_grad,
    const torch::Tensor& read_v_grad,
    const torch::Tensor& correction_v_grad,
    const c10::optional<torch::Tensor>& read_prev_k_grad,
    const c10::optional<torch::Tensor>& correction_prev_k_grad,
    const c10::optional<torch::Tensor>& read_prev_v_grad,
    const c10::optional<torch::Tensor>& correction_prev_v_grad);

}  // namespace ops
}  // namespace xattn

#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

torch::Tensor FlashSoftDeltaGateFwd(
    const torch::Tensor& read,
    const torch::Tensor& correction,
    const torch::Tensor& gate);

torch::Tensor FlashSoftDeltaPairGateFwd(
    const torch::Tensor& pair_output,
    const torch::Tensor& gate);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FlashSoftDeltaGateBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& correction,
    const torch::Tensor& gate);

std::tuple<torch::Tensor, torch::Tensor>
FlashSoftDeltaPairGateBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_gate_state,
    const torch::Tensor& gate);


}  // namespace ops
}  // namespace xattn

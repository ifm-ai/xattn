#pragma once
#include <torch/types.h>
namespace xattn::ops::attention::precision::ordered {
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> MainBackward(
    const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,
    const torch::Tensor& g,const torch::Tensor& delta,const torch::Tensor& log_z,
    const torch::Tensor& stats,const torch::Tensor& metadata,
    const c10::optional<torch::Tensor>& q_segment,
    const c10::optional<torch::Tensor>& k_segment,
    const c10::optional<torch::Tensor>& q_positions,
    double scale,int visibility,int span,bool strict_past,bool reset,
    const torch::Tensor& rho_exponent);
}

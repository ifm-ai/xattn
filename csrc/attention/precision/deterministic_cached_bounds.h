#pragma once
#include <torch/types.h>
namespace xattn::ops::attention::precision::cached_bounds {
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> DeterministicBackward(
 const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
 const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
 const torch::Tensor&,const torch::Tensor&,const c10::optional<torch::Tensor>&,
 const c10::optional<torch::Tensor>&,double,int);
torch::Tensor DeterministicRepair(const torch::Tensor&,const torch::Tensor&,
 const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
 torch::Tensor,torch::Tensor,torch::Tensor,double,int);
}

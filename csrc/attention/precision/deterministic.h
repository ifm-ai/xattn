#pragma once
#include <torch/types.h>
namespace xattn::ops::attention::precision {
// Anchor shares of the deterministic level-3/4 repairs, ordered per KV head.
torch::Tensor DeterministicRepair(const torch::Tensor&,const torch::Tensor&,
 const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
 torch::Tensor,torch::Tensor,torch::Tensor,double,int);
}

#pragma once
#include <torch/types.h>
namespace xattn::ops::attention::precision {
// Anchor shares of the deterministic repairs, ordered per KV head.
torch::Tensor DeterministicRepair(const torch::Tensor&,const torch::Tensor&,
 const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
 torch::Tensor,torch::Tensor,torch::Tensor,double);
}

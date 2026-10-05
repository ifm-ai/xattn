#pragma once
#include <torch/types.h>
namespace xattn::ops::attention::precision {
// Apply a capacity-checked probability anchor correction to FP32 dV.
// Returns per-KV-head fallback flags [B,Hkv]. A flagged head must be replaced
// in its entirety by the normalized nonnegative fallback before output cast.
torch::Tensor ProbabilityRepair(const torch::Tensor& g,const torch::Tensor& stats,
                               const torch::Tensor& metadata,torch::Tensor dv);
}

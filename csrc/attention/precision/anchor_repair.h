#pragma once
#include <torch/types.h>

namespace xattn::ops::attention::precision {
// In-place FP32 accumulator correction. Rho is the COMPLETE sum of actual
// dS hi(+lo) operands, including pre-cast error, not just quantization error.
// Q/K: [B,Nq,Hq,D]/[B,Nk,Hkv,D]; rho/anchor: [B,Hq,Nq].
// anchor=-1 denotes an empty visible support and must have rho=0.
// The producer must select a visible anchor. This primitive has no mask policy.
// It neither edits P nor casts any gradient back to the input dtype.
void AnchorRepair(const torch::Tensor& q, const torch::Tensor& k,
                  const torch::Tensor& rho, const torch::Tensor& anchor,
                  torch::Tensor dq, torch::Tensor dk, double scale);
}  // namespace xattn::ops::attention::precision

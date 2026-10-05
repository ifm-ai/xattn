#pragma once
#include <torch/types.h>
namespace xattn::ops::attention::precision {
// Lossless per-(batch,head,feature) constant shift. Each feature falls back to
// zero shift unless every input is exactly reconstructible in real arithmetic.
std::tuple<torch::Tensor, torch::Tensor> Center(const torch::Tensor& input);
// Divide each dO row by a nonnegative even power of two, only when all input
// components survive round-trip in the original dtype. Scale shape [B,H,N].
std::tuple<torch::Tensor, torch::Tensor> ScaleDO(const torch::Tensor& input);
}  // namespace xattn::ops::attention::precision

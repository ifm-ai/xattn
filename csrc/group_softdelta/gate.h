#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

// output[b, t, h, c, :] = read[b, t, h, c, :] - sigmoid(gate[b, t, h, c]) * neg_read[b, t, h / R, c, :]
// read [B, L, H, C, Dg], neg_read [B, L, Hkv, C, Dg] (input dtype or FP32 state), gate [B, L, H, C, 1].
torch::Tensor GroupSoftDeltaGateFwd(
    const torch::Tensor& read,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate);

// The same gate over the fused forward's readers [B, L, Hkv + H, C, Dg] (head g the
// negative read of KV head g, head Hkv + h the read of query head h).
torch::Tensor GroupSoftDeltaGateFwdReaders(const torch::Tensor& readers, const torch::Tensor& gate);

// (neg_read_grad [B, L, Hkv, C, Dg], gate_grad [B, L, H, C, 1]) in the gate dtype; the read gradient is
// dy itself. neg_read_grad sums the R query heads of each KV head in a fixed order.
std::tuple<torch::Tensor, torch::Tensor> GroupSoftDeltaGateBwd(
    const torch::Tensor& dy,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate);

// The gate backward over the fused forward's readers (negative reads in heads [0, Hkv)).
std::tuple<torch::Tensor, torch::Tensor> GroupSoftDeltaGateBwdReaders(
    const torch::Tensor& dy, const torch::Tensor& readers, const torch::Tensor& gate);

// Sums the two readers' K/V gradients in FP32 and rounds once:
//   key_grad = read_key_grad + neg_read_key_grad (+ dneg_q on the last Lq rows)
//   value_grad = read_value_grad + neg_read_value_grad
// All [B, Lk, Hkv, *] contiguous; dneg_q [B, Lq, Hkv, D] or None. Writes into the read gradients.
std::tuple<torch::Tensor, torch::Tensor> GroupSoftDeltaMergeGradients(
    torch::Tensor read_key_grad,
    const torch::Tensor& neg_read_key_grad,
    const c10::optional<torch::Tensor>& dneg_q,
    torch::Tensor read_value_grad,
    const torch::Tensor& neg_read_value_grad);

}  // namespace ops
}  // namespace xattn

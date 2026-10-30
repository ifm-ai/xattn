#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

// y[b, t, h, c, :] = read[b, t, h, c, :] - sigmoid(gate[b, t, h, c]) * neg_read[b, t, h / R, c, :]
// read [B, L, H, C, Dg] and neg_read [B, L, Hkv, C, Dg] (the gate dtype, or both FP32),
// gate [B, L, H, C, 1] contiguous. read, neg_read and output may be head slices of the
// readers [B, L, Hkv + H, C, Dg] (packed heads, unit column stride); output, in the gate
// dtype, may alias read (the in-place gate). Without output a contiguous y is returned.
torch::Tensor SoftDeltaGateFwd(
    const torch::Tensor& read,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate,
    const c10::optional<torch::Tensor>& output);

// The read recovered from the gated output: y + sigmoid(gate) * neg_read, contiguous
// [B, L, H, C, Dg] in the gate dtype (operands as in SoftDeltaGateFwd).
torch::Tensor SoftDeltaGateRestore(
    const torch::Tensor& y,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate);

// (neg_read_grad [B, L, Hkv, C, Dg], gate_grad [B, L, H, C, 1]) in the gate dtype; the read gradient is
// dy itself. neg_read_grad sums the R query heads of each KV head in a fixed order. dy is contiguous;
// neg_read (gate dtype or FP32) may be a head slice of the readers.
std::tuple<torch::Tensor, torch::Tensor> SoftDeltaGateBwd(
    const torch::Tensor& dy,
    const torch::Tensor& neg_read,
    const torch::Tensor& gate);

// Sums the two readers' K/V gradients in FP32 and rounds once:
//   key_grad = read_key_grad + neg_read_key_grad (+ dneg_q on the last Lq rows)
//   value_grad = read_value_grad + neg_read_value_grad
// All [B, Lk, Hkv, *]; dneg_q [B, Lq, Hkv, D] or None. 16-bit read gradients are updated in place;
// FP32 read accumulators give new gradients in the negative read's dtype.
std::tuple<torch::Tensor, torch::Tensor> SoftDeltaMergeGradients(
    torch::Tensor read_key_grad,
    const torch::Tensor& neg_read_key_grad,
    const c10::optional<torch::Tensor>& dneg_q,
    torch::Tensor read_value_grad,
    const torch::Tensor& neg_read_value_grad);

}  // namespace ops
}  // namespace xattn

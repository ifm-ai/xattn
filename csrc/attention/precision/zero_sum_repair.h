#pragma once
#include <torch/types.h>

namespace xattn::ops::attention::precision {
// Zero-sum repair of the actual dS operand, in place on the FP32 accumulators.
// Row i's residual rho_i = sum_j dS_ij(actual operand) is removed with the same
// virtual dS change on dQ and dK: the share s_a * rho_i on the row's anchor key,
// the rest uniformly over its last <= window admitted keys, with
// s_a = p_a = exp(-log_z), the anchor probability. A row whose window is not a
// single admitted run (k_segment boundary inside), or whose bounds would break
// monotonicity, sends all of rho to the anchor. rho_anchor receives each row's
// anchor share. With anchor_scatter=false only the uniform shares are applied
// and the anchor share (dQ and dK) is left to an ordered caller (deterministic
// path); otherwise dK anchor updates use per-warp aggregated atomics.
// Q/K: [B,Nq,Hq,D]/[B,Nk,Hkv,D]; rho/log_z: [B,Hq,Nq]; metadata [6,B,Hq,Nq]
// (anchor, first and last admitted key in planes 3/4/5, head independent).
void ZeroSumRepair(const torch::Tensor& q, const torch::Tensor& k,
                   const torch::Tensor& rho, const torch::Tensor& metadata,
                   const torch::Tensor& log_z,
                   const c10::optional<torch::Tensor>& k_segment,
                   int64_t window, bool anchor_scatter,
                   torch::Tensor rho_anchor, torch::Tensor dq, torch::Tensor dk,
                   double scale);
}  // namespace xattn::ops::attention::precision

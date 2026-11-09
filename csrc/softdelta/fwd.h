#pragma once

#include <torch/types.h>

#include <tuple>

namespace xattn {
namespace ops {

// Both SoftDelta reads in one launch. Returns the class-major readers
// [B, Lq, Hkv + H, V] and their log-sum-exp [B, Hkv + H, Lq] (FP32): head g is the
// negative read (strict past) of neg_q head g, head Hkv + h the query read
// (inclusive) of query head h. output_state optionally receives the FP32 readers.
// visibility / span (softdelta/visibility.h) limit both reads to a sliding
// window or sliding chunks; reset restarts chunk positions in each segment.
std::tuple<torch::Tensor, torch::Tensor> SoftDeltaFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& neg_q, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const c10::optional<torch::Tensor>& output_state, int64_t visibility = 0, int64_t span = 0,
    bool reset = false);

// The same readers and log-sum-exp for packed sequences (equal Q / K lengths,
// sequences given by cu_seqlens over the B * L rows): the inclusive read runs
// as varlen causal attention (PackGQA) and the strict read as varlen strict
// attention with Hkv heads, each written into its readers heads.
std::tuple<torch::Tensor, torch::Tensor> SoftDeltaFwdVarlen(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& neg_q, const torch::Tensor& cu_seqlens,
    int64_t max_seqlen, double scale);

}  // namespace ops
}  // namespace xattn

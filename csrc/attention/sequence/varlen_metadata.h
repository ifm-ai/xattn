#pragma once

#include <limits>

#include <torch/types.h>

namespace xattn {
namespace ops {
namespace attention {
namespace sequence {

struct VarlenMetadata {
  int num_sequences;
  int k_sequence_offset;
};

inline VarlenMetadata CheckVarlenMetadata(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    const char* owner) {
  TORCH_CHECK(
      cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1,
      owner, " varlen cu_seqlens_q/cu_seqlens_k must be 1D");
  TORCH_CHECK(
      cu_seqlens_q.scalar_type() == at::kInt &&
          cu_seqlens_k.scalar_type() == at::kInt,
      owner, " varlen cu_seqlens_q/cu_seqlens_k must be int32");
  TORCH_CHECK(
      cu_seqlens_q.device().type() == torch::kCUDA &&
          cu_seqlens_k.device().type() == torch::kCUDA,
      owner, " varlen cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
  TORCH_CHECK(
      cu_seqlens_q.device() == q.device() &&
          cu_seqlens_k.device() == q.device(),
      owner, " varlen cu_seqlens must share the Q/K/V device");
  TORCH_CHECK(
      cu_seqlens_q.numel() >= 2,
      owner, " varlen cu_seqlens must contain at least one sequence");
  TORCH_CHECK(
      cu_seqlens_k.numel() >= cu_seqlens_q.numel(),
      owner, " right-aligned varlen requires at least as many K "
      "sequences as Q sequences; got ",
      cu_seqlens_k.numel() - 1, " K sequences and ",
      cu_seqlens_q.numel() - 1, " Q sequences");
  TORCH_CHECK(
      max_seqlen_q > 0 && max_seqlen_k > 0,
      owner, " varlen max_seqlen_q/max_seqlen_k must be positive");
  TORCH_CHECK(
      q.size(0) <= std::numeric_limits<int>::max() &&
          k.size(0) <= std::numeric_limits<int>::max(),
      owner, " varlen total_q/total_k must fit int32");
  TORCH_CHECK(
      max_seqlen_q <= std::numeric_limits<int>::max() &&
          max_seqlen_k <= std::numeric_limits<int>::max(),
      owner, " varlen max_seqlen_q/max_seqlen_k must fit int32");
  const int64_t num_sequences = cu_seqlens_q.numel() - 1;
  const int64_t k_sequence_offset =
      cu_seqlens_k.numel() - cu_seqlens_q.numel();
  TORCH_CHECK(
      num_sequences <= std::numeric_limits<int>::max() &&
          k_sequence_offset <= std::numeric_limits<int>::max(),
      owner, " varlen sequence counts must fit int32");
  return {
      static_cast<int>(num_sequences),
      static_cast<int>(k_sequence_offset),
  };
}

}  // namespace sequence
}  // namespace attention
}  // namespace ops
}  // namespace xattn

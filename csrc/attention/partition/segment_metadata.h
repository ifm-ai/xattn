#pragma once

#include <cstdint>

#include <torch/types.h>

namespace xattn {
namespace ops {
namespace attention {
namespace partition {

struct SegmentMetadata {
  torch::Tensor q_segment_idx;
  torch::Tensor k_segment_idx;

  bool enabled() const {
    return q_segment_idx.defined();
  }

  int64_t batch_count() const {
    return enabled() ? q_segment_idx.size(0) : 0;
  }

  int64_t q_length() const {
    return enabled() ? q_segment_idx.size(1) : 0;
  }

  int64_t k_length() const {
    return enabled() ? k_segment_idx.size(1) : 0;
  }
};

inline void CheckSegmentMetadata(
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    int64_t const batch_count, int64_t const q_length,
    int64_t const k_length, const torch::Device& device,
    const char* const owner) {
  TORCH_CHECK(
      q_segment_idx.has_value() == k_segment_idx.has_value(),
      owner,
      " q_segment_idx and k_segment_idx must both be provided or both "
      "be None");
  if (!q_segment_idx.has_value()) {
    return;
  }

  const torch::Tensor& q_idx = q_segment_idx.value();
  const torch::Tensor& k_idx = k_segment_idx.value();
  TORCH_CHECK(q_idx.dim() == 2 && k_idx.dim() == 2,
              owner, " segment metadata must be 2D");
  TORCH_CHECK(q_idx.scalar_type() == at::kLong,
              owner, " q_segment_idx must be int64");
  TORCH_CHECK(k_idx.scalar_type() == at::kLong,
              owner, " k_segment_idx must be int64");
  TORCH_CHECK(q_idx.device() == device && k_idx.device() == device,
              owner, " segment metadata must share the Q/K/V device");
  TORCH_CHECK(
      q_idx.size(0) == batch_count && q_idx.size(1) == q_length,
      owner, " q_segment_idx must have shape [", batch_count, ", ",
      q_length, "]");
  TORCH_CHECK(
      k_idx.size(0) == batch_count && k_idx.size(1) == k_length,
      owner, " k_segment_idx must have shape [", batch_count, ", ",
      k_length, "]");
}

inline SegmentMetadata NormalizeSegmentMetadata(
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    int64_t const batch_count, int64_t const q_length,
    int64_t const k_length, const torch::Device& device,
    const char* const owner) {
  CheckSegmentMetadata(
      q_segment_idx, k_segment_idx, batch_count, q_length, k_length,
      device, owner);
  if (!q_segment_idx.has_value()) {
    return {};
  }
  return {
      q_segment_idx.value().contiguous(),
      k_segment_idx.value().contiguous(),
  };
}

}  // namespace partition
}  // namespace attention
}  // namespace ops
}  // namespace xattn

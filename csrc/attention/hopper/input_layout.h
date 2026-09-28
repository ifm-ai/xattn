#pragma once

#include <cstdint>

#include <torch/types.h>

namespace xattn::ops::attention::hopper {

// TMA query slices require aligned, non-overlapping, unit-feature-stride storage.
inline bool CanUseStridedTmaQuery(const torch::Tensor& q) {
  return q.stride(3) == 1 &&
      q.stride(2) >= q.size(3) &&
      q.stride(1) >= q.size(2) * q.stride(2) &&
      q.stride(0) >= q.size(1) * q.stride(1) &&
      reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
      (q.stride(0) * q.element_size()) % 16 == 0 &&
      (q.stride(1) * q.element_size()) % 16 == 0 &&
      (q.stride(2) * q.element_size()) % 16 == 0;
}

}  // namespace xattn::ops::attention::hopper

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>

#include "attention/partition/segment_metadata.h"
#include "flash_sca/sliding_chunk_attention.h"

namespace xattn {
namespace ops {
namespace flash_sca {

inline void CheckInputs(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx) {
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q, k, v must have shape [B, L, H, D]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == v.scalar_type(),
              "q, k, v must have the same dtype");
  TORCH_CHECK(q.scalar_type() == at::kHalf ||
                  q.scalar_type() == at::kBFloat16,
              "FlashSCA supports only fp16 and bf16 q, k, v input");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "q, k, v batch sizes must match");
  TORCH_CHECK(q.size(1) > 0 && q.size(1) <= k.size(1) && k.size(1) == v.size(1),
              "Q length must be positive and <= matching K/V lengths");
  TORCH_CHECK(k.size(2) == v.size(2),
              "k and v head counts must match");
  TORCH_CHECK(q.size(2) > 0 && k.size(2) > 0,
              "q and kv head counts must be positive");
  TORCH_CHECK(q.size(2) % k.size(2) == 0,
              "q head count must be divisible by kv head count; got Hq=",
              q.size(2), ", Hkv=", k.size(2));
  TORCH_CHECK(q.size(3) == k.size(3),
              "q and k qk head dims must match");
  TORCH_CHECK(q.size(3) <= 2048, "qk head dim > 2048 is not supported yet");
  TORCH_CHECK(v.size(3) <= 2048, "v head dim > 2048 is not supported yet");
  TORCH_CHECK(prev_k.has_value() == prev_v.has_value(),
              "prev_k and prev_v must both be provided or both be None");
  if (prev_k.has_value()) {
    TORCH_CHECK(prev_k.value().scalar_type() == q.scalar_type() &&
                    prev_v.value().scalar_type() == q.scalar_type(),
                "prev_k and prev_v must have the same dtype as q, k, v");
    TORCH_CHECK(prev_k.value().size(0) == q.size(0) &&
                    prev_k.value().size(1) == chunk_size &&
                    prev_k.value().size(2) == k.size(2) &&
                    prev_k.value().size(3) == q.size(3),
                "prev_k must have shape [B, chunk_size, Hkv, qk_dim]");
    TORCH_CHECK(prev_v.value().size(0) == q.size(0) &&
                    prev_v.value().size(1) == chunk_size &&
                    prev_v.value().size(2) == v.size(2) &&
                    prev_v.value().size(3) == v.size(3),
                "prev_v must have shape [B, chunk_size, Hkv, v_dim]");
  }
  const int64_t expected_k_segment_len =
      k.size(1) + (prev_k.has_value() ? chunk_size : int64_t(0));
  attention::partition::CheckSegmentMetadata(
      q_segment_idx, k_segment_idx, q.size(0), q.size(1),
      expected_k_segment_len, q.device(), "FlashSCA");
}

inline torch::Tensor MaybeCastCompute(const torch::Tensor& x,
                                      c10::ScalarType compute_dtype,
                                      bool cast_compute) {
  return (cast_compute ? x.to(compute_dtype) : x).contiguous();
}

enum class Backend {
  kAuto,
  kCUDA,
  kSM90,
};

inline Backend ParseBackend(std::string backend) {
  std::transform(
      backend.begin(), backend.end(), backend.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (backend.empty() || backend == "auto") {
    return Backend::kAuto;
  }
  if (backend == "cuda") {
    return Backend::kCUDA;
  }
  if (backend == "sm90") {
    return Backend::kSM90;
  }
  TORCH_CHECK(false,
              "FlashSCA backend must be one of auto, cuda, or sm90; got ",
              backend);
  return Backend::kAuto;
}

inline bool ShouldUseSM90(const torch::Tensor& q, const std::string& backend) {
  const Backend parsed_backend = ParseBackend(backend);
  if (parsed_backend == Backend::kCUDA) {
    return false;
  }
#ifndef XATTN_HAS_SM90
  TORCH_CHECK(parsed_backend != Backend::kSM90,
              "FlashSCA backend='",
              backend,
              "' requested SM90, but this xattn build was compiled "
              "without the FlashSCA SM90 backend. Set XATTN_TARGET_SM=90 "
              "or XATTN_TARGET_SM=all when building xattn.");
  return false;
#else
  const bool sm90_available = FlashSCASM90Available(q);
  TORCH_CHECK(parsed_backend != Backend::kSM90 || sm90_available,
              "FlashSCA backend='",
              backend,
              "' requested SM90, but SM90 is unavailable or disabled");
  return sm90_available;
#endif
}

}  // namespace flash_sca
}  // namespace ops
}  // namespace xattn

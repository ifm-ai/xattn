// Author: Shicheng Wen

#include "attention/causal_attention.h"

#include <algorithm>
#include <cctype>
#include <limits>

#include "attention/partition/segment_metadata.h"
#ifdef XATTN_HAS_SM90
#include "attention/hopper/launch.h"
#endif

namespace xattn {
namespace ops {
namespace attention {
namespace {

void CheckOptionalCUDA(
    const c10::optional<torch::Tensor>& tensor, const char* name) {
  if (tensor.has_value()) {
    TORCH_CHECK(tensor.value().device().type() == torch::kCUDA,
                name, " must be a CUDA tensor");
  }
}

void CheckCausalAttentionInputs(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool allow_short_q) {
  TORCH_CHECK(
      window_size >= 0 &&
          window_size <= std::numeric_limits<int>::max(),
      "Causal attention window_size must be in [0, INT_MAX]");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "Causal attention q, k, v must have shape [B, L, H, D]");
  TORCH_CHECK(
      q.device().type() == torch::kCUDA &&
          k.device() == q.device() && v.device() == q.device(),
      "Causal attention q, k, v must be same-device CUDA tensors");
  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() &&
          q.scalar_type() == v.scalar_type(),
      "Causal attention q, k, v must have the same dtype");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf ||
          q.scalar_type() == at::kBFloat16,
      "Causal attention supports only fp16 and bf16 q, k, v input");
  TORCH_CHECK(
      q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
          k.size(1) == v.size(1) &&
          (allow_short_q ? q.size(1) <= k.size(1) : q.size(1) == k.size(1)),
      "Causal attention requires matching batches and K/V lengths; Q length "
      "must match K/V (or be shorter for full attention)");
  TORCH_CHECK(q.size(1) > 0,
              "Causal attention current sequence length must be positive");
  TORCH_CHECK(k.size(2) == v.size(2),
              "Causal attention k and v head counts must match");
  TORCH_CHECK(q.size(2) > 0 && k.size(2) > 0 &&
                  q.size(2) % k.size(2) == 0,
              "Causal attention q head count must be a positive multiple of "
              "the KV head count");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) > 0,
              "Causal attention q and k head dims must match and be positive");
  TORCH_CHECK(v.size(3) > 0,
              "Causal attention value head dim must be positive");
  TORCH_CHECK(prev_k.has_value() == prev_v.has_value(),
              "Causal attention prev_k and prev_v must both be provided or "
              "both be None");
  int64_t prev_length = 0;
  if (prev_k.has_value()) {
    const torch::Tensor& pk = prev_k.value();
    const torch::Tensor& pv = prev_v.value();
    TORCH_CHECK(
        pk.dim() == 4 && pv.dim() == 4 &&
            pk.device() == q.device() && pv.device() == q.device(),
        "Causal attention prev_k/prev_v must be 4D tensors on the Q device");
    TORCH_CHECK(
        pk.scalar_type() == q.scalar_type() &&
            pv.scalar_type() == q.scalar_type(),
        "Causal attention previous and current K/V must share a dtype");
    TORCH_CHECK(
        pk.size(0) == q.size(0) && pv.size(0) == q.size(0) &&
            pk.size(1) == pv.size(1) && pk.size(1) > 0 &&
            pk.size(2) == k.size(2) && pv.size(2) == v.size(2) &&
            pk.size(3) == k.size(3) && pv.size(3) == v.size(3),
        "Causal attention previous K/V shapes are incompatible with Q/K/V");
    prev_length = pk.size(1);
  }
  attention::partition::CheckSegmentMetadata(
      q_segment_idx, k_segment_idx, q.size(0), q.size(1),
      k.size(1) + prev_length, q.device(), "Causal attention");
}

std::string NormalizeBackend(std::string backend) {
  std::transform(
      backend.begin(), backend.end(), backend.begin(),
      [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
      });
  return backend;
}

void CheckBackend(
    const torch::Tensor& q, const std::string& backend) {
  const std::string normalized = NormalizeBackend(backend);
  TORCH_CHECK(
      normalized.empty() || normalized == "auto" ||
          normalized == "sm90",
      "Causal attention currently supports backend='auto' or backend='sm90'");
#ifdef XATTN_HAS_SM90
  TORCH_CHECK(
      hopper::CausalAttentionSM90Available(q),
      "Causal attention currently requires an SM90 GPU");
#else
  TORCH_CHECK(
      false,
      "This xattn build does not include the SM90 lowering required by "
      "Causal attention");
#endif
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> CausalAttentionFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend, bool strict_past,
    const c10::optional<torch::Tensor>& output_state,
    bool causal_flash_attn, bool use_fast_reciprocal, bool output_fp32) {
  CheckCausalAttentionInputs(
      q, k, v, window_size, prev_k, prev_v, q_segment_idx,
      k_segment_idx, true);
  CheckOptionalCUDA(prev_k, "prev_k");
  CheckOptionalCUDA(prev_v, "prev_v");
  CheckOptionalCUDA(q_segment_idx, "q_segment_idx");
  CheckOptionalCUDA(k_segment_idx, "k_segment_idx");
  CheckOptionalCUDA(output_state, "output_state");
  CheckBackend(q, backend);
  if (causal_flash_attn) {
    window_size = k.size(1) + (prev_k.has_value() ? prev_k.value().size(1) : 0) - 1;
  }
#ifdef XATTN_HAS_SM90
  return hopper::CausalAttentionSM90Fwd(
      q, k, v, window_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, strict_past, output_state, causal_flash_attn,
      use_fast_reciprocal, output_fp32);
#else
  TORCH_CHECK(false, "Causal attention SM90 support is not built");
#endif
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
CausalAttentionBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend, bool strict_past,
    bool causal_flash_attn) {
  CheckCausalAttentionInputs(
      q, k, v, window_size, prev_k, prev_v, q_segment_idx,
      k_segment_idx, true);
  TORCH_CHECK(
      y_grad.device() == q.device() && y.device() == q.device() &&
          lse.device() == q.device(),
      "Causal attention backward tensors must share the Q device");
  CheckBackend(q, backend);
  if (causal_flash_attn) {
    window_size = k.size(1) + (prev_k.has_value() ? prev_k.value().size(1) : 0) - 1;
  }
#ifdef XATTN_HAS_SM90
  return hopper::CausalAttentionSM90Bwd(
      y_grad, q, k, v, y, lse, window_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic, strict_past, causal_flash_attn);
#else
  TORCH_CHECK(false, "Causal attention SM90 support is not built");
#endif
}

std::tuple<torch::Tensor, torch::Tensor> CausalAttentionVarlenFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale,
    const std::string& backend, bool strict_past,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  CheckOptionalCUDA(output_state, "output_state");
  CheckBackend(q, backend);
#ifdef XATTN_HAS_SM90
  return hopper::CausalAttentionSM90VarlenFwd(
      q, k, v, cu_seqlens_q, cu_seqlens_k,
      max_seqlen_q, max_seqlen_k, window_size, scale, strict_past,
      output_state, output_fp32);
#else
  TORCH_CHECK(false, "Causal attention SM90 support is not built");
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CausalAttentionVarlenBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale, bool deterministic,
    const std::string& backend, bool strict_past) {
  CheckBackend(q, backend);
#ifdef XATTN_HAS_SM90
  return hopper::CausalAttentionSM90VarlenBwd(
      y_grad, q, k, v, y, lse, cu_seqlens_q, cu_seqlens_k,
      max_seqlen_q, max_seqlen_k, window_size, scale, deterministic,
      strict_past);
#else
  TORCH_CHECK(false, "Causal attention SM90 support is not built");
#endif
}

}  // namespace attention
}  // namespace ops
}  // namespace xattn

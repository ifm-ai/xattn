#include <ATen/DeviceGuard.h>
#include <c10/core/MemoryFormat.h>
#include <c10/cuda/CUDAGuard.h>

#include <tuple>

#include <utility>
#include <vector>

#include "flash_sca/cuda/instantiate.h"
#include "flash_sca/runtime.h"
#include "flash_sca/sliding_chunk_attention.h"

namespace xattn {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> FlashSCACUDAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    bool reset_chunk_pos_per_seq,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  flash_sca::CheckInputs(
      q, k, v, chunk_size, prev_k, prev_v, q_segment_idx, k_segment_idx);
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
#ifdef XATTN_HAS_SM90
  if (flash_sca::ShouldUseSM90(q, backend)) {
    return FlashSCASM90Fwd(
        q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx,
        k_segment_idx, reset_chunk_pos_per_seq, false, output_state, output_fp32);
  }
#else
  flash_sca::ShouldUseSM90(q, backend);
#endif
  TORCH_CHECK(q.size(1) == k.size(1),
              "FlashSCA unequal current Q/KV lengths require the SM90 backend");
  TORCH_CHECK(
      !output_state.has_value() && !output_fp32,
      "FlashSCA FP32 output/state requires the SM90 backend");
  TORCH_CHECK(!reset_chunk_pos_per_seq,
              "reset_chunk_pos_per_seq=true is only supported by the SM90 "
              "FlashSCA backend");
  const bool cast_compute = q.scalar_type() == at::kFloat;
  const c10::ScalarType compute_dtype =
      q.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kHalf;
  torch::Tensor q_c = flash_sca::MaybeCastCompute(
      q, compute_dtype, cast_compute);
  torch::Tensor k_c = flash_sca::MaybeCastCompute(
      k, compute_dtype, cast_compute);
  torch::Tensor v_c = flash_sca::MaybeCastCompute(
      v, compute_dtype, cast_compute);

  const bool has_prev = prev_k.has_value();
  const bool has_segment = q_segment_idx.has_value();
  torch::Tensor prev_k_c;
  torch::Tensor prev_v_c;
  torch::Tensor q_segment_idx_c;
  torch::Tensor k_segment_idx_c;
  if (has_prev) {
    prev_k_c = flash_sca::MaybeCastCompute(
        prev_k.value(), compute_dtype, cast_compute);
    prev_v_c = flash_sca::MaybeCastCompute(
        prev_v.value(), compute_dtype, cast_compute);
  }
  if (has_segment) {
    q_segment_idx_c = q_segment_idx.value().contiguous();
    k_segment_idx_c = k_segment_idx.value().contiguous();
  }

  const int64_t B = q.size(0);
  const int64_t L = q.size(1);
  const int64_t H = q.size(2);
  const int64_t V = v.size(3);
  torch::Tensor y_compute = torch::empty(
      {B, L, H, V},
      v_c.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor lse = torch::empty(
      {B, H, L},
      q.options().dtype(at::kFloat).memory_format(at::MemoryFormat::Contiguous));

  if (q_c.scalar_type() == at::kHalf) {
    FlashSCAFwdFP16(
        q_c, k_c, v_c, chunk_size, static_cast<float>(scale), prev_k_c,
        prev_v_c, q_segment_idx_c, k_segment_idx_c, has_prev, has_segment,
        y_compute, lse);
  } else {
    FlashSCAFwdBF16(
        q_c, k_c, v_c, chunk_size, static_cast<float>(scale), prev_k_c,
        prev_v_c, q_segment_idx_c, k_segment_idx_c, has_prev, has_segment,
        y_compute, lse);
  }
  torch::Tensor y =
      cast_compute ? y_compute.to(q.scalar_type()) : y_compute;
  return std::make_tuple<torch::Tensor, torch::Tensor>(std::move(y),
                                                       std::move(lse));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCACUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend,
    bool reset_chunk_pos_per_seq) {
  flash_sca::CheckInputs(
      q, k, v, chunk_size, prev_k, prev_v, q_segment_idx, k_segment_idx);
  const std::vector<int64_t> expected_y_shape{
      q.size(0), q.size(1), q.size(2), v.size(3)};
  TORCH_CHECK(y.sizes().vec() == expected_y_shape,
              "y must have shape [B, L, Hq, v_dim]");
  TORCH_CHECK(y_grad.sizes().vec() == expected_y_shape,
              "y_grad must have shape [B, L, Hq, v_dim]");
  TORCH_CHECK(y.scalar_type() == v.scalar_type() &&
                  y_grad.scalar_type() == v.scalar_type(),
              "y and y_grad must have the same dtype as v");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse must be float32");
  TORCH_CHECK(lse.size(0) == q.size(0) && lse.size(1) == q.size(2) &&
                  lse.size(2) == q.size(1),
              "lse must have shape [B, H, L]");
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
#ifdef XATTN_HAS_SM90
  if (flash_sca::ShouldUseSM90(q, backend)) {
    return FlashSCASM90Bwd(
        y_grad, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,
        q_segment_idx, k_segment_idx, deterministic,
        reset_chunk_pos_per_seq);
  }
#else
  flash_sca::ShouldUseSM90(q, backend);
#endif
  TORCH_CHECK(q.size(1) == k.size(1),
              "FlashSCA unequal current Q/KV lengths require the SM90 backend");
  TORCH_CHECK(!reset_chunk_pos_per_seq,
              "reset_chunk_pos_per_seq=true is only supported by the SM90 "
              "FlashSCA backend");
  const bool cast_compute = q.scalar_type() == at::kFloat;
  const c10::ScalarType compute_dtype =
      q.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kHalf;
  torch::Tensor y_grad_c =
      flash_sca::MaybeCastCompute(y_grad, compute_dtype, cast_compute);
  torch::Tensor y_c = flash_sca::MaybeCastCompute(
      y, compute_dtype, cast_compute);
  torch::Tensor q_c = flash_sca::MaybeCastCompute(
      q, compute_dtype, cast_compute);
  torch::Tensor k_c = flash_sca::MaybeCastCompute(
      k, compute_dtype, cast_compute);
  torch::Tensor v_c = flash_sca::MaybeCastCompute(
      v, compute_dtype, cast_compute);
  const bool has_prev = prev_k.has_value();
  const bool has_segment = q_segment_idx.has_value();
  torch::Tensor prev_k_c;
  torch::Tensor prev_v_c;
  torch::Tensor q_segment_idx_c;
  torch::Tensor k_segment_idx_c;
  if (has_prev) {
    prev_k_c = flash_sca::MaybeCastCompute(
        prev_k.value(), compute_dtype, cast_compute);
    prev_v_c = flash_sca::MaybeCastCompute(
        prev_v.value(), compute_dtype, cast_compute);
  }
  if (has_segment) {
    q_segment_idx_c = q_segment_idx.value().contiguous();
    k_segment_idx_c = k_segment_idx.value().contiguous();
  }

  torch::Tensor q_grad_c;
  if (deterministic) {
    q_grad_c = torch::empty_like(q_c);
  }
  torch::Tensor k_grad_c = torch::empty_like(k_c);
  torch::Tensor v_grad_c = torch::empty_like(v_c);
  c10::optional<torch::Tensor> prev_k_grad_c = c10::nullopt;
  c10::optional<torch::Tensor> prev_v_grad_c = c10::nullopt;
  if (has_prev) {
    prev_k_grad_c = c10::make_optional(torch::empty_like(prev_k_c));
    prev_v_grad_c = c10::make_optional(torch::empty_like(prev_v_c));
  }

  if (q_c.scalar_type() == at::kHalf) {
    FlashSCABwdFP16(
        y_grad_c, q_c, k_c, v_c, y_c, lse, chunk_size,
        static_cast<float>(scale), prev_k_c, prev_v_c, q_segment_idx_c,
        k_segment_idx_c, has_prev, has_segment, q_grad_c, k_grad_c,
        v_grad_c, prev_k_grad_c, prev_v_grad_c, deterministic);
  } else {
    FlashSCABwdBF16(
        y_grad_c, q_c, k_c, v_c, y_c, lse, chunk_size,
        static_cast<float>(scale), prev_k_c, prev_v_c, q_segment_idx_c,
        k_segment_idx_c, has_prev, has_segment, q_grad_c, k_grad_c,
        v_grad_c, prev_k_grad_c, prev_v_grad_c, deterministic);
  }

  torch::Tensor q_grad =
      cast_compute ? q_grad_c.to(q.scalar_type()) : q_grad_c;
  torch::Tensor k_grad =
      cast_compute ? k_grad_c.to(k.scalar_type()) : k_grad_c;
  torch::Tensor v_grad =
      cast_compute ? v_grad_c.to(v.scalar_type()) : v_grad_c;
  c10::optional<torch::Tensor> prev_k_grad = c10::nullopt;
  c10::optional<torch::Tensor> prev_v_grad = c10::nullopt;
  if (has_prev) {
    prev_k_grad =
        cast_compute
            ? c10::make_optional(prev_k_grad_c.value().to(prev_k.value().scalar_type()))
            : prev_k_grad_c;
    prev_v_grad =
        cast_compute
            ? c10::make_optional(prev_v_grad_c.value().to(prev_v.value().scalar_type()))
            : prev_v_grad_c;
  }
  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         c10::optional<torch::Tensor>,
                         c10::optional<torch::Tensor>>(
      std::move(q_grad), std::move(k_grad), std::move(v_grad),
      std::move(prev_k_grad), std::move(prev_v_grad));
}

}
}

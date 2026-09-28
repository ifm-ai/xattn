
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "attention/hopper/launch.h"
#include "attention/hopper/input_layout.h"
#include "attention/partition/segment_metadata.h"
#include "attention/sequence/varlen_metadata.h"
#include "flash_swa/hopper/bwd.h"
#include "flash_swa/hopper/fwd.h"

namespace xattn {
namespace ops {
namespace attention {
namespace hopper {
namespace {

int CurrentDeviceArch(const torch::Tensor& q) {
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  return prop->major * 10 + prop->minor;
}

int64_t RoundUp(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

int HeadDimBucket(int head_dim) {
  return head_dim <= 32
      ? 32
      : (head_dim <= 64
             ? 64
             : (head_dim <= 96
                    ? 96
                    : (head_dim <= 128
                           ? 128
                           : (head_dim <= 160
                                  ? 160
                                  : (head_dim <= 192 ? 192 : 256)))));
}

torch::Tensor PadLastDim(
    const torch::Tensor& tensor, int64_t target_dim) {
  const int64_t last_dim = tensor.dim() - 1;
  if (tensor.size(last_dim) == target_dim) {
    return tensor.contiguous();
  }
  TORCH_CHECK(
      tensor.size(last_dim) < target_dim,
      "Causal attention SM90 pad target must be larger than the tensor head dim");
  std::vector<int64_t> padded_sizes(
      tensor.sizes().begin(), tensor.sizes().end());
  padded_sizes[last_dim] = target_dim;
  torch::Tensor padded = torch::zeros(
      padded_sizes,
      tensor.options().memory_format(at::MemoryFormat::Contiguous));
  padded.narrow(last_dim, 0, tensor.size(last_dim)).copy_(tensor);
  return padded;
}

torch::Tensor EmptyLastDimLike(
    const torch::Tensor& tensor, int64_t target_dim) {
  const int64_t last_dim = tensor.dim() - 1;
  std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
  sizes[last_dim] = target_dim;
  return torch::empty(
      sizes,
      tensor.options().memory_format(at::MemoryFormat::Contiguous));
}

void CheckPreviousKV(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v) {
  TORCH_CHECK(
      prev_k.dim() == 4 && prev_v.dim() == 4,
      "Causal attention SM90 prev_k and prev_v must be 4D tensors");
  TORCH_CHECK(
      prev_k.scalar_type() == q.scalar_type() &&
          prev_v.scalar_type() == q.scalar_type(),
      "Causal attention SM90 prev_k/prev_v dtype must match q/k/v");
  TORCH_CHECK(
      prev_k.size(0) == q.size(0) && prev_v.size(0) == q.size(0),
      "Causal attention SM90 prev_k/prev_v batch size must match q");
  TORCH_CHECK(
      prev_k.size(1) == prev_v.size(1) && prev_k.size(1) > 0,
      "Causal attention SM90 prev_k/prev_v lengths must match and be positive");
  TORCH_CHECK(
      prev_k.size(2) == k.size(2) && prev_v.size(2) == v.size(2),
      "Causal attention SM90 prev_k/prev_v head count must match k/v");
  TORCH_CHECK(
      prev_k.size(3) == k.size(3),
      "Causal attention SM90 prev_k head dim must match k");
  TORCH_CHECK(
      prev_v.size(3) == v.size(3),
      "Causal attention SM90 prev_v head dim must match v");
}

void CheckBackwardTensors(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse) {
  TORCH_CHECK(
      y_grad.dim() == 4 && y.dim() == 4,
      "Causal attention SM90 y_grad and y must be 4D tensors");
  TORCH_CHECK(
      y_grad.scalar_type() == q.scalar_type() &&
          (y.scalar_type() == q.scalar_type() ||
           y.scalar_type() == at::kFloat),
      "Causal attention SM90 y_grad dtype must match q and y must either match "
      "q or be the float32 backward output state");
  TORCH_CHECK(
      y_grad.sizes() == y.sizes(),
      "Causal attention SM90 y_grad and y shapes must match");
  TORCH_CHECK(
      y_grad.size(0) == q.size(0) &&
          y_grad.size(1) == q.size(1) &&
          y_grad.size(2) == q.size(2) &&
          y_grad.size(3) == v.size(3),
      "Causal attention SM90 y_grad/y shape must be [B, L, Hq, V]");
  TORCH_CHECK(
      lse.dim() == 3 && lse.scalar_type() == at::kFloat,
      "Causal attention SM90 lse must be a float32 [B, Hq, L] tensor");
  TORCH_CHECK(
      lse.size(0) == q.size(0) && lse.size(1) == q.size(2) &&
          lse.size(2) == q.size(1),
      "Causal attention SM90 lse shape must be [B, Hq, L]");
}

void CheckPackedQKV(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v) {
  TORCH_CHECK(
      q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
      "Causal attention SM90 varlen q/k/v must be packed 3D tensors "
      "[total_tokens, H, D]");
  TORCH_CHECK(
      q.device().type() == torch::kCUDA &&
          k.device() == q.device() && v.device() == q.device(),
      "Causal attention SM90 varlen q/k/v must be same-device CUDA tensors");
  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() &&
          q.scalar_type() == v.scalar_type(),
      "Causal attention SM90 varlen q/k/v must have the same dtype");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf ||
          q.scalar_type() == at::kBFloat16,
      "Causal attention SM90 varlen supports only fp16 and bf16 input");
  TORCH_CHECK(
      k.size(0) == v.size(0),
      "Causal attention SM90 varlen k/v total token counts must match");
  TORCH_CHECK(
      k.size(1) == v.size(1),
      "Causal attention SM90 varlen k/v head counts must match");
  TORCH_CHECK(
      q.size(1) > 0 && k.size(1) > 0 &&
          q.size(1) % k.size(1) == 0,
      "Causal attention SM90 varlen q head count must be divisible by "
      "the KV head count");
  TORCH_CHECK(
      q.size(2) == k.size(2) && q.size(2) > 0,
      "Causal attention SM90 varlen q/k head dims must match and be positive");
  TORCH_CHECK(
      v.size(2) > 0,
      "Causal attention SM90 varlen value head dim must be positive");
}

void CheckVarlenBackwardTensors(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse) {
  CheckPackedQKV(q, k, v);
  TORCH_CHECK(
      y_grad.dim() == 3 && y.dim() == 3,
      "Causal attention SM90 varlen y_grad/y must be packed 3D tensors");
  TORCH_CHECK(
      y_grad.scalar_type() == q.scalar_type() &&
          (y.scalar_type() == q.scalar_type() ||
           y.scalar_type() == at::kFloat),
      "Causal attention SM90 varlen y_grad dtype must match q and y must either "
      "match q or be the float32 backward output state");
  TORCH_CHECK(
      y_grad.sizes() == y.sizes() &&
          y_grad.size(0) == q.size(0) &&
          y_grad.size(1) == q.size(1) &&
          y_grad.size(2) == v.size(2),
      "Causal attention SM90 varlen y_grad/y shape must be [total_q, Hq, V]");
  TORCH_CHECK(
      lse.dim() == 2 && lse.scalar_type() == at::kFloat &&
          lse.size(0) == q.size(1) && lse.size(1) == q.size(0),
      "Causal attention SM90 varlen lse must be float32 [Hq, total_q]");
}

void FillFwdParams(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, float scale,
    torch::Tensor& y, torch::Tensor& lse,
    const c10::optional<torch::Tensor>& output_state,
    const int64_t* q_segment_idx, const int64_t* k_segment_idx,
    int64_t k_segment_len, int window_size_right,
    AttentionFwdParams& params) {
  std::memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = y.data_ptr();
  params.output_fp32 = y.scalar_type() == at::kFloat;
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    params.o_state_ptr = state.data_ptr<float>();
    params.o_state_batch_stride = state.stride(0);
    params.o_state_row_stride = state.stride(1);
    params.o_state_head_stride = state.stride(2);
  }
  params.softmax_lse_ptr = lse.data_ptr<float>();

  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.q_row_stride = q.stride(1);
  params.k_row_stride = k.stride(1);
  params.v_row_stride = v.stride(1);
  params.q_head_stride = q.stride(2);
  params.k_head_stride = k.stride(2);
  params.v_head_stride = v.stride(2);
  params.o_batch_stride = y.stride(0);
  params.o_row_stride = y.stride(1);
  params.o_head_stride = y.stride(2);

  params.b = static_cast<int>(q.size(0));
  params.num_sequences = params.b;
  params.seqlen_q = static_cast<int>(q.size(1));
  params.seqlen_k = static_cast<int>(k.size(1));
  params.max_seqlen_q = params.seqlen_q;
  params.max_seqlen_k = params.seqlen_k;
  params.d = static_cast<int>(q.size(3));
  params.dv = static_cast<int>(v.size(3));
  params.h = static_cast<int>(q.size(2));
  params.h_k = static_cast<int>(k.size(2));

  params.scale_softmax = scale;
  params.window_size_left = static_cast<int>(window_size);
  params.window_size_right = window_size_right;
  params.attention_chunk = 0;
  params.q_segment_idx = q_segment_idx;
  params.k_segment_idx = k_segment_idx;
  params.k_segment_len = static_cast<int>(k_segment_len);
  params.q_position_offsets = nullptr;
  params.q_chunk_positions = nullptr;
  params.reset_attention_chunk = 0;
  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqused_k = nullptr;
  params.tile_count_semaphore = nullptr;

  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  params.num_sm = prop->multiProcessorCount;
}

void FillBwdParams(
    const torch::Tensor& dy, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, float scale, torch::Tensor& dq,
    torch::Tensor& dk, torch::Tensor& dv,
    const int64_t* q_segment_idx, const int64_t* k_segment_idx,
    int64_t k_segment_len, int window_size_right,
    int odd_head_window_right_delta,
    AttentionBwdParams& params) {
  std::memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.dy_ptr = dy.data_ptr();
  params.y_ptr = y.data_ptr();
  params.y_is_fp32 = y.scalar_type() == at::kFloat;
  params.lse_ptr = lse.data_ptr<float>();
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();

  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.dy_batch_stride = dy.stride(0);
  params.y_batch_stride = y.stride(0);
  params.dq_batch_stride = dq.stride(0);
  params.dk_batch_stride = dk.stride(0);
  params.dv_batch_stride = dv.stride(0);
  params.q_row_stride = q.stride(1);
  params.k_row_stride = k.stride(1);
  params.v_row_stride = v.stride(1);
  params.dy_row_stride = dy.stride(1);
  params.y_row_stride = y.stride(1);
  params.dq_row_stride = dq.stride(1);
  params.dk_row_stride = dk.stride(1);
  params.dv_row_stride = dv.stride(1);
  params.q_head_stride = q.stride(2);
  params.k_head_stride = k.stride(2);
  params.v_head_stride = v.stride(2);
  params.dy_head_stride = dy.stride(2);
  params.y_head_stride = y.stride(2);
  params.dq_head_stride = dq.stride(2);
  params.dk_head_stride = dk.stride(2);
  params.dv_head_stride = dv.stride(2);

  params.b = static_cast<int>(q.size(0));
  params.num_sequences = params.b;
  params.seqlen_q = static_cast<int>(q.size(1));
  params.seqlen_k = static_cast<int>(k.size(1));
  params.max_seqlen_q = params.seqlen_q;
  params.max_seqlen_k = params.seqlen_k;
  params.seqlen_q_padded =
      static_cast<int>(RoundUp(params.seqlen_q, 128));
  params.d = static_cast<int>(q.size(3));
  params.dv = static_cast<int>(v.size(3));
  params.h = static_cast<int>(q.size(2));
  params.h_k = static_cast<int>(k.size(2));

  params.scale_softmax = scale;
  params.window_size_left = static_cast<int>(window_size);
  params.window_size_right = window_size_right;
  params.odd_head_window_right_delta = odd_head_window_right_delta;
  params.attention_chunk = 0;
  params.q_segment_idx = q_segment_idx;
  params.k_segment_idx = k_segment_idx;
  params.k_segment_len = static_cast<int>(k_segment_len);
  params.q_position_offsets = nullptr;
  params.q_chunk_positions = nullptr;
  params.reset_attention_chunk = 0;
  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqused_k = nullptr;

  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  params.num_sm = prop->multiProcessorCount;
}

void FillVarlenFwdParams(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, float scale,
    torch::Tensor& y, torch::Tensor& lse,
    const c10::optional<torch::Tensor>& output_state,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    int num_sequences, int64_t max_seqlen_q,
    int64_t max_seqlen_k, bool strict_past,
    AttentionFwdParams& params) {
  std::memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = y.data_ptr();
  params.output_fp32 = y.scalar_type() == at::kFloat;
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    params.o_state_ptr = state.data_ptr<float>();
    params.o_state_batch_stride = 0;
    params.o_state_row_stride = state.stride(0);
    params.o_state_head_stride = state.stride(1);
  }
  params.softmax_lse_ptr = lse.data_ptr<float>();

  params.q_batch_stride = 0;
  params.k_batch_stride = 0;
  params.v_batch_stride = 0;
  params.q_row_stride = q.stride(0);
  params.k_row_stride = k.stride(0);
  params.v_row_stride = v.stride(0);
  params.q_head_stride = q.stride(1);
  params.k_head_stride = k.stride(1);
  params.v_head_stride = v.stride(1);
  params.o_batch_stride = 0;
  params.o_row_stride = y.stride(0);
  params.o_head_stride = y.stride(1);

  params.b = 1;
  params.num_sequences = num_sequences;
  params.seqlen_q = static_cast<int>(q.size(0));
  params.seqlen_k = static_cast<int>(k.size(0));
  params.max_seqlen_q = static_cast<int>(max_seqlen_q);
  params.max_seqlen_k = static_cast<int>(max_seqlen_k);
  params.d = static_cast<int>(q.size(2));
  params.dv = static_cast<int>(v.size(2));
  params.h = static_cast<int>(q.size(1));
  params.h_k = static_cast<int>(k.size(1));

  params.scale_softmax = scale;
  params.window_size_left = static_cast<int>(window_size);
  params.window_size_right = strict_past ? -1 : 0;
  params.attention_chunk = 0;
  params.q_segment_idx = nullptr;
  params.k_segment_idx = nullptr;
  params.k_segment_len = 0;
  params.q_position_offsets = nullptr;
  params.q_chunk_positions = nullptr;
  params.reset_attention_chunk = 0;
  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_k = cu_seqlens_k;
  params.seqused_k = nullptr;
  params.tile_count_semaphore = nullptr;

  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  params.num_sm = prop->multiProcessorCount;
}

void FillVarlenBwdParams(
    const torch::Tensor& dy, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, float scale, torch::Tensor& dq,
    torch::Tensor& dk, torch::Tensor& dv,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    int num_sequences, int64_t max_seqlen_q,
    int64_t max_seqlen_k, int64_t seqlen_q_padded,
    bool strict_past, AttentionBwdParams& params) {
  std::memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.dy_ptr = dy.data_ptr();
  params.y_ptr = y.data_ptr();
  params.y_is_fp32 = y.scalar_type() == at::kFloat;
  params.lse_ptr = lse.data_ptr<float>();
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();

  params.q_batch_stride = 0;
  params.k_batch_stride = 0;
  params.v_batch_stride = 0;
  params.dy_batch_stride = 0;
  params.y_batch_stride = 0;
  params.dq_batch_stride = 0;
  params.dk_batch_stride = 0;
  params.dv_batch_stride = 0;
  params.q_row_stride = q.stride(0);
  params.k_row_stride = k.stride(0);
  params.v_row_stride = v.stride(0);
  params.dy_row_stride = dy.stride(0);
  params.y_row_stride = y.stride(0);
  params.dq_row_stride = dq.stride(0);
  params.dk_row_stride = dk.stride(0);
  params.dv_row_stride = dv.stride(0);
  params.q_head_stride = q.stride(1);
  params.k_head_stride = k.stride(1);
  params.v_head_stride = v.stride(1);
  params.dy_head_stride = dy.stride(1);
  params.y_head_stride = y.stride(1);
  params.dq_head_stride = dq.stride(1);
  params.dk_head_stride = dk.stride(1);
  params.dv_head_stride = dv.stride(1);

  params.b = 1;
  params.num_sequences = num_sequences;
  params.seqlen_q = static_cast<int>(q.size(0));
  params.seqlen_k = static_cast<int>(k.size(0));
  params.max_seqlen_q = static_cast<int>(max_seqlen_q);
  params.max_seqlen_k = static_cast<int>(max_seqlen_k);
  params.seqlen_q_padded = static_cast<int>(seqlen_q_padded);
  params.d = static_cast<int>(q.size(2));
  params.dv = static_cast<int>(v.size(2));
  params.h = static_cast<int>(q.size(1));
  params.h_k = static_cast<int>(k.size(1));

  params.scale_softmax = scale;
  params.window_size_left = static_cast<int>(window_size);
  params.window_size_right = strict_past ? -1 : 0;
  params.attention_chunk = 0;
  params.q_segment_idx = nullptr;
  params.k_segment_idx = nullptr;
  params.k_segment_len = 0;
  params.q_position_offsets = nullptr;
  params.q_chunk_positions = nullptr;
  params.reset_attention_chunk = 0;
  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_k = cu_seqlens_k;
  params.seqused_k = nullptr;

  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  params.num_sm = prop->multiProcessorCount;
}

template <
    typename Element, int kHeadDim, int kHeadDimV,
    bool kVarlen, bool kHasSegment,
    typename Visibility =
        attention::semantics::CausalSlidingWindowVisibility>
void DispatchFwdBucket(
    AttentionFwdParams& params, cudaStream_t stream) {
  flash_swa::RunFlashSWAFwdSm90VD<
      Element, Element, kVarlen, kHasSegment, kHeadDim, kHeadDimV,
      Visibility>(
      params, stream);
}

template <
    typename Element, int kHeadDim, bool kVarlen,
    bool kHasSegment,
    typename Visibility =
        attention::semantics::CausalSlidingWindowVisibility>
void DispatchFwdV(AttentionFwdParams& params, cudaStream_t stream) {
  if (params.dv <= 32) {
    DispatchFwdBucket<
        Element, kHeadDim, 32, kVarlen, kHasSegment, Visibility>(
        params, stream);
  } else if (params.dv <= 64) {
    DispatchFwdBucket<
        Element, kHeadDim, 64, kVarlen, kHasSegment, Visibility>(
        params, stream);
  } else if (params.dv <= 96) {
    DispatchFwdBucket<
        Element, kHeadDim, 96, kVarlen, kHasSegment, Visibility>(
        params, stream);
  } else if (params.dv <= 128) {
    DispatchFwdBucket<
        Element, kHeadDim, 128, kVarlen, kHasSegment, Visibility>(
        params, stream);
  } else if (params.dv <= 160) {
    DispatchFwdBucket<
        Element, kHeadDim, 160, kVarlen, kHasSegment, Visibility>(
        params, stream);
  } else if (params.dv <= 192) {
    DispatchFwdBucket<
        Element, kHeadDim, 192, kVarlen, kHasSegment, Visibility>(
        params, stream);
  } else if (params.dv <= 256) {
    DispatchFwdBucket<
        Element, kHeadDim, 256, kVarlen, kHasSegment, Visibility>(
        params, stream);
  } else {
    TORCH_CHECK(
        false, "Causal attention SM90 FWD supports V <= 256; got V=",
        params.dv);
  }
}

template <typename Element, int kHeadDim>
void DispatchCausalFlashAttnFwdV(
    AttentionFwdParams& params, cudaStream_t stream) {
  DispatchFwdV<
      Element, kHeadDim, false, false,
      attention::semantics::CausalFullVisibility>(params, stream);
}

template <typename Element>
void DispatchCausalFlashAttnFwd(
    AttentionFwdParams& params, cudaStream_t stream) {
  if (params.d <= 32) {
    DispatchCausalFlashAttnFwdV<Element, 32>(params, stream);
  } else if (params.d <= 64) {
    DispatchCausalFlashAttnFwdV<Element, 64>(params, stream);
  } else if (params.d <= 96) {
    DispatchCausalFlashAttnFwdV<Element, 96>(params, stream);
  } else if (params.d <= 128) {
    DispatchCausalFlashAttnFwdV<Element, 128>(params, stream);
  } else if (params.d <= 160) {
    DispatchCausalFlashAttnFwdV<Element, 160>(params, stream);
  } else if (params.d <= 192) {
    DispatchCausalFlashAttnFwdV<Element, 192>(params, stream);
  } else if (params.d <= 256) {
    DispatchCausalFlashAttnFwdV<Element, 256>(params, stream);
  } else {
    TORCH_CHECK(
        false, "Causal flash attention SM90 FWD supports D <= 256; got D=",
        params.d);
  }
}

template <typename Element, int kHeadDim>
void DispatchFwdSegment(
    AttentionFwdParams& params, cudaStream_t stream) {
  if (params.q_segment_idx == nullptr) {
    DispatchFwdV<Element, kHeadDim, false, false>(params, stream);
  } else {
    DispatchFwdV<Element, kHeadDim, false, true>(params, stream);
  }
}

template <typename Element>
void DispatchFwd(AttentionFwdParams& params, cudaStream_t stream) {
  if (params.d <= 32) {
    DispatchFwdSegment<Element, 32>(params, stream);
  } else if (params.d <= 64) {
    DispatchFwdSegment<Element, 64>(params, stream);
  } else if (params.d <= 96) {
    DispatchFwdSegment<Element, 96>(params, stream);
  } else if (params.d <= 128) {
    DispatchFwdSegment<Element, 128>(params, stream);
  } else if (params.d <= 160) {
    DispatchFwdSegment<Element, 160>(params, stream);
  } else if (params.d <= 192) {
    DispatchFwdSegment<Element, 192>(params, stream);
  } else if (params.d <= 256) {
    DispatchFwdSegment<Element, 256>(params, stream);
  } else {
    TORCH_CHECK(
        false, "Causal attention SM90 FWD supports D <= 256; got D=",
        params.d);
  }
}

template <typename Element, int kHeadDim>
void DispatchVarlenFwdV(
    AttentionFwdParams& params, cudaStream_t stream) {
  DispatchFwdV<Element, kHeadDim, true, false>(params, stream);
}

template <typename Element>
void DispatchVarlenFwd(
    AttentionFwdParams& params, cudaStream_t stream) {
  if (params.d <= 32) {
    DispatchVarlenFwdV<Element, 32>(params, stream);
  } else if (params.d <= 64) {
    DispatchVarlenFwdV<Element, 64>(params, stream);
  } else if (params.d <= 96) {
    DispatchVarlenFwdV<Element, 96>(params, stream);
  } else if (params.d <= 128) {
    DispatchVarlenFwdV<Element, 128>(params, stream);
  } else if (params.d <= 160) {
    DispatchVarlenFwdV<Element, 160>(params, stream);
  } else if (params.d <= 192) {
    DispatchVarlenFwdV<Element, 192>(params, stream);
  } else if (params.d <= 256) {
    DispatchVarlenFwdV<Element, 256>(params, stream);
  } else {
    TORCH_CHECK(
        false, "Causal attention SM90 varlen FWD supports D <= 256; got D=",
        params.d);
  }
}

template <bool kFull, typename Element, int kHeadDim, int kHeadDimV,
          int kBlockN>
void RunBwdVariant(
    AttentionBwdParams& params, bool deterministic,
    cudaStream_t stream) {
  if constexpr (kFull) {
    flash_swa::RunCausalFlashAttnBwdSm90Variant<
        Element, kHeadDim, kHeadDimV, kBlockN, 0>(
        params, deterministic, stream);
  } else {
    flash_swa::RunFlashSWABwdSm90Variant<
        Element, kHeadDim, kHeadDimV, kBlockN, 0>(
        params, deterministic, stream);
  }
}

template <bool kFull, typename Element, int kHeadDim, int kHeadDimV,
          int kBlockN>
void RunBwdNonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  if constexpr (kFull) {
    flash_swa::RunCausalFlashAttnBwdSm90NonDetVariant<
        Element, kHeadDim, kHeadDimV, kBlockN, 0>(params, stream);
  } else {
    flash_swa::RunFlashSWABwdSm90NonDetVariant<
        Element, kHeadDim, kHeadDimV, kBlockN, 0>(params, stream);
  }
}

template <typename Element, int kHeadDim, int kHeadDimV,
          bool kFull = false>
void DispatchBwdBucket(
    AttentionBwdParams& params, bool deterministic,
    cudaStream_t stream) {
  if constexpr (kHeadDim == 192 && kHeadDimV == 160) {
    // Two-component dV for unequal-length GQA needs a smaller N tile.
    if (params.seqlen_q < params.seqlen_k && params.h != params.h_k) {
      RunBwdVariant<kFull, Element, kHeadDim, kHeadDimV, 64>(
          params, deterministic, stream);
      return;
    }
  }
  if constexpr (kHeadDim == 256) {
    if (deterministic) {
      RunBwdVariant<kFull, Element, kHeadDim, kHeadDimV, 64>(
          params, deterministic, stream);
    } else if constexpr (kHeadDimV == 256) {
      RunBwdNonDetVariant<kFull, Element, kHeadDim, kHeadDimV, 80>(
          params, stream);
    } else {
      RunBwdNonDetVariant<kFull, Element, kHeadDim, kHeadDimV, 128>(
          params, stream);
    }
  } else if constexpr (
      kHeadDim == 192 && kHeadDimV == 192) {
    RunBwdVariant<kFull, Element, kHeadDim, kHeadDimV, 96>(
        params, deterministic, stream);
  } else if constexpr (
      kHeadDim == 192 && kHeadDimV == 256) {
    RunBwdVariant<kFull, Element, kHeadDim, kHeadDimV, 64>(
        params, deterministic, stream);
  } else if constexpr (
      kHeadDim == 160 &&
      (kHeadDimV == 160 || kHeadDimV == 256)) {
    RunBwdVariant<kFull, Element, kHeadDim, kHeadDimV, 64>(
        params, deterministic, stream);
  } else {
    RunBwdVariant<kFull, Element, kHeadDim, kHeadDimV, 128>(
        params, deterministic, stream);
  }
}

template <typename Element, int kHeadDim, bool kFull = false>
void DispatchBwdV(
    AttentionBwdParams& params, bool deterministic,
    cudaStream_t stream) {
  if (params.dv <= 32) {
    DispatchBwdBucket<Element, kHeadDim, 32, kFull>(
        params, deterministic, stream);
  } else if (params.dv <= 64) {
    DispatchBwdBucket<Element, kHeadDim, 64, kFull>(
        params, deterministic, stream);
  } else if (params.dv <= 96) {
    DispatchBwdBucket<Element, kHeadDim, 96, kFull>(
        params, deterministic, stream);
  } else if (params.dv <= 128) {
    DispatchBwdBucket<Element, kHeadDim, 128, kFull>(
        params, deterministic, stream);
  } else if (params.dv <= 160) {
    DispatchBwdBucket<Element, kHeadDim, 160, kFull>(
        params, deterministic, stream);
  } else if (params.dv <= 192) {
    DispatchBwdBucket<Element, kHeadDim, 192, kFull>(
        params, deterministic, stream);
  } else if (params.dv <= 256) {
    DispatchBwdBucket<Element, kHeadDim, 256, kFull>(
        params, deterministic, stream);
  } else {
    TORCH_CHECK(
        false, "Causal attention SM90 BWD supports padded V <= 256; got V=",
        params.dv);
  }
}

// Square dKV layout for selected mixed shapes; TMA loads use runtime D/V bounds.
// Deterministic D256 uses rectangular kernels.
int CausalFlashAttnBwdMixedSquareDim(int d, int dv, bool deterministic) {
  if (deterministic && d == 192 && dv == 96) {
    return 192;
  }
  if ((dv == 192 && (d == 128 || d == 160)) ||
      (d == 192 && dv == 160)) {
    return 192;
  }
  if (!deterministic &&
      ((d == 128 && dv == 256) ||
       (d == 256 && (dv == 128 || dv == 160 || dv == 192)))) {
    return 256;
  }
  return 0;
}

template <typename Element>
void DispatchCausalFlashAttnBwd(
    AttentionBwdParams& params, bool deterministic,
    cudaStream_t stream) {
  const bool use_d192 =
      (params.d == 192 && params.dv == 128) ||
      (params.d == 160 && params.dv == 128) ||
      (params.d == 160 && params.dv == 160);
  const int mixed_square_dim =
      params.seqlen_q == params.seqlen_k &&
              params.odd_head_window_right_delta == 0
          ? CausalFlashAttnBwdMixedSquareDim(params.d, params.dv, deterministic)
          : 0;
  // Deterministic short-Q GQA uses BN48 for the D192 shared-memory budget.
  if (deterministic && params.seqlen_q < params.seqlen_k &&
      params.h != params.h_k &&
      (use_d192 || (params.d == 192 && params.dv == 192))) {
    RunBwdVariant<true, Element, 192, 192, 48>(params, true, stream);
  } else if (mixed_square_dim == 256 ||
      (!deterministic && params.d == 64 && params.dv == 256)) {
    DispatchBwdBucket<Element, 256, 256, true>(
        params, deterministic, stream);
  } else if (params.d == 64 && params.dv == 96) {
    DispatchBwdBucket<Element, 96, 96, true>(
        params, deterministic, stream);
  } else if (use_d192 || mixed_square_dim == 192) {
    DispatchBwdBucket<Element, 192, 192, true>(
        params, deterministic, stream);
  } else if (params.d <= 32) {
    DispatchBwdV<Element, 32, true>(params, deterministic, stream);
  } else if (params.d <= 64) {
    DispatchBwdV<Element, 64, true>(params, deterministic, stream);
  } else if (params.d <= 96) {
    DispatchBwdV<Element, 96, true>(params, deterministic, stream);
  } else if (params.d <= 128) {
    DispatchBwdV<Element, 128, true>(params, deterministic, stream);
  } else if (params.d <= 160) {
    DispatchBwdV<Element, 160, true>(params, deterministic, stream);
  } else if (params.d <= 192) {
    DispatchBwdV<Element, 192, true>(params, deterministic, stream);
  } else if (params.d <= 256) {
    DispatchBwdV<Element, 256, true>(params, deterministic, stream);
  } else {
    TORCH_CHECK(
        false, "Causal flash attention SM90 BWD supports D <= 256; got D=",
        params.d);
  }
}

bool UseNativeWindow160Bwd(
    int d, int dv, int batch, int length_q, int length_k,
    int heads_q, int heads_k, int window_left, int window_right,
    int odd_head_delta, bool deterministic) {
  return !deterministic && d == 160 && dv == 160 && batch == 1 &&
      length_q == length_k && length_q >= 4096 && heads_q == 8 &&
      (heads_k == 1 || heads_k == 2) && window_left == 255 &&
      window_right == 0 && odd_head_delta == 0;
}

template <typename Element>
void DispatchBwd(
    AttentionBwdParams& params, bool deterministic,
    cudaStream_t stream) {
  const bool dense = params.q_segment_idx == nullptr;
  const bool native_window160 = dense && UseNativeWindow160Bwd(
      params.d, params.dv, params.b, params.seqlen_q, params.seqlen_k,
      params.h, params.h_k, params.window_size_left, params.window_size_right,
      params.odd_head_window_right_delta, deterministic);
  const bool use_d192 =
      dense && !native_window160 &&
      ((params.d == 192 && params.dv == 128) ||
       (params.d == 160 && (params.dv == 128 || params.dv == 160)));
  if (dense && params.d == 64 && params.dv == 96) {
    DispatchBwdBucket<Element, 96, 96>(
        params, deterministic, stream);
  } else if (use_d192) {
    DispatchBwdBucket<Element, 192, 192>(
        params, deterministic, stream);
  } else if (params.d <= 32) {
    DispatchBwdV<Element, 32>(params, deterministic, stream);
  } else if (params.d <= 64) {
    DispatchBwdV<Element, 64>(params, deterministic, stream);
  } else if (params.d <= 96) {
    DispatchBwdV<Element, 96>(params, deterministic, stream);
  } else if (params.d <= 128) {
    DispatchBwdV<Element, 128>(params, deterministic, stream);
  } else if (params.d <= 160) {
    DispatchBwdV<Element, 160>(params, deterministic, stream);
  } else if (params.d <= 192) {
    DispatchBwdV<Element, 192>(params, deterministic, stream);
  } else if (params.d <= 256) {
    DispatchBwdV<Element, 256>(params, deterministic, stream);
  } else {
    TORCH_CHECK(
        false, "Causal attention SM90 BWD supports padded D <= 256; got D=",
        params.d);
  }
}

template <typename Element, int kHeadDim, int kHeadDimV>
void DispatchVarlenBwdBucket(
    AttentionBwdParams& params, bool deterministic,
    cudaStream_t stream) {
  if constexpr (kHeadDim == 256) {
    if (deterministic) {
      if constexpr (kHeadDimV <= 128) {
        flash_swa::RunFlashSWABwdSm90VarlenVariant<
            Element, 256, 128, 64, 0>(
            params, deterministic, stream);
      } else {
        flash_swa::RunFlashSWABwdSm90VarlenVariant<
            Element, 256, 256, 64, 0>(
            params, deterministic, stream);
      }
    } else {
      constexpr int kBlockN = kHeadDimV == 256 ? 80 : 128;
      flash_swa::RunFlashSWABwdSm90VarlenNonDetVariant<
          Element, 256, kHeadDimV, kBlockN, 0>(params, stream);
    }
  } else if constexpr (
      kHeadDim == 192 && kHeadDimV == 192) {
    flash_swa::RunFlashSWABwdSm90VarlenVariant<
        Element, kHeadDim, kHeadDimV, 96, 0>(
        params, deterministic, stream);
  } else {
    flash_swa::RunFlashSWABwdSm90VarlenVariant<
        Element, kHeadDim, kHeadDimV, 128, 0>(
        params, deterministic, stream);
  }
}

template <typename Element, int kHeadDim>
void DispatchVarlenBwdV(
    AttentionBwdParams& params, bool deterministic,
    cudaStream_t stream) {
  if constexpr (kHeadDim == 256) {
    if (params.dv <= 64) {
      DispatchVarlenBwdBucket<Element, 256, 64>(
          params, deterministic, stream);
    } else if (params.dv <= 96) {
      DispatchVarlenBwdBucket<Element, 256, 96>(
          params, deterministic, stream);
    } else if (params.dv <= 128) {
      DispatchVarlenBwdBucket<Element, 256, 128>(
          params, deterministic, stream);
    } else if (params.dv <= 160) {
      DispatchVarlenBwdBucket<Element, 256, 160>(
          params, deterministic, stream);
    } else if (params.dv <= 192) {
      DispatchVarlenBwdBucket<Element, 256, 192>(
          params, deterministic, stream);
    } else if (params.dv <= 256) {
      DispatchVarlenBwdBucket<Element, 256, 256>(
          params, deterministic, stream);
    } else {
      TORCH_CHECK(
          false,
          "Causal attention SM90 varlen BWD supports padded V <= 256; got V=",
          params.dv);
    }
  } else if (params.dv <= 32) {
    DispatchVarlenBwdBucket<Element, kHeadDim, 32>(
        params, deterministic, stream);
  } else if (params.dv <= 64) {
    DispatchVarlenBwdBucket<Element, kHeadDim, 64>(
        params, deterministic, stream);
  } else if (params.dv <= 96) {
    DispatchVarlenBwdBucket<Element, kHeadDim, 96>(
        params, deterministic, stream);
  } else if (params.dv <= 128) {
    DispatchVarlenBwdBucket<Element, kHeadDim, 128>(
        params, deterministic, stream);
  } else if (params.dv <= 160) {
    DispatchVarlenBwdBucket<Element, kHeadDim, 160>(
        params, deterministic, stream);
  } else if (params.dv <= 192) {
    DispatchVarlenBwdBucket<Element, kHeadDim, 192>(
        params, deterministic, stream);
  } else if (params.dv <= 256) {
    if constexpr (kHeadDim == 160 || kHeadDim == 192) {
      TORCH_CHECK(
          false,
          "Causal attention SM90 varlen BWD requires rounded D=160/192,V=256 "
          "to be padded to D=256");
    } else {
      DispatchVarlenBwdBucket<Element, kHeadDim, 256>(
          params, deterministic, stream);
    }
  } else {
    TORCH_CHECK(
        false,
        "Causal attention SM90 varlen BWD supports padded V <= 256; got V=",
        params.dv);
  }
}

template <typename Element>
void DispatchVarlenBwd(
    AttentionBwdParams& params, bool deterministic,
    cudaStream_t stream) {
  if (params.d <= 32) {
    DispatchVarlenBwdV<Element, 32>(params, deterministic, stream);
  } else if (params.d <= 64) {
    DispatchVarlenBwdV<Element, 64>(params, deterministic, stream);
  } else if (params.d <= 96) {
    DispatchVarlenBwdV<Element, 96>(params, deterministic, stream);
  } else if (params.d <= 128) {
    DispatchVarlenBwdV<Element, 128>(params, deterministic, stream);
  } else if (params.d <= 160) {
    DispatchVarlenBwdV<Element, 160>(params, deterministic, stream);
  } else if (params.d <= 192) {
    DispatchVarlenBwdV<Element, 192>(params, deterministic, stream);
  } else if (params.d <= 256) {
    DispatchVarlenBwdV<Element, 256>(params, deterministic, stream);
  } else {
    TORCH_CHECK(
        false,
        "Causal attention SM90 varlen BWD supports padded D <= 256; got D=",
        params.d);
  }
}

}  // namespace

bool CausalAttentionSM90Available(const torch::Tensor& q) {
  TORCH_CHECK(
      q.device().type() == torch::kCUDA,
      "Causal attention SM90 dispatch expects q to be CUDA");
  return CurrentDeviceArch(q) == 90;
}

std::tuple<torch::Tensor, torch::Tensor> CausalAttentionSM90Fwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool strict_past,
    const c10::optional<torch::Tensor>& output_state,
    bool causal_flash_attn, bool use_fast_reciprocal, bool output_fp32) {
  const int arch = CurrentDeviceArch(q);
  TORCH_CHECK(
      arch == 90, "Causal attention SM90 FWD requires Hopper sm90, got sm",
      arch);
  TORCH_CHECK(
      window_size >= 0 &&
          window_size <= std::numeric_limits<int>::max(),
      "Causal attention SM90 FWD window_size must fit int32");
  TORCH_CHECK(
      prev_k.has_value() == prev_v.has_value(),
      "Causal attention prev_k and prev_v must both be provided or both be None");
  TORCH_CHECK(
      q.size(3) <= 256 && v.size(3) <= 256,
      "Causal attention SM90 FWD supports D,V <= 256; got D=",
      q.size(3), ", V=", v.size(3));
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  // Reader-pair head slices have TMA-compatible strides.
  torch::Tensor q_c = CanUseStridedTmaQuery(q) ? q : q.contiguous();
  torch::Tensor k_c = k.contiguous();
  torch::Tensor v_c = v.contiguous();
  const bool has_prev = prev_k.has_value();
  const int64_t prev_length =
      has_prev ? prev_k.value().size(1) : int64_t(0);
  if (has_prev) {
    CheckPreviousKV(q, k, v, prev_k.value(), prev_v.value());
    k_c = torch::cat({prev_k.value().contiguous(), k_c}, 1);
    v_c = torch::cat({prev_v.value().contiguous(), v_c}, 1);
  }

  attention::partition::SegmentMetadata segment_metadata =
      attention::partition::NormalizeSegmentMetadata(
          q_segment_idx, k_segment_idx, q.size(0), q.size(1),
          k.size(1) + prev_length, q.device(), "FlashSWA");
  const int64_t* q_segment_idx_data = nullptr;
  const int64_t* k_segment_idx_data = nullptr;
  int64_t k_segment_len = 0;
  if (segment_metadata.enabled()) {
    q_segment_idx_data =
        segment_metadata.q_segment_idx.data_ptr<int64_t>();
    k_segment_idx_data =
        segment_metadata.k_segment_idx.data_ptr<int64_t>();
    k_segment_len = segment_metadata.k_length();
  }
  TORCH_CHECK(
      k_segment_len <= std::numeric_limits<int>::max(),
      "Causal attention SM90 FWD k_segment_idx is too long");

  const int64_t B = q.size(0);
  const int64_t L = q.size(1);
  const int64_t H = q.size(2);
  const int64_t V = v.size(3);
  TORCH_CHECK(!output_fp32 || !output_state.has_value(),
              "FP32-only output and output_state are mutually exclusive");
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    TORCH_CHECK(
        state.device() == q.device() && state.scalar_type() == at::kFloat &&
            state.is_contiguous() && state.dim() == 4 &&
            state.size(0) == B && state.size(1) == L &&
            state.size(2) == H && state.size(3) == V,
        "Causal attention SM90 output_state must be contiguous float32 "
        "[B, L, Hq, V] on the Q device");
  }
  torch::Tensor y = torch::empty(
      {B, L, H, V},
      v.options().dtype(output_fp32 ? at::kFloat : v.scalar_type())
          .memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor lse = torch::empty(
      {B, H, L},
      q.options().dtype(at::kFloat).memory_format(
          at::MemoryFormat::Contiguous));

  AttentionFwdParams params;
  FillFwdParams(
      q_c, k_c, v_c, window_size, static_cast<float>(scale), y, lse,
      output_state,
      q_segment_idx_data, k_segment_idx_data, k_segment_len,
      strict_past ? -1 : 0, params);
  params.use_fast_reciprocal = use_fast_reciprocal;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_full_visibility =
      causal_flash_attn && !segment_metadata.enabled();
  if (q.scalar_type() == at::kHalf) {
    if (use_full_visibility) {
      DispatchCausalFlashAttnFwd<cutlass::half_t>(params, stream);
    } else {
      DispatchFwd<cutlass::half_t>(params, stream);
    }
  } else {
    if (use_full_visibility) {
      DispatchCausalFlashAttnFwd<cutlass::bfloat16_t>(params, stream);
    } else {
      DispatchFwd<cutlass::bfloat16_t>(params, stream);
    }
  }
  return std::make_tuple<torch::Tensor, torch::Tensor>(
      std::move(y), std::move(lse));
}

void AttentionSM90FwdInto(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool strict_past,
    const c10::optional<torch::Tensor>& output_state,
    torch::Tensor& y,
    torch::Tensor& lse, bool causal_flash_attn) {
  const int arch = CurrentDeviceArch(q);
  TORCH_CHECK(
      arch == 90, "Causal attention SM90 FWD requires Hopper sm90, got sm",
      arch);
  TORCH_CHECK(
      window_size >= 0 &&
          window_size <= std::numeric_limits<int>::max(),
      "Causal attention SM90 FWD window_size must fit int32");
  TORCH_CHECK(
      prev_k.has_value() == prev_v.has_value(),
      "Causal attention prev_k and prev_v must both be provided or both be None");
  TORCH_CHECK(
      q.size(3) <= 256 && v.size(3) <= 256,
      "Causal attention SM90 FWD supports D,V <= 256; got D=",
      q.size(3), ", V=", v.size(3));
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  torch::Tensor q_c = q.stride(3) == 1 ? q : q.contiguous();
  torch::Tensor k_c = k.contiguous();
  torch::Tensor v_c = v.contiguous();
  const bool has_prev = prev_k.has_value();
  const int64_t prev_length =
      has_prev ? prev_k.value().size(1) : int64_t(0);
  if (has_prev) {
    CheckPreviousKV(q, k, v, prev_k.value(), prev_v.value());
    k_c = torch::cat({prev_k.value().contiguous(), k_c}, 1);
    v_c = torch::cat({prev_v.value().contiguous(), v_c}, 1);
  }

  attention::partition::SegmentMetadata segment_metadata =
      attention::partition::NormalizeSegmentMetadata(
          q_segment_idx, k_segment_idx, q.size(0), q.size(1),
          k.size(1) + prev_length, q.device(), "FlashSWA");
  const int64_t* q_segment_idx_data = nullptr;
  const int64_t* k_segment_idx_data = nullptr;
  int64_t k_segment_len = 0;
  if (segment_metadata.enabled()) {
    q_segment_idx_data =
        segment_metadata.q_segment_idx.data_ptr<int64_t>();
    k_segment_idx_data =
        segment_metadata.k_segment_idx.data_ptr<int64_t>();
    k_segment_len = segment_metadata.k_length();
  }
  TORCH_CHECK(
      k_segment_len <= std::numeric_limits<int>::max(),
      "Causal attention SM90 FWD k_segment_idx is too long");

  const int64_t B = q.size(0);
  const int64_t L = q.size(1);
  const int64_t H = q.size(2);
  const int64_t V = v.size(3);
  TORCH_CHECK(
      y.device() == q.device() && y.scalar_type() == v.scalar_type() &&
          y.dim() == 4 && y.size(0) == B && y.size(1) == L &&
          y.size(2) == H && y.size(3) == V && y.stride(3) == 1,
      "Causal attention SM90 FWD output view must be [B, L, Hq, V] with "
      "unit value stride on the Q device");
  TORCH_CHECK(
      lse.device() == q.device() && lse.scalar_type() == at::kFloat &&
          lse.is_contiguous() && lse.dim() == 3 &&
          lse.size(0) == B && lse.size(1) == H && lse.size(2) == L,
      "Causal attention SM90 FWD LSE output must be contiguous float32 "
      "[B, Hq, L] on the Q device");
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    TORCH_CHECK(
        state.device() == q.device() && state.scalar_type() == at::kFloat &&
            state.dim() == 4 && state.size(0) == B &&
            state.size(1) == L && state.size(2) == H &&
            state.size(3) == V && state.stride(3) == 1,
        "Causal attention SM90 FWD output_state view must be float32 "
        "[B, L, Hq, V] with unit value stride on the Q device");
  }

  AttentionFwdParams params;
  FillFwdParams(
      q_c, k_c, v_c, window_size, static_cast<float>(scale), y, lse,
      output_state,
      q_segment_idx_data, k_segment_idx_data, k_segment_len,
      strict_past ? -1 : 0, params);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_full_visibility =
      causal_flash_attn && !segment_metadata.enabled();
  if (q.scalar_type() == at::kHalf) {
    if (use_full_visibility) {
      DispatchCausalFlashAttnFwd<cutlass::half_t>(params, stream);
    } else {
      DispatchFwd<cutlass::half_t>(params, stream);
    }
  } else {
    if (use_full_visibility) {
      DispatchCausalFlashAttnFwd<cutlass::bfloat16_t>(params, stream);
    } else {
      DispatchFwd<cutlass::bfloat16_t>(params, stream);
    }
  }
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
CausalAttentionSM90BwdWithHeadBoundary(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, bool strict_past,
    int odd_head_window_right_delta, bool causal_flash_attn) {
  const int arch = CurrentDeviceArch(q);
  TORCH_CHECK(
      arch == 90, "Causal attention SM90 BWD requires Hopper sm90, got sm",
      arch);
  TORCH_CHECK(
      window_size >= 0 &&
          window_size <= std::numeric_limits<int>::max(),
      "Causal attention SM90 BWD window_size must fit int32");
  TORCH_CHECK(
      odd_head_window_right_delta == 0 ||
          odd_head_window_right_delta == -1,
      "Causal attention SM90 BWD odd-head boundary delta must be 0 or -1");
  TORCH_CHECK(
      odd_head_window_right_delta == 0 ||
          (!strict_past && q.size(2) % 2 == 0),
      "Causal attention SM90 BWD odd-head boundary mode requires inclusive "
      "base semantics and interleaved head pairs");
  TORCH_CHECK(
      prev_k.has_value() == prev_v.has_value(),
      "Causal attention prev_k and prev_v must both be provided or both be None");
  CheckBackwardTensors(y_grad, q, v, y, lse);
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  torch::Tensor q_c = q.contiguous();
  torch::Tensor k_c = k.contiguous();
  torch::Tensor v_c = v.contiguous();
  torch::Tensor dy_c = y_grad.contiguous();
  torch::Tensor y_c = y.contiguous();
  torch::Tensor lse_c = lse.contiguous();
  // A contiguous view can have an unaligned storage offset. TMA requires
  // aligned base addresses even when head dimensions need no padding.
  for (torch::Tensor* input : {&q_c, &k_c, &v_c, &dy_c, &y_c}) {
    if (reinterpret_cast<std::uintptr_t>(input->data_ptr()) % 16 != 0) {
      *input = input->clone();
    }
  }
  const bool has_prev = prev_k.has_value();
  const int64_t prev_length =
      has_prev ? prev_k.value().size(1) : int64_t(0);
  if (has_prev) {
    CheckPreviousKV(q, k, v, prev_k.value(), prev_v.value());
    k_c = torch::cat({prev_k.value().contiguous(), k_c}, 1);
    v_c = torch::cat({prev_v.value().contiguous(), v_c}, 1);
  }

  attention::partition::SegmentMetadata segment_metadata =
      attention::partition::NormalizeSegmentMetadata(
          q_segment_idx, k_segment_idx, q.size(0), q.size(1),
          k.size(1) + prev_length, q.device(), "FlashSWA");
  const int64_t* q_segment_idx_data = nullptr;
  const int64_t* k_segment_idx_data = nullptr;
  int64_t k_segment_len = 0;
  if (segment_metadata.enabled()) {
    q_segment_idx_data =
        segment_metadata.q_segment_idx.data_ptr<int64_t>();
    k_segment_idx_data =
        segment_metadata.k_segment_idx.data_ptr<int64_t>();
    k_segment_len = segment_metadata.k_length();
  }
  TORCH_CHECK(
      k_segment_len <= std::numeric_limits<int>::max(),
      "Causal attention SM90 BWD k_segment_idx is too long");

  const int64_t D = q.size(3);
  const int64_t V = v.size(3);
  const bool dense = !segment_metadata.enabled();
  const bool use_full_visibility =
      causal_flash_attn && dense;
  const int bucket_dim_qk = HeadDimBucket(static_cast<int>(D));
  const int bucket_dim_v = HeadDimBucket(static_cast<int>(V));
  TORCH_CHECK(
      D <= 256 && V <= 256,
      "Causal attention SM90 BWD supports D,V <= 256; got D=", D, ", V=", V);
  // Use runtime dimensions whenever their static alignment permits it.
  // Sequence length does not select the padding strategy.
  const bool use_runtime_head_dims =
      use_full_visibility && !has_prev && odd_head_window_right_delta == 0 &&
      bucket_dim_qk == bucket_dim_v &&
      (bucket_dim_qk == 64 || bucket_dim_qk == 128) &&
      D % 8 == 0 && V % 8 == 0;
  const bool needs_qk_dim_pad = D != bucket_dim_qk && !use_runtime_head_dims;
  const bool needs_v_dim_pad = V != bucket_dim_v && !use_runtime_head_dims;
  if (needs_qk_dim_pad) {
    q_c = PadLastDim(q_c, bucket_dim_qk);
    k_c = PadLastDim(k_c, bucket_dim_qk);
  }
  if (needs_v_dim_pad) {
    v_c = PadLastDim(v_c, bucket_dim_v);
    dy_c = PadLastDim(dy_c, bucket_dim_v);
    y_c = PadLastDim(y_c, bucket_dim_v);
  }

  torch::Tensor dq_pad = torch::empty_like(q_c);
  const bool grouped_heads = q.size(2) != k.size(2);
  const bool native_window160 = dense && !use_full_visibility &&
      UseNativeWindow160Bwd(
          D, V, q.size(0), q.size(1), k_c.size(1), q.size(2), k.size(2),
          window_size, strict_past ? -1 : 0, odd_head_window_right_delta,
          deterministic);
  const bool use_d192 =
      dense && !native_window160 &&
      ((D == 192 && V == 128) || (D == 160 && V == 128) ||
       (D == 160 && V == 160));
  const bool use_d96 = dense && D == 64 && V == 96;
  const bool use_d256 =
      use_full_visibility && !deterministic && D == 64 && V == 256;
  // The causal-flash-attn cast can remove square-template padding while writing
  // grouped gradients directly into their actual D/V shape.
  const bool trim_grouped_grads =
      use_full_visibility && !has_prev &&
      odd_head_window_right_delta == 0 &&
      ((D == 160 && V == 160) ||
       CausalFlashAttnBwdMixedSquareDim(q_c.size(3), v_c.size(3), deterministic) != 0);
  const bool pad_grouped_k_grad =
      grouped_heads && !trim_grouped_grads &&
      ((use_d192 && D != 192) || use_d96 || use_d256);
  const bool pad_grouped_v_grad =
      grouped_heads && !trim_grouped_grads && use_d192 && V != 192;
  const int64_t grouped_k_dim = use_d256 ? 256 : (use_d192 ? 192 : 96);
  torch::Tensor dk_total_pad = pad_grouped_k_grad
      ? EmptyLastDimLike(k_c, grouped_k_dim)
      : torch::empty_like(k_c);
  torch::Tensor dv_total_pad = pad_grouped_v_grad
      ? EmptyLastDimLike(v_c, 192)
      : torch::empty_like(v_c);

  AttentionBwdParams params;
  FillBwdParams(
      dy_c, q_c, k_c, v_c, y_c, lse_c, window_size,
      static_cast<float>(scale), dq_pad, dk_total_pad, dv_total_pad,
      q_segment_idx_data, k_segment_idx_data, k_segment_len,
      strict_past ? -1 : 0, odd_head_window_right_delta, params);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kHalf) {
    if (use_full_visibility) {
      DispatchCausalFlashAttnBwd<cutlass::half_t>(params, deterministic, stream);
    } else {
      DispatchBwd<cutlass::half_t>(params, deterministic, stream);
    }
  } else {
    if (use_full_visibility) {
      DispatchCausalFlashAttnBwd<cutlass::bfloat16_t>(params, deterministic, stream);
    } else {
      DispatchBwd<cutlass::bfloat16_t>(params, deterministic, stream);
    }
  }

  torch::Tensor dq = needs_qk_dim_pad
      ? dq_pad.narrow(3, 0, D).contiguous()
      : dq_pad;
  torch::Tensor dk;
  torch::Tensor dv;
  c10::optional<torch::Tensor> dprev_k = c10::nullopt;
  c10::optional<torch::Tensor> dprev_v = c10::nullopt;
  if (has_prev) {
    torch::Tensor dprev_k_view =
        dk_total_pad.narrow(1, 0, prev_length);
    torch::Tensor dprev_v_view =
        dv_total_pad.narrow(1, 0, prev_length);
    torch::Tensor dk_view =
        dk_total_pad.narrow(1, prev_length, k.size(1));
    torch::Tensor dv_view =
        dv_total_pad.narrow(1, prev_length, k.size(1));
    if (needs_qk_dim_pad || pad_grouped_k_grad) {
      dprev_k_view = dprev_k_view.narrow(3, 0, D);
      dk_view = dk_view.narrow(3, 0, D);
    }
    if (needs_v_dim_pad || pad_grouped_v_grad) {
      dprev_v_view = dprev_v_view.narrow(3, 0, V);
      dv_view = dv_view.narrow(3, 0, V);
    }
    dprev_k = c10::make_optional(dprev_k_view.contiguous());
    dprev_v = c10::make_optional(dprev_v_view.contiguous());
    dk = dk_view.contiguous();
    dv = dv_view.contiguous();
  } else {
    dk = needs_qk_dim_pad || pad_grouped_k_grad
        ? dk_total_pad.narrow(3, 0, D).contiguous()
        : dk_total_pad;
    dv = needs_v_dim_pad || pad_grouped_v_grad
        ? dv_total_pad.narrow(3, 0, V).contiguous()
        : dv_total_pad;
  }

  return std::make_tuple<
      torch::Tensor, torch::Tensor, torch::Tensor,
      c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>(
      std::move(dq), std::move(dk), std::move(dv),
      std::move(dprev_k), std::move(dprev_v));
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
CausalAttentionSM90Bwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, bool strict_past, bool causal_flash_attn) {
  return CausalAttentionSM90BwdWithHeadBoundary(
      y_grad, q, k, v, y, lse, window_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic, strict_past,
      0 /*odd_head_window_right_delta*/, causal_flash_attn);
}

std::tuple<torch::Tensor, torch::Tensor> CausalAttentionSM90VarlenFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale, bool strict_past,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  const int arch = CurrentDeviceArch(q);
  TORCH_CHECK(
      arch == 90,
      "Causal attention SM90 varlen FWD requires Hopper sm90, got sm", arch);
  TORCH_CHECK(
      window_size >= 0 &&
          window_size <= std::numeric_limits<int>::max(),
      "Causal attention SM90 varlen FWD window_size must fit int32");
  CheckPackedQKV(q, k, v);
  const attention::sequence::VarlenMetadata metadata =
      attention::sequence::CheckVarlenMetadata(
          q, k, cu_seqlens_q, cu_seqlens_k,
          max_seqlen_q, max_seqlen_k, "Causal attention SM90");
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  torch::Tensor q_c = q.contiguous();
  torch::Tensor k_c = k.contiguous();
  torch::Tensor v_c = v.contiguous();
  torch::Tensor cu_q_c = cu_seqlens_q.contiguous();
  torch::Tensor cu_k_c = cu_seqlens_k.contiguous();
  const int* cu_k_aligned =
      cu_k_c.data_ptr<int>() + metadata.k_sequence_offset;

  const int64_t total_q = q.size(0);
  const int64_t H = q.size(1);
  const int64_t V = v.size(2);
  TORCH_CHECK(!output_fp32 || !output_state.has_value(),
              "FP32-only output and output_state are mutually exclusive");
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    TORCH_CHECK(
        state.device() == q.device() && state.scalar_type() == at::kFloat &&
            state.is_contiguous() && state.dim() == 3 &&
            state.size(0) == total_q && state.size(1) == H &&
            state.size(2) == V,
        "Causal attention SM90 varlen output_state must be contiguous float32 "
        "[total_q, Hq, V] on the Q device");
  }
  torch::Tensor y = torch::empty(
      {total_q, H, V},
      v.options().dtype(output_fp32 ? at::kFloat : v.scalar_type())
          .memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor lse = torch::empty(
      {H, total_q},
      q.options().dtype(at::kFloat).memory_format(
          at::MemoryFormat::Contiguous));

  AttentionFwdParams params;
  FillVarlenFwdParams(
      q_c, k_c, v_c, window_size, static_cast<float>(scale), y, lse,
      output_state,
      cu_q_c.data_ptr<int>(), cu_k_aligned, metadata.num_sequences,
      max_seqlen_q, max_seqlen_k, strict_past, params);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kHalf) {
    DispatchVarlenFwd<cutlass::half_t>(params, stream);
  } else {
    DispatchVarlenFwd<cutlass::bfloat16_t>(params, stream);
  }
  return std::make_tuple<torch::Tensor, torch::Tensor>(
      std::move(y), std::move(lse));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CausalAttentionSM90VarlenBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale, bool deterministic,
    bool strict_past) {
  const int arch = CurrentDeviceArch(q);
  TORCH_CHECK(
      arch == 90,
      "Causal attention SM90 varlen BWD requires Hopper sm90, got sm", arch);
  TORCH_CHECK(
      window_size >= 0 &&
          window_size <= std::numeric_limits<int>::max(),
      "Causal attention SM90 varlen BWD window_size must fit int32");
  CheckVarlenBackwardTensors(y_grad, q, k, v, y, lse);
  const attention::sequence::VarlenMetadata metadata =
      attention::sequence::CheckVarlenMetadata(
          q, k, cu_seqlens_q, cu_seqlens_k,
          max_seqlen_q, max_seqlen_k, "Causal attention SM90");
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  torch::Tensor q_c = q.contiguous();
  torch::Tensor k_c = k.contiguous();
  torch::Tensor v_c = v.contiguous();
  torch::Tensor dy_c = y_grad.contiguous();
  torch::Tensor y_c = y.contiguous();
  torch::Tensor lse_c = lse.contiguous();
  torch::Tensor cu_q_c = cu_seqlens_q.contiguous();
  torch::Tensor cu_k_c = cu_seqlens_k.contiguous();
  const int* cu_k_aligned =
      cu_k_c.data_ptr<int>() + metadata.k_sequence_offset;

  const int64_t D = q.size(2);
  const int64_t V = v.size(2);
  int bucket_dim_qk = HeadDimBucket(static_cast<int>(D));
  int bucket_dim_v = HeadDimBucket(static_cast<int>(V));
  // Short-Q grouped 192/160 needs two-component dV. The native varlen
  // tile exceeds SMEM at N128, while N64 has an incompatible shared layout.
  // D256 bucket for this unequal-length shape.
  if (q.size(0) < k.size(0) && q.size(1) != k.size(1) &&
      bucket_dim_qk == 192 && bucket_dim_v == 160) {
    bucket_dim_qk = 256;
  }
  if ((bucket_dim_qk == 160 || bucket_dim_qk == 192) &&
      bucket_dim_v == 256) {
    bucket_dim_qk = 256;
  }
  if (bucket_dim_qk == 256 && bucket_dim_v == 32) {
    bucket_dim_v = 64;
  }
  if (deterministic && bucket_dim_qk == 256) {
    bucket_dim_v = bucket_dim_v <= 128 ? 128 : 256;
  }
  TORCH_CHECK(
      D <= 256 && V <= 256,
      "Causal attention SM90 varlen BWD supports D,V <= 256; got D=",
      D, ", V=", V);
  const bool needs_qk_dim_pad = D != bucket_dim_qk;
  const bool needs_v_dim_pad = V != bucket_dim_v;
  if (needs_qk_dim_pad) {
    q_c = PadLastDim(q_c, bucket_dim_qk);
    k_c = PadLastDim(k_c, bucket_dim_qk);
  }
  if (needs_v_dim_pad) {
    v_c = PadLastDim(v_c, bucket_dim_v);
    dy_c = PadLastDim(dy_c, bucket_dim_v);
    y_c = PadLastDim(y_c, bucket_dim_v);
  }

  torch::Tensor dq_pad = torch::empty_like(q_c);
  torch::Tensor dk_pad = metadata.k_sequence_offset > 0
      ? torch::zeros_like(k_c)
      : torch::empty_like(k_c);
  torch::Tensor dv_pad = metadata.k_sequence_offset > 0
      ? torch::zeros_like(v_c)
      : torch::empty_like(v_c);
  const int64_t bwd_block_m =
      bucket_dim_qk <= 64 && bucket_dim_v <= 128 ? 128 : 64;
  const int64_t seqlen_q_padded = RoundUp(
      q.size(0) + int64_t(metadata.num_sequences) * bwd_block_m,
      bwd_block_m);

  AttentionBwdParams params;
  FillVarlenBwdParams(
      dy_c, q_c, k_c, v_c, y_c, lse_c, window_size,
      static_cast<float>(scale), dq_pad, dk_pad, dv_pad,
      cu_q_c.data_ptr<int>(), cu_k_aligned, metadata.num_sequences,
      max_seqlen_q, max_seqlen_k, seqlen_q_padded, strict_past, params);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kHalf) {
    DispatchVarlenBwd<cutlass::half_t>(
        params, deterministic, stream);
  } else {
    DispatchVarlenBwd<cutlass::bfloat16_t>(
        params, deterministic, stream);
  }

  torch::Tensor dq = needs_qk_dim_pad
      ? dq_pad.narrow(2, 0, D).contiguous()
      : dq_pad;
  torch::Tensor dk = needs_qk_dim_pad
      ? dk_pad.narrow(2, 0, D).contiguous()
      : dk_pad;
  torch::Tensor dv = needs_v_dim_pad
      ? dv_pad.narrow(2, 0, V).contiguous()
      : dv_pad;
  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
      std::move(dq), std::move(dk), std::move(dv));
}

}  // namespace hopper
}  // namespace attention
}  // namespace ops
}  // namespace xattn

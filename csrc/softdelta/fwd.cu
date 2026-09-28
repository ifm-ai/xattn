#include "softdelta/fwd.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstring>
#include <limits>

#include "softdelta/gate.h"
#include "attention/hopper/launch.h"
#include "softdelta/fwd_policy.h"

namespace xattn {
namespace ops {
namespace {

int CurrentDeviceArch(const torch::Tensor& tensor) {
  at::cuda::OptionalCUDAGuard guard(at::device_of(tensor));
  const cudaDeviceProp* properties =
      at::cuda::getCurrentDeviceProperties();
  return properties->major * 10 + properties->minor;
}

bool IsHeadDimBucket(int64_t dimension) {
  return dimension == 32 || dimension == 64 || dimension == 96 ||
      dimension == 128 || dimension == 160 || dimension == 192 ||
      dimension == 256;
}

attention::semantics::VisibilityKind SemanticVisibility(
    FlashSoftDeltaVisibility visibility) {
  if (visibility == FlashSoftDeltaVisibility::kFull) {
    return attention::semantics::VisibilityKind::kCausalFull;
  }
  if (visibility == FlashSoftDeltaVisibility::kSlidingWindow) {
    return attention::semantics::VisibilityKind::kCausalSlidingWindow;
  }
  return attention::semantics::VisibilityKind::kSlidingChunk;
}

void CheckInputTensor(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == reference.device(),
      name, " must share a device with q");
  TORCH_CHECK(
      tensor.scalar_type() == reference.scalar_type(),
      name, " must share a dtype with q");
}

void CheckInputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    int64_t span,
    double scale,
    int64_t visibility) {
  TORCH_CHECK(q.is_cuda(), "Flash SoftDelta FWD requires CUDA tensors");
  TORCH_CHECK(
      CurrentDeviceArch(q) == 90,
      "Flash SoftDelta FWD requires Hopper sm90");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf ||
          q.scalar_type() == at::kBFloat16,
      "Flash SoftDelta FWD supports only fp16 and bf16");
  TORCH_CHECK(
      q.dim() == 4 && k.dim() == 4 && v.dim() == 5 &&
          gate.dim() == 5,
      "Flash SoftDelta FWD expects 4D q/k and 5D v/gate tensors");
  CheckInputTensor(k, q, "k");
  CheckInputTensor(v, q, "v");
  CheckInputTensor(gate, q, "gate");
  TORCH_CHECK(
      q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 &&
          q.size(3) > 0,
      "Flash SoftDelta FWD q dimensions must be positive");
  TORCH_CHECK(
      q.size(2) % 2 == 0,
      "Flash SoftDelta FWD q heads must contain interleaved pairs");
  TORCH_CHECK(
      k.size(0) == q.size(0) && v.size(0) == q.size(0) &&
          k.size(1) == v.size(1) &&
          (visibility == 0 ? q.size(1) <= k.size(1) : q.size(1) == k.size(1)),
      "Flash SoftDelta FWD requires matching batches and K/V lengths; Q length "
      "must match K/V (or be shorter for full attention)");
  TORCH_CHECK(
      k.size(2) > 0 && v.size(2) == k.size(2),
      "Flash SoftDelta FWD k/v head counts must match");
  TORCH_CHECK(
      (q.size(2) / 2) % k.size(2) == 0,
      "Flash SoftDelta FWD logical q heads must be divisible by KV heads");
  TORCH_CHECK(
      k.size(3) == q.size(3),
      "Flash SoftDelta FWD q/k head dimensions must match");
  TORCH_CHECK(
      v.size(3) > 0 && v.size(4) > 0,
      "Flash SoftDelta FWD value group dimensions must be positive");
  TORCH_CHECK(
      gate.size(0) == q.size(0) && gate.size(1) == q.size(1) &&
          gate.size(2) == q.size(2) / 2 &&
          gate.size(3) == v.size(3) && gate.size(4) == 1,
      "Flash SoftDelta FWD gate shape is incompatible with q/v");
  TORCH_CHECK(
      IsHeadDimBucket(q.size(3)) &&
          IsHeadDimBucket(v.size(3) * v.size(4)),
      "Flash SoftDelta FWD requires D,V in {32,64,96,128,160,192,256}");
  TORCH_CHECK(
      std::isfinite(scale),
      "Flash SoftDelta FWD scale must be finite");
  TORCH_CHECK(
      visibility >= static_cast<int64_t>(FlashSoftDeltaVisibility::kFull) &&
          visibility <= static_cast<int64_t>(
              FlashSoftDeltaVisibility::kSlidingChunk),
      "Flash SoftDelta FWD visibility is invalid");
  const auto visibility_kind =
      static_cast<FlashSoftDeltaVisibility>(visibility);
  TORCH_CHECK(
      span >= 0 && span <= std::numeric_limits<int>::max() &&
          (visibility_kind != FlashSoftDeltaVisibility::kSlidingChunk ||
           span > 0),
      "Flash SoftDelta FWD span must be a nonnegative int32 value, and "
      "sliding-chunk span must be positive");
}

void FillParams(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    torch::Tensor& output,
    torch::Tensor* lse,
    const c10::optional<torch::Tensor>& output_state,
    int64_t span,
    float scale,
    FlashSoftDeltaVisibility visibility,
    FlashSoftDeltaFwdParams& params) {
  std::memset(&params, 0, sizeof(params));
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = output.data_ptr();
  params.gate_ptr = gate.data_ptr();
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    params.o_state_ptr = state.data_ptr<float>();
    params.o_state_batch_stride = state.stride(0);
    params.o_state_row_stride = state.stride(1);
    params.o_state_head_stride = state.stride(2);
  }
  params.softmax_lse_ptr =
      lse == nullptr ? nullptr : lse->data_ptr<float>();
  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.q_row_stride = q.stride(1);
  params.k_row_stride = k.stride(1);
  params.v_row_stride = v.stride(1);
  params.q_head_stride = q.stride(2);
  params.k_head_stride = k.stride(2);
  params.v_head_stride = v.stride(2);
  params.o_batch_stride = output.stride(0);
  params.o_row_stride = output.stride(1);
  params.o_head_stride = output.stride(2);
  params.gate_batch_stride = gate.stride(0);
  params.gate_row_stride = gate.stride(1);
  params.gate_head_stride = gate.stride(2);
  params.gate_group_stride = gate.stride(3);
  params.gate_group_dim = static_cast<int>(v.size(3) / gate.size(3));
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
  params.window_size_right = 0;
  if (visibility == FlashSoftDeltaVisibility::kSlidingChunk) {
    params.window_size_left = static_cast<int>(2 * span - 1);
    params.attention_chunk = static_cast<int>(span);
  } else {
    params.window_size_left = static_cast<int>(span);
    params.attention_chunk = 0;
  }
  const cudaDeviceProp* properties =
      at::cuda::getCurrentDeviceProperties();
  params.num_sm = properties->multiProcessorCount;
}

template <typename Element, int kHeadDim, int kHeadDimV>
void DispatchVisibility(
    AttentionFwdParams& params,
    FlashSoftDeltaVisibility visibility,
    cudaStream_t stream) {
  if (visibility == FlashSoftDeltaVisibility::kFull) {
    RunFlashSoftDeltaFwdSm90VD<
        Element, Element, kHeadDim, kHeadDimV,
        attention::semantics::CausalFullVisibility>(params, stream);
  } else if (visibility == FlashSoftDeltaVisibility::kSlidingWindow) {
    RunFlashSoftDeltaFwdSm90VD<
        Element, Element, kHeadDim, kHeadDimV,
        attention::semantics::CausalSlidingWindowVisibility>(
        params, stream);
  } else {
    RunFlashSoftDeltaFwdSm90VD<
        Element, Element, kHeadDim, kHeadDimV,
        attention::semantics::SlidingChunkVisibility>(params, stream);
  }
}

template <typename Element, int kHeadDim>
void DispatchValueDim(
    AttentionFwdParams& params,
    FlashSoftDeltaVisibility visibility,
    cudaStream_t stream) {
  if (params.dv == 32) {
    DispatchVisibility<Element, kHeadDim, 32>(
        params, visibility, stream);
  } else if (params.dv == 64) {
    DispatchVisibility<Element, kHeadDim, 64>(
        params, visibility, stream);
  } else if (params.dv == 96) {
    DispatchVisibility<Element, kHeadDim, 96>(
        params, visibility, stream);
  } else if (params.dv == 128) {
    DispatchVisibility<Element, kHeadDim, 128>(
        params, visibility, stream);
  } else if (params.dv == 160) {
    DispatchVisibility<Element, kHeadDim, 160>(
        params, visibility, stream);
  } else if (params.dv == 192) {
    DispatchVisibility<Element, kHeadDim, 192>(
        params, visibility, stream);
  } else {
    DispatchVisibility<Element, kHeadDim, 256>(
        params, visibility, stream);
  }
}

template <typename Element>
void DispatchDimensions(
    AttentionFwdParams& params,
    FlashSoftDeltaVisibility visibility,
    cudaStream_t stream) {
  if (params.d == 32) {
    DispatchValueDim<Element, 32>(params, visibility, stream);
  } else if (params.d == 64) {
    DispatchValueDim<Element, 64>(params, visibility, stream);
  } else if (params.d == 96) {
    DispatchValueDim<Element, 96>(params, visibility, stream);
  } else if (params.d == 128) {
    DispatchValueDim<Element, 128>(params, visibility, stream);
  } else if (params.d == 160) {
    DispatchValueDim<Element, 160>(params, visibility, stream);
  } else if (params.d == 192) {
    DispatchValueDim<Element, 192>(params, visibility, stream);
  } else {
    DispatchValueDim<Element, 256>(params, visibility, stream);
  }
}

bool UseOrdinaryPairTrainingFwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    int64_t span,
    FlashSoftDeltaVisibility visibility) {
  const cudaDeviceProp* properties =
      at::cuda::getCurrentDeviceProperties();
  return visibility == FlashSoftDeltaVisibility::kFull &&
      q.scalar_type() == at::kBFloat16 &&
      q.size(0) == 1 &&
      q.size(1) == 65536 &&
      q.size(2) == 16 && k.size(2) == 8 &&
      q.size(3) == 256 && v.size(3) == 256 &&
      gate.size(3) == 4 && span == q.size(1) - 1 &&
      std::strcmp(properties->name, "NVIDIA H200") == 0;
}

__global__ void InterleavePairLseKernel(
    const float* inclusive,
    const float* strict_past,
    float* pair_lse,
    int64_t vectors,
    int64_t vectors_per_head,
    int heads) {
  const int64_t vector =
      int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vector >= vectors) {
    return;
  }
  const int64_t head_vector = vector % vectors_per_head;
  const int64_t batch_head = vector / vectors_per_head;
  const int64_t batch = batch_head / heads;
  const int64_t head = batch_head - batch * heads;
  const int64_t pair_head = batch * (2 * int64_t(heads)) + 2 * head;
  const int64_t destination =
      pair_head * vectors_per_head + head_vector;
  const float4* inclusive_vectors =
      reinterpret_cast<const float4*>(inclusive);
  const float4* strict_past_vectors =
      reinterpret_cast<const float4*>(strict_past);
  float4* pair_vectors = reinterpret_cast<float4*>(pair_lse);
  pair_vectors[destination] = inclusive_vectors[vector];
  pair_vectors[destination + vectors_per_head] =
      strict_past_vectors[vector];
}

void InterleavePairLse(
    const torch::Tensor& inclusive,
    const torch::Tensor& strict_past,
    torch::Tensor& pair_lse) {
  TORCH_CHECK(
      inclusive.size(2) % 4 == 0,
      "paired-reader LSE interleave requires a length divisible by four");
  constexpr int threads = 256;
  const int64_t vectors = inclusive.numel() / 4;
  const int blocks = static_cast<int>((vectors + threads - 1) / threads);
  InterleavePairLseKernel<<<
      blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          inclusive.data_ptr<float>(), strict_past.data_ptr<float>(),
          pair_lse.data_ptr<float>(), vectors, inclusive.size(2) / 4,
          static_cast<int>(inclusive.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor FlashSoftDeltaFwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    int64_t span,
    double scale,
    int64_t visibility) {
  CheckInputs(q, k, v, gate, span, scale, visibility);
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  torch::Tensor q_c = q.contiguous();
  torch::Tensor k_c = k.contiguous();
  torch::Tensor v_c = v.contiguous().flatten(3);
  torch::Tensor gate_c = gate.contiguous();
  const int64_t batch = q.size(0);
  const int64_t length = q.size(1);
  const int64_t packed_heads = q.size(2);
  FlashSoftDeltaFwdParams params;
  const auto visibility_kind =
      static_cast<FlashSoftDeltaVisibility>(visibility);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&] {
    if (q.scalar_type() == at::kHalf) {
      DispatchDimensions<cutlass::half_t>(
          params, visibility_kind, stream);
    } else {
      DispatchDimensions<cutlass::bfloat16_t>(
          params, visibility_kind, stream);
    }
  };
  const bool writes_output_directly =
      FlashSoftDeltaUseDirectOutputFwdSm90(
          q_c.size(3), v_c.size(3), gate_c.size(3),
          SemanticVisibility(visibility_kind));
  if (writes_output_directly) {
    torch::Tensor output = torch::empty(
        {batch, length, packed_heads / 2, v.size(3), v.size(4)},
        v.options().memory_format(at::MemoryFormat::Contiguous));
    FillParams(
        q_c, k_c, v_c, gate_c, output, nullptr, c10::nullopt, span,
        static_cast<float>(scale), visibility_kind, params);
    launch();
    return output;
  }
  torch::Tensor pair_output = torch::empty(
      {batch, length, packed_heads, v_c.size(3)},
      v.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor lse = torch::empty(
      {batch, packed_heads, length},
      q.options().dtype(at::kFloat).memory_format(
          at::MemoryFormat::Contiguous));
  FillParams(
      q_c, k_c, v_c, gate_c, pair_output, &lse, c10::nullopt, span,
      static_cast<float>(scale), visibility_kind, params);
  launch();
  return FlashSoftDeltaPairGateFwd(pair_output, gate_c);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FlashSoftDeltaTrainingFwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    int64_t span,
    double scale,
    int64_t visibility,
    const c10::optional<torch::Tensor>& output_state) {
  CheckInputs(q, k, v, gate, span, scale, visibility);
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  torch::Tensor q_c = q.contiguous();
  torch::Tensor k_c = k.contiguous();
  torch::Tensor v_c = v.contiguous().flatten(3);
  torch::Tensor gate_c = gate.contiguous();
  const int64_t batch = q.size(0);
  const int64_t length = q.size(1);
  const int64_t packed_heads = q.size(2);
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    TORCH_CHECK(
        state.device() == q.device() &&
            state.scalar_type() == at::kFloat && state.is_contiguous() &&
            state.dim() == 4 && state.size(0) == batch &&
            state.size(1) == length && state.size(2) == packed_heads &&
            state.size(3) == v_c.size(3),
        "Flash SoftDelta training FWD output_state must be contiguous "
        "float32 [B, L, 2H, V] on the Q device");
  }

  torch::Tensor pair_output = torch::empty(
      {batch, length, packed_heads, v_c.size(3)},
      v.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor lse = torch::empty(
      {batch, packed_heads, length},
      q.options().dtype(at::kFloat).memory_format(
          at::MemoryFormat::Contiguous));
  const auto visibility_kind =
      static_cast<FlashSoftDeltaVisibility>(visibility);
  if (UseOrdinaryPairTrainingFwd(
          q_c, k_c, v_c, gate_c, span, visibility_kind)) {
    const int64_t logical_heads = packed_heads / 2;
    torch::Tensor inclusive_output =
        pair_output.slice(2, 0, packed_heads, 2);
    torch::Tensor strict_past_output =
        pair_output.slice(2, 1, packed_heads, 2);
    torch::Tensor inclusive_lse = torch::empty(
        {batch, logical_heads, length},
        q.options().dtype(at::kFloat).memory_format(
            at::MemoryFormat::Contiguous));
    torch::Tensor strict_past_lse = torch::empty_like(inclusive_lse);
    c10::optional<torch::Tensor> inclusive_state = c10::nullopt;
    c10::optional<torch::Tensor> strict_past_state = c10::nullopt;
    if (output_state.has_value()) {
      inclusive_state =
          output_state.value().slice(2, 0, packed_heads, 2);
      strict_past_state =
          output_state.value().slice(2, 1, packed_heads, 2);
    }
    attention::hopper::AttentionSM90FwdInto(
        q_c.slice(2, 0, packed_heads, 2), k_c, v_c, span, scale,
        c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt,
        false, inclusive_state, inclusive_output, inclusive_lse,
        visibility_kind == FlashSoftDeltaVisibility::kFull);
    attention::hopper::AttentionSM90FwdInto(
        q_c.slice(2, 1, packed_heads, 2), k_c, v_c, span, scale,
        c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt,
        true, strict_past_state, strict_past_output, strict_past_lse,
        visibility_kind == FlashSoftDeltaVisibility::kFull);
    InterleavePairLse(inclusive_lse, strict_past_lse, lse);
    inclusive_lse.reset();
    strict_past_lse.reset();
    torch::Tensor output = FlashSoftDeltaPairGateFwd(pair_output, gate_c);
    return {std::move(output), std::move(pair_output), std::move(lse)};
  }
  FlashSoftDeltaFwdParams params;
  FillParams(
      q_c, k_c, v_c, gate_c, pair_output, &lse, output_state, span,
      static_cast<float>(scale), visibility_kind, params);
  params.emit_ungated_pair = true;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kHalf) {
    DispatchDimensions<cutlass::half_t>(
        params, visibility_kind, stream);
  } else {
    DispatchDimensions<cutlass::bfloat16_t>(
        params, visibility_kind, stream);
  }
  torch::Tensor output = FlashSoftDeltaPairGateFwd(pair_output, gate_c);
  return {std::move(output), std::move(pair_output), std::move(lse)};
}


}  // namespace ops
}  // namespace xattn

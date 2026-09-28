// Author: Shicheng Wen

#include "bindings.h"
#include "softdelta/composition.h"

#include <ATen/cuda/CUDAEvent.h>
#if __has_include(<c10/cuda/CUDAEvent.h>)
#include <c10/cuda/CUDAEvent.h>
#endif
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "flash_sca/sliding_chunk_attention.h"
#include "attention/causal_attention.h"
#ifdef XATTN_HAS_SM90
#include "attention/hopper/launch.h"
#endif
#include "softdelta/gate.h"
#include "softdelta/gradient_merge.h"

namespace xattn {
namespace ops {
namespace {

using AttentionGradients = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>;

std::pair<c10::cuda::CUDAStream, c10::cuda::CUDAStream>
GetBranchStreams(const c10::cuda::CUDAStream& current) {
  thread_local std::unordered_map<
      c10::DeviceIndex,
      std::unordered_map<
          uintptr_t,
          std::pair<c10::cuda::CUDAStream, c10::cuda::CUDAStream>>> streams;
  auto& device_streams = streams[current.device_index()];
  const uintptr_t key = reinterpret_cast<uintptr_t>(current.stream());
  auto iterator = device_streams.find(key);
  if (iterator == device_streams.end()) {
    c10::cuda::CUDAStream read_stream =
        c10::cuda::getStreamFromPool(false, current.device_index());
    while (read_stream.stream() == current.stream()) {
      read_stream =
          c10::cuda::getStreamFromPool(false, current.device_index());
    }
    c10::cuda::CUDAStream correction_stream =
        c10::cuda::getStreamFromPool(false, current.device_index());
    while (correction_stream.stream() == current.stream() ||
           correction_stream.stream() == read_stream.stream()) {
      correction_stream =
          c10::cuda::getStreamFromPool(false, current.device_index());
    }
    iterator = device_streams.emplace(
        key, std::make_pair(read_stream, correction_stream)).first;
  }
  return iterator->second;
}

void RecordOnStream(
    const torch::Tensor& tensor,
    const c10::cuda::CUDAStream& stream) {
  c10::cuda::CUDACachingAllocator::recordStream(
      tensor.storage().data_ptr(), stream);
}

void RecordOnStream(
    const c10::optional<torch::Tensor>& tensor,
    const c10::cuda::CUDAStream& stream) {
  if (tensor.has_value()) {
    RecordOnStream(*tensor, stream);
  }
}

void RecordOnStream(
    const AttentionGradients& gradients,
    const c10::cuda::CUDAStream& stream) {
  RecordOnStream(std::get<0>(gradients), stream);
  RecordOnStream(std::get<1>(gradients), stream);
  RecordOnStream(std::get<2>(gradients), stream);
  RecordOnStream(std::get<3>(gradients), stream);
  RecordOnStream(std::get<4>(gradients), stream);
}

}  // namespace

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSoftDeltaBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    const torch::Tensor& read,
    const torch::Tensor& read_lse,
    const torch::Tensor& correction,
    const torch::Tensor& correction_lse,
    int64_t span,
    double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    int visibility,
    bool reset_chunk_pos_per_seq,
    bool deterministic) {
  c10::cuda::CUDAGuard device_guard(q.device());
  const c10::cuda::CUDAStream current =
      c10::cuda::getCurrentCUDAStream(q.device().index());
  const torch::Tensor read_q = q.slice(2, 0, q.size(2), 2);
  const torch::Tensor correction_q = q.slice(2, 1, q.size(2), 2);
  const torch::Tensor correction_output =
      correction.reshape(output_grad.sizes());
  const torch::Tensor output_grad_contiguous = output_grad.contiguous();
  const torch::Tensor read_grad = output_grad_contiguous.flatten(3);
  torch::Tensor correction_grad;
  torch::Tensor gate_grad;

  auto run_gate = [&]() {
    auto gate_gradients = FlashSoftDeltaGateBwd(
        output_grad_contiguous, correction_output, gate.contiguous());
    correction_grad = std::get<1>(gate_gradients).flatten(3);
    gate_grad = std::get<2>(gate_gradients);
  };

  auto run_read = [&]() -> AttentionGradients {
    if (visibility == 2) {
      return FlashSCABwd(
          read_grad, read_q, k, v, read, read_lse, span, scale,
          prev_k, prev_v, q_segment_idx, k_segment_idx, deterministic,
          "sm90", reset_chunk_pos_per_seq);
    }
    return attention::CausalAttentionBwd(
        read_grad, read_q, k, v, read, read_lse, span, scale,
        prev_k, prev_v, q_segment_idx, k_segment_idx, deterministic,
        "sm90", false /*strict_past*/, visibility == 0);
  };
  auto run_correction = [&]() -> AttentionGradients {
    if (visibility == 2) {
      return FlashSCAStrictPastBwd(
          correction_grad, correction_q, k, v, correction,
          correction_lse, span, scale, prev_k, prev_v, q_segment_idx,
          k_segment_idx, deterministic, "sm90", reset_chunk_pos_per_seq);
    }
    return attention::CausalAttentionBwd(
        correction_grad, correction_q, k, v, correction,
        correction_lse, span, scale, prev_k, prev_v, q_segment_idx,
        k_segment_idx, deterministic, "sm90", true /*strict_past*/, visibility == 0);
  };

  cudaStreamCaptureStatus capture_status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(current.stream(), &capture_status));
  AttentionGradients read_gradients;
  AttentionGradients correction_gradients;
  if (capture_status != cudaStreamCaptureStatusNone) {
    run_gate();
    read_gradients = run_read();
    correction_gradients = run_correction();
  } else {
    const auto branch_streams = GetBranchStreams(current);
    const c10::cuda::CUDAStream read_stream = branch_streams.first;
    const c10::cuda::CUDAStream correction_stream = branch_streams.second;
    // PyTorch 2.11+ event-pool API.
#if __has_include(<c10/cuda/CUDAEvent.h>)
    static c10::cuda::CUDAEventPool event_pool;
#else
    static at::cuda::EventPool event_pool;
#endif
    auto inputs_ready = event_pool.get(current.device_index());
    auto gate_ready = event_pool.get(current.device_index());
    auto read_done = event_pool.get(current.device_index());
    auto correction_done = event_pool.get(current.device_index());
    inputs_ready->record(current);
    inputs_ready->block(read_stream);
    {
      c10::cuda::CUDAStreamGuard stream_guard(read_stream);
      read_gradients = run_read();
      read_done->record(read_stream);
    }
    run_gate();
    gate_ready->record(current);
    gate_ready->block(correction_stream);
    {
      c10::cuda::CUDAStreamGuard stream_guard(correction_stream);
      correction_gradients = run_correction();
      correction_done->record(correction_stream);
    }
    read_done->block(current);
    correction_done->block(current);
    RecordOnStream(read_gradients, current);
    RecordOnStream(correction_gradients, current);
  }

  auto merged = FlashSoftDeltaMergeGradients(
      std::get<0>(read_gradients).contiguous(),
      std::get<0>(correction_gradients).contiguous(),
      std::get<1>(read_gradients).contiguous(),
      std::get<1>(correction_gradients).contiguous(),
      std::get<2>(read_gradients).contiguous(),
      std::get<2>(correction_gradients).contiguous(),
      std::get<3>(read_gradients).has_value()
          ? c10::make_optional(std::get<3>(read_gradients)->contiguous())
          : c10::nullopt,
      std::get<3>(correction_gradients).has_value()
          ? c10::make_optional(std::get<3>(correction_gradients)->contiguous())
          : c10::nullopt,
      std::get<4>(read_gradients).has_value()
          ? c10::make_optional(std::get<4>(read_gradients)->contiguous())
          : c10::nullopt,
      std::get<4>(correction_gradients).has_value()
          ? c10::make_optional(std::get<4>(correction_gradients)->contiguous())
          : c10::nullopt);
  return {
      std::get<0>(merged), std::get<1>(merged), std::get<2>(merged),
      gate_grad, std::get<3>(merged), std::get<4>(merged)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
FlashSoftDeltaPairedBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    const torch::Tensor& pair_gate_state,
    const torch::Tensor& attention_output_state,
    const torch::Tensor& pair_lse,
    int64_t span,
    double scale,
    int visibility,
    bool deterministic) {
#ifdef XATTN_HAS_SM90
  TORCH_CHECK(
      q.dim() == 4 && q.size(2) % 2 == 0,
      "paired-reader BWD expects interleaved 4D Q heads");
  TORCH_CHECK(
      pair_gate_state.dim() == 4 &&
          pair_gate_state.size(0) == q.size(0) &&
          pair_gate_state.size(1) == q.size(1) &&
          pair_gate_state.size(2) == q.size(2),
      "paired-reader BWD pair gate state must be [B, L, 2H, V]");
  TORCH_CHECK(
      attention_output_state.sizes() == pair_gate_state.sizes() &&
          (attention_output_state.scalar_type() == q.scalar_type() ||
           attention_output_state.scalar_type() == at::kFloat),
      "paired-reader BWD attention output state must match pair gate state "
      "shape and use the input dtype or float32");
  TORCH_CHECK(
      pair_lse.dim() == 3 && pair_lse.size(0) == q.size(0) &&
          pair_lse.size(1) == q.size(2) &&
          pair_lse.size(2) == q.size(1),
      "paired-reader BWD LSE must be [B, 2H, L]");
  TORCH_CHECK(
      span >= 0 && (visibility != 2 || span > 0),
      "paired-reader BWD span must be nonnegative, and sliding-chunk "
      "span must be positive");
  c10::cuda::CUDAGuard device_guard(q.device());
  const torch::Tensor output_grad_contiguous = output_grad.contiguous();
  auto gate_gradients = FlashSoftDeltaPairGateBwd(
      output_grad_contiguous, pair_gate_state.contiguous(),
      gate.contiguous());
  const torch::Tensor& pair_grad = std::get<0>(gate_gradients);
  const torch::Tensor& gate_grad = std::get<1>(gate_gradients);

  AttentionGradients attention_gradients;
  if (visibility == 2) {
    attention_gradients = FlashSCASM90BwdWithHeadBoundary(
        pair_grad, q, k, v, attention_output_state, pair_lse,
        span, scale, c10::nullopt, c10::nullopt, c10::nullopt,
        c10::nullopt, deterministic, false /*reset_chunk_pos_per_seq*/,
        false /*strict_past*/, -1 /*odd_head_window_right_delta*/);
  } else {
    attention_gradients =
        attention::hopper::CausalAttentionSM90BwdWithHeadBoundary(
            pair_grad, q, k, v, attention_output_state, pair_lse,
            span, scale, c10::nullopt, c10::nullopt, c10::nullopt,
            c10::nullopt, deterministic, false /*strict_past*/,
            -1 /*odd_head_window_right_delta*/, visibility == 0);
  }
  TORCH_CHECK(
      !std::get<3>(attention_gradients).has_value() &&
          !std::get<4>(attention_gradients).has_value(),
      "paired-reader dense BWD unexpectedly produced previous-K/V gradients");
  return {
      std::move(std::get<0>(attention_gradients)),
      std::move(std::get<1>(attention_gradients)),
      std::move(std::get<2>(attention_gradients)), gate_grad};
#else
  TORCH_CHECK(false, "paired-reader BWD requires SM90 attention support");
  return {};
#endif
}

void DefineFlashSoftDeltaCompositionOps(py::module& m) {
  m.def(
      "_flash_softdelta_bwd", &FlashSoftDeltaBwd,
      "Composed Flash SoftDelta BWD",
      py::arg("output_grad"), py::arg("q"), py::arg("k"), py::arg("v"),
      py::arg("gate"), py::arg("read"), py::arg("read_lse"),
      py::arg("correction"), py::arg("correction_lse"), py::arg("span"),
      py::arg("scale"), py::arg("prev_k") = py::none(),
      py::arg("prev_v") = py::none(),
      py::arg("q_segment_idx") = py::none(),
      py::arg("k_segment_idx") = py::none(),
      py::arg("visibility") = 0,
      py::arg("reset_chunk_pos_per_seq") = false,
      py::arg("deterministic") = false)
      .def(
          "_flash_softdelta_paired_bwd", &FlashSoftDeltaPairedBwd,
          "Interleaved paired-reader Flash SoftDelta BWD",
          py::arg("output_grad"), py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("gate"), py::arg("pair_output"),
          py::arg("attention_output_state"), py::arg("pair_lse"),
          py::arg("span"), py::arg("scale"),
          py::arg("visibility") = 0,
          py::arg("deterministic") = false);
}

}  // namespace ops
}  // namespace xattn

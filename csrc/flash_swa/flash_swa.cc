// Author: Shicheng Wen

#include "bindings.h"
#include "flash_swa/flash_swa.h"

#include "attention/causal_attention.h"

namespace xattn {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> FlashSWAFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  return attention::CausalAttentionFwd(
      q, k, v, window_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, backend, false, output_state, false, false, output_fp32);
}

std::tuple<torch::Tensor, torch::Tensor> FlashSWAStrictPastFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  return attention::CausalAttentionFwd(
      q, k, v, window_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, backend, true, output_state, false, false, output_fp32);
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSWABwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend) {
  return attention::CausalAttentionBwd(
      y_grad, q, k, v, y, lse, window_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic, backend, false);
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSWAStrictPastBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    int64_t window_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend) {
  return attention::CausalAttentionBwd(
      y_grad, q, k, v, y, lse, window_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic, backend, true);
}

std::tuple<torch::Tensor, torch::Tensor> FlashSWAVarlenFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale,
    const std::string& backend, bool strict_past,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  return attention::CausalAttentionVarlenFwd(
      q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
      window_size, scale, backend, strict_past, output_state, output_fp32);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FlashSWAVarlenBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t window_size, double scale, bool deterministic,
    const std::string& backend, bool strict_past) {
  return attention::CausalAttentionVarlenBwd(
      y_grad, q, k, v, y, lse, cu_seqlens_q, cu_seqlens_k,
      max_seqlen_q, max_seqlen_k, window_size, scale, deterministic,
      backend, strict_past);
}

void DefineFlashSWAOps(py::module& m) {
  m.def(
       "flash_swa_fwd", &FlashSWAFwd, "Segment-aware causal FlashSWA FWD",
       py::arg("q"), py::arg("k"), py::arg("v"),
       py::arg("window_size"), py::arg("scale"),
       py::arg("prev_k") = py::none(),
       py::arg("prev_v") = py::none(),
       py::arg("q_segment_idx") = py::none(),
       py::arg("k_segment_idx") = py::none(),
       py::arg("backend") = "auto",
       py::arg("output_state") = py::none(),
          py::arg("output_fp32") = false)
      .def(
          "flash_swa_bwd", &FlashSWABwd,
          "Segment-aware causal FlashSWA BWD",
          py::arg("y_grad"), py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("y"), py::arg("lse"),
          py::arg("window_size"), py::arg("scale"),
          py::arg("prev_k") = py::none(),
          py::arg("prev_v") = py::none(),
          py::arg("q_segment_idx") = py::none(),
          py::arg("k_segment_idx") = py::none(),
          py::arg("deterministic") = false,
          py::arg("backend") = "auto")
      .def(
          "_flash_swa_sm90_strict_past_fwd", &FlashSWAStrictPastFwd,
          "Strict-past FlashSWA SM90 FWD",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("window_size"), py::arg("scale"),
          py::arg("prev_k") = py::none(),
          py::arg("prev_v") = py::none(),
          py::arg("q_segment_idx") = py::none(),
          py::arg("k_segment_idx") = py::none(),
          py::arg("backend") = "auto",
          py::arg("output_state") = py::none(),
          py::arg("output_fp32") = false)
      .def(
          "_flash_swa_sm90_strict_past_bwd", &FlashSWAStrictPastBwd,
          "Strict-past FlashSWA SM90 BWD",
          py::arg("y_grad"), py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("y"), py::arg("lse"),
          py::arg("window_size"), py::arg("scale"),
          py::arg("prev_k") = py::none(),
          py::arg("prev_v") = py::none(),
          py::arg("q_segment_idx") = py::none(),
          py::arg("k_segment_idx") = py::none(),
          py::arg("deterministic") = false,
          py::arg("backend") = "auto")
      .def(
          "_flash_swa_sm90_varlen_fwd", &FlashSWAVarlenFwd,
          "Packed varlen segment-isolated causal FlashSWA FWD",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"), py::arg("max_seqlen_k"),
          py::arg("window_size"), py::arg("scale"),
          py::arg("backend") = "auto",
          py::arg("strict_past") = false,
          py::arg("output_state") = py::none(),
          py::arg("output_fp32") = false)
      .def(
          "_flash_swa_sm90_varlen_bwd", &FlashSWAVarlenBwd,
          "Packed varlen segment-isolated causal FlashSWA BWD",
          py::arg("y_grad"), py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("y"), py::arg("lse"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"), py::arg("max_seqlen_k"),
          py::arg("window_size"), py::arg("scale"),
          py::arg("deterministic") = false,
          py::arg("backend") = "auto",
          py::arg("strict_past") = false);
}

}  // namespace ops
}  // namespace xattn

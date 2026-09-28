// Author: Shicheng Wen

#include "bindings.h"
#include "causal_flash_attn/causal_flash_attn.h"

#include "attention/causal_attention.h"

namespace xattn {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> CausalFlashAttnComponentFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    const c10::optional<torch::Tensor>& output_state, bool strict_past,
    bool use_fast_reciprocal, bool output_fp32) {
  return attention::CausalAttentionFwd(
      q, k, v, 0, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, backend, strict_past, output_state,
      true /*causal_flash_attn*/, use_fast_reciprocal, output_fp32);
}


std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
CausalFlashAttnComponentBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse,
    double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend, bool strict_past) {
  return attention::CausalAttentionBwd(
      y_grad, q, k, v, y, lse, 0, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic, backend, strict_past, true /*causal_flash_attn*/);
}


std::tuple<torch::Tensor, torch::Tensor> CausalFlashAttnFwd(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    const c10::optional<torch::Tensor>& output_state,
    bool use_fast_reciprocal, bool output_fp32) {
  return CausalFlashAttnComponentFwd(
      q, k, v, scale, c10::nullopt, c10::nullopt,
      q_segment_idx, k_segment_idx, backend, output_state,
      false /*strict_past*/, use_fast_reciprocal, output_fp32);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CausalFlashAttnBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& y, const torch::Tensor& lse, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend) {
  auto gradients = CausalFlashAttnComponentBwd(
      y_grad, q, k, v, y, lse, scale, c10::nullopt, c10::nullopt,
      q_segment_idx, k_segment_idx, deterministic, backend,
      false /*strict_past*/);
  return {std::get<0>(gradients), std::get<1>(gradients),
          std::get<2>(gradients)};
}

void DefineCausalFlashAttnOps(py::module& m) {
  m.def("causal_flash_attn_fwd", &CausalFlashAttnFwd,
        "Segment-aware causal flash attention FWD",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("q_segment_idx") = py::none(),
        py::arg("k_segment_idx") = py::none(),
        py::arg("backend") = "auto",
        py::arg("output_state") = py::none(),
        py::arg("use_fast_reciprocal") = true,
        py::arg("output_fp32") = false)
      .def("causal_flash_attn_bwd", &CausalFlashAttnBwd,
           "Segment-aware causal flash attention BWD: dq, dk, dv",
           py::arg("y_grad"), py::arg("q"), py::arg("k"), py::arg("v"),
           py::arg("y"), py::arg("lse"), py::arg("scale"),
           py::arg("q_segment_idx") = py::none(),
           py::arg("k_segment_idx") = py::none(),
           py::arg("deterministic") = false, py::arg("backend") = "auto");
  // Composition readers retain history and strict-past semantics internally.
  m.def(
       "_causal_flash_attn_component_fwd", &CausalFlashAttnComponentFwd,
       py::arg("q"), py::arg("k"), py::arg("v"),
       py::arg("scale"),
       py::arg("prev_k") = py::none(),
       py::arg("prev_v") = py::none(),
       py::arg("q_segment_idx") = py::none(),
       py::arg("k_segment_idx") = py::none(),
       py::arg("backend") = "auto",
       py::arg("output_state") = py::none(),
       py::arg("strict_past") = false,
       py::arg("use_fast_reciprocal") = true,
       py::arg("output_fp32") = false)
      .def(
          "_causal_flash_attn_component_bwd", &CausalFlashAttnComponentBwd,
          py::arg("y_grad"), py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("y"), py::arg("lse"),
          py::arg("scale"),
          py::arg("prev_k") = py::none(),
          py::arg("prev_v") = py::none(),
          py::arg("q_segment_idx") = py::none(),
          py::arg("k_segment_idx") = py::none(),
          py::arg("deterministic") = false,
          py::arg("backend") = "auto",
          py::arg("strict_past") = false);
}

}  // namespace ops
}  // namespace xattn

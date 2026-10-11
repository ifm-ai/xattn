#include "attention/precision/constant_keys.h"
#include "attention/precision/deterministic.h"
#include "bindings.h"
#include "attention/precision/zero_sum_repair.h"
#include "attention/precision/preprocess.h"
#include "attention/precision/main.h"
#include "attention/precision/ordered.h"
#include "attention/precision/row_statistics.h"
#include "attention/precision/probability_repair.h"
namespace xattn::ops {
void DefineAttentionPrecisionOps(py::module& m) {
  m.def("_attention_precision_key_runs", &attention::precision::KeyEqualityRuns);
  m.def("_attention_precision_enforce_constant_keys", &attention::precision::EnforceConstantKeys);
  m.def("_attention_precision_deterministic_repair", &attention::precision::DeterministicRepair);
  m.def("_attention_precision_zero_sum_repair", &attention::precision::ZeroSumRepair);
  m.def("_attention_precision_row_statistics", &attention::precision::RowStatistics,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("left"), py::arg("right"),
        py::arg("q_segment"), py::arg("k_segment"), py::arg("scale"), py::arg("exact") = py::none());
  m.def("_attention_precision_probability_repair", &attention::precision::ProbabilityRepair);
  m.def("_attention_precision_fallback_dv", &attention::precision::FallbackDV);
  m.def("_attention_precision_ordered_backward", &attention::precision::ordered::MainBackward);
  m.def("_attention_precision_main_backward", &attention::precision::MainBackward);
  m.def("_attention_precision_center", &attention::precision::Center);
  m.def("_attention_precision_scale_do", &attention::precision::ScaleDO);
}
}  // namespace xattn::ops

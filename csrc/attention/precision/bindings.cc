#include "attention/precision/constant_keys.h"
#include "attention/precision/deterministic.h"
#include "attention/precision/deterministic_uncached.h"
#include "bindings.h"
#include "attention/precision/anchor_repair.h"
#include "attention/precision/preprocess.h"
#include "attention/precision/main.h"
#include "attention/precision/ordered_level2.h"
#include "attention/precision/row_statistics.h"
#include "attention/precision/probability_repair.h"
namespace xattn::ops {
void DefineAttentionPrecisionOps(py::module& m) {
  m.def("_attention_precision_key_runs", &attention::precision::KeyEqualityRuns);
  m.def("_attention_precision_enforce_constant_keys", &attention::precision::EnforceConstantKeys);
  m.def("_attention_precision_uncached_deterministic_backward", &attention::precision::uncached::DeterministicBackward);
  m.def("_attention_precision_deterministic_backward", &attention::precision::DeterministicBackward);
  m.def("_attention_precision_deterministic_repair", &attention::precision::DeterministicRepair);
  m.def("_attention_precision_anchor_repair", &attention::precision::AnchorRepair);
  m.def("_attention_precision_row_statistics", &attention::precision::RowStatistics);
  m.def("_attention_precision_probability_repair", &attention::precision::ProbabilityRepair);
  m.def("_attention_precision_fallback_dv", &attention::precision::FallbackDV);
  m.def("_attention_precision_ordered_level2_backward", &attention::precision::ordered_level2::MainBackward);
  m.def("_attention_precision_main_backward", &attention::precision::MainBackward);
  m.def("_attention_precision_center", &attention::precision::Center);
  m.def("_attention_precision_scale_do", &attention::precision::ScaleDO);
}
}  // namespace xattn::ops

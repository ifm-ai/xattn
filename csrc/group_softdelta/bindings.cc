#include "bindings.h"
#include "group_softdelta/gate.h"
#ifdef XATTN_HAS_SM90
#include "group_softdelta/bwd.h"
#include "group_softdelta/fwd.h"
#endif

namespace xattn {
namespace ops {

void DefineGroupSoftDeltaOps(py::module& m) {
  m.def(
       "_group_softdelta_gate_fwd", &GroupSoftDeltaGateFwd,
       "group_softdelta GQA-broadcast gate FWD",
       py::arg("read"), py::arg("neg_read"), py::arg("gate"))
      .def(
          "_group_softdelta_gate_bwd", &GroupSoftDeltaGateBwd,
          "group_softdelta GQA-broadcast gate BWD: (neg_read_grad, gate_grad)",
          py::arg("dy"), py::arg("neg_read"), py::arg("gate"))
      .def(
          "_group_softdelta_merge_gradients", &GroupSoftDeltaMergeGradients,
          "group_softdelta K/V gradient merge (in place into the read gradients)",
          py::arg("read_key_grad"), py::arg("neg_read_key_grad"),
          py::arg("dneg_q"), py::arg("read_value_grad"),
          py::arg("neg_read_value_grad"))
      .def(
          "_group_softdelta_gate_fwd_readers", &GroupSoftDeltaGateFwdReaders,
          "group_softdelta gate FWD over the fused readers",
          py::arg("readers"), py::arg("gate"))
      .def(
          "_group_softdelta_gate_bwd_readers", &GroupSoftDeltaGateBwdReaders,
          "group_softdelta gate BWD over the fused readers: (neg_read_grad, gate_grad)",
          py::arg("dy"), py::arg("readers"), py::arg("gate"));
#ifdef XATTN_HAS_SM90
  m.def(
      "_group_softdelta_fwd", &GroupSoftDeltaFwd,
      "group_softdelta fused reads FWD: (output [B, Lq, Hkv + H, V], lse [B, Hkv + H, Lq]), class-major readers",
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("neg_q"), py::arg("scale"),
      py::arg("q_segment_idx") = py::none(), py::arg("k_segment_idx") = py::none(),
      py::arg("output_state") = py::none());
  m.def(
      "_group_softdelta_fwd_varlen", &GroupSoftDeltaFwdVarlen,
      "group_softdelta packed-sequence reads FWD (varlen causal + varlen strict): _group_softdelta_fwd's readers, lse",
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("neg_q"), py::arg("cu_seqlens"),
      py::arg("max_seqlen"), py::arg("scale"));
  m.def(
      "_group_softdelta_bwd_states", &GroupSoftDeltaBwdStates,
      "group_softdelta fused BWD: (dq, dk, dv, dneg_q, gate_grad); dneg_q is empty with fold_neg_q",
      py::arg("dy"), py::arg("gate"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("neg_q"),
      py::arg("neg_read_state"), py::arg("read_state"), py::arg("neg_read_lse"), py::arg("read_lse"), py::arg("scale"),
      py::arg("q_segment_idx") = py::none(), py::arg("k_segment_idx") = py::none(),
      py::arg("gated_output") = false, py::arg("fold_neg_q") = false, py::arg("reader_groups") = -1);
  m.def(
      "_group_softdelta_bwd", &GroupSoftDeltaBwd,
      "group_softdelta fused BWD from the forward readers: (dq, dk, dv, dneg_q, gate_grad in the gate's shape)",
      py::arg("dy"), py::arg("gate"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("neg_q"),
      py::arg("readers"), py::arg("lse"), py::arg("scale"), py::arg("q_segment_idx") = py::none(),
      py::arg("k_segment_idx") = py::none(), py::arg("fold_neg_q") = false, py::arg("reader_groups") = -1);
#endif
}

}  // namespace ops
}  // namespace xattn

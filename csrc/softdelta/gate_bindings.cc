#include "bindings.h"
#include "softdelta/gate.h"

namespace xattn {
namespace ops {

void DefineFlashSoftDeltaGateOps(py::module& m) {
  m.def(
       "_flash_softdelta_gate_fwd", &FlashSoftDeltaGateFwd,
       "Flash SoftDelta gate FWD",
       py::arg("read"), py::arg("correction"), py::arg("gate"))
      .def(
          "_flash_softdelta_gate_bwd", &FlashSoftDeltaGateBwd,
          "Flash SoftDelta gate BWD",
          py::arg("output_grad"), py::arg("correction"),
          py::arg("gate"))
      .def(
          "_flash_softdelta_pair_gate_bwd", &FlashSoftDeltaPairGateBwd,
          "Flash SoftDelta interleaved pair gate BWD",
          py::arg("output_grad"), py::arg("pair_output"),
          py::arg("gate"));
}

}  // namespace ops
}  // namespace xattn

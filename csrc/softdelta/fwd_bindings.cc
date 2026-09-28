#include "bindings.h"
#include "softdelta/fwd.h"

namespace xattn {
namespace ops {

void DefineFlashSoftDeltaFwdOps(py::module& m) {
  m.def(
      "_flash_softdelta_fwd", &FlashSoftDeltaFwd,
      "Flash SoftDelta fused FWD",
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("gate"),
      py::arg("span"), py::arg("scale"), py::arg("visibility"))
      .def(
          "_flash_softdelta_training_fwd", &FlashSoftDeltaTrainingFwd,
          "Flash SoftDelta interleaved paired-reader training FWD",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("gate"),
          py::arg("span"), py::arg("scale"), py::arg("visibility"),
          py::arg("output_state") = py::none());
}

}  // namespace ops
}  // namespace xattn

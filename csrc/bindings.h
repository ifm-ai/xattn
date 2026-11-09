#pragma once

// Python bindings are compiled by the host compiler, never by NVCC.
#include <torch/extension.h>

namespace py = pybind11;

namespace xattn {
namespace ops {

void DefineCausalFlashAttnOps(py::module& m);
void DefineFlashSWAOps(py::module& m);
void DefineFlashSCAOps(py::module& m);
void DefineAttentionPrecisionOps(py::module& m);
void DefineSoftDeltaOps(py::module& m);

}  // namespace ops
}  // namespace xattn

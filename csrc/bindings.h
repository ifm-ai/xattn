#pragma once

// Python bindings are compiled by the host compiler, never by NVCC.
#include <torch/extension.h>

namespace py = pybind11;

namespace xattn {
namespace ops {

void DefineFlashSoftDeltaGateOps(py::module& m);
void DefineFlashSoftDeltaCompositionOps(py::module& m);
void DefineFlashSoftDeltaFwdOps(py::module& m);
void DefineCausalFlashAttnOps(py::module& m);
void DefineFlashSWAOps(py::module& m);
void DefineFlashSCAOps(py::module& m);

}  // namespace ops
}  // namespace xattn

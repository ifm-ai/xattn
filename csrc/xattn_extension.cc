#include "bindings.h"

#include "flash_sca/sliding_chunk_attention.h"
#include "flash_swa/flash_swa.h"
#include "causal_flash_attn/causal_flash_attn.h"

namespace xattn {

PYBIND11_MODULE(xattn_cuda, m) {
  m.doc() = "xattn CUDA extensions.";

  py::module m_ops = m.def_submodule("ops", "Submodule for custom ops.");
  ops::DefineFlashSCAOps(m_ops);
  ops::DefineFlashSWAOps(m_ops);
  ops::DefineCausalFlashAttnOps(m_ops);
  ops::DefineSoftDeltaOps(m_ops);
#ifdef XATTN_HAS_SM90
  ops::DefineAttentionPrecisionOps(m_ops);
#endif
}

}  // namespace xattn

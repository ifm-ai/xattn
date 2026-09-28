#include "bindings.h"

#include "flash_sca/sliding_chunk_attention.h"
#include "flash_swa/flash_swa.h"
#include "causal_flash_attn/causal_flash_attn.h"
#include "softdelta/composition.h"
#include "softdelta/gate.h"
#include "softdelta/gradient_merge.h"
#ifdef XATTN_HAS_SM90
#include "softdelta/fwd.h"
#endif

namespace xattn {

PYBIND11_MODULE(xattn_cuda, m) {
  m.doc() = "xattn CUDA extensions.";

  py::module m_ops = m.def_submodule("ops", "Submodule for custom ops.");
  ops::DefineFlashSCAOps(m_ops);
  ops::DefineFlashSWAOps(m_ops);
  ops::DefineCausalFlashAttnOps(m_ops);
  ops::DefineFlashSoftDeltaCompositionOps(m_ops);
  ops::DefineFlashSoftDeltaGateOps(m_ops);
#ifdef XATTN_HAS_SM90
  ops::DefineFlashSoftDeltaFwdOps(m_ops);
#endif
}

}  // namespace xattn

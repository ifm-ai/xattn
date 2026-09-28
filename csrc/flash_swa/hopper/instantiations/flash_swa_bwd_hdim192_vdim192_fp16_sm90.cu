#include "flash_swa/hopper/bwd_template.cuh"

XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE(cutlass::half_t, 192, 192, 96)

// Short-Q full attention needs a smaller tile for two-component dV.
namespace xattn::ops::flash_swa {
template void RunCausalFlashAttnBwdSm90Variant<cutlass::half_t, 192, 192, 48, 0>(
    AttentionBwdParams&, bool, cudaStream_t);
}

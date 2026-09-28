#include "flash_sca/hopper/bwd_template.cuh"

XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_CHUNK_GLOBAL(cutlass::bfloat16_t, 160, 160, 64)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_CHUNK_RESET(cutlass::bfloat16_t, 160, 160, 128)

// Native dense deterministic tile for the 256-token chunk route.
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_DET(
    cutlass::bfloat16_t, 160, 160, 128, 64, 0, 0, 0, false)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_GQA(
    cutlass::bfloat16_t, 160, 160, 128, 64, 0, 0, 0, true, true)

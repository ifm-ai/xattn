#include "flash_sca/hopper/bwd_template.cuh"

XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE(cutlass::bfloat16_t, 128, 32, 128)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_DET(
    cutlass::bfloat16_t, 128, 32, 128, 64, 4, 1, 1, false)

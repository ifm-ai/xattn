#include "flash_sca/hopper/bwd_template.cuh"

XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE(cutlass::bfloat16_t, 192, 192, 96)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_DET(
    cutlass::bfloat16_t, 192, 192, 96, 64, 1, 1, 1, false)

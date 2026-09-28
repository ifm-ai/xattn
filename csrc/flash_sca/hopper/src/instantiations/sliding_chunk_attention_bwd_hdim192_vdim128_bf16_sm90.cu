#include "flash_sca/hopper/bwd_template.cuh"

XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE(cutlass::bfloat16_t, 192, 128, 128)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_GQA(
    cutlass::bfloat16_t, 192, 128, 128, 64, 1, 1, 1, false, false)

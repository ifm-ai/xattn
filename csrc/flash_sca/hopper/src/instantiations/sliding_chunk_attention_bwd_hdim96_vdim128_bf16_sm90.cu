#include "flash_sca/hopper/bwd_template.cuh"

XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE(cutlass::bfloat16_t, 96, 128, 128)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_DET(
    cutlass::bfloat16_t, 96, 128, 128, 64, 4, 1, 1, true)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_GQA(
    cutlass::bfloat16_t, 96, 128, 128, 64, 2, 2, 2, true, false)

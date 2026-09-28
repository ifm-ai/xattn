#include "flash_sca/hopper/bwd_template.cuh"

XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE(cutlass::half_t, 192, 64, 128)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_DET(
    cutlass::half_t, 192, 64, 128, 64, 1, 1, 1, false)

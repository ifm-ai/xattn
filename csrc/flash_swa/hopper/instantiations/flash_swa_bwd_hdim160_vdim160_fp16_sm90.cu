#include "flash_swa/hopper/bwd_template.cuh"

XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_DENSE_SEGMENT(
    cutlass::half_t, 160, 160, 64)
XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_VARLEN(
    cutlass::half_t, 160, 160, 128)

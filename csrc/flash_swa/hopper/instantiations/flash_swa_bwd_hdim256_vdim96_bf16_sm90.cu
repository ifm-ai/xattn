#include "flash_swa/hopper/bwd_template.cuh"

XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_DENSE_SEGMENT(cutlass::bfloat16_t, 256, 96, 64)
XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_NONDET(cutlass::bfloat16_t, 256, 96, 128)

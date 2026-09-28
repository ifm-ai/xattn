#include "flash_swa/hopper/bwd_template.cuh"

XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_DENSE_SEGMENT(cutlass::half_t, 256, 32, 64)
XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_NONDET_DENSE_SEGMENT(cutlass::half_t, 256, 32, 128)

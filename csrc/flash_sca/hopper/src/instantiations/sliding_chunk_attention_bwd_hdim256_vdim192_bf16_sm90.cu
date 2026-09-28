#include "flash_sca/hopper/bwd_template.cuh"

XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_CHUNK_GLOBAL(cutlass::bfloat16_t, 256, 192, 64)
XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_NONDET(cutlass::bfloat16_t, 256, 192, 128)

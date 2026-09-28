#include "softdelta/fwd_dynamic_chunk_tile_instantiation.cuh"

XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_CHUNK_TILE(
    cutlass::bfloat16_t, 64, 256, 128, 96, 2)

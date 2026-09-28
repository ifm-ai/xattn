#include "softdelta/fwd_dynamic_full_chunk_instantiation.cuh"

XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_FULL_CHUNK(
    cutlass::bfloat16_t, 128, 96, 128, 128, 2,
    xattn::ops::FlashSoftDeltaKVHeadTileGrouping<2>, 128, 128, 2)

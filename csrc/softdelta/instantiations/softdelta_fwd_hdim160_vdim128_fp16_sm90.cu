#include "softdelta/fwd_dynamic_full_tile_instantiation.cuh"

XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_FULL_TILE(
    cutlass::half_t, 160, 128, 128, 128, 2,
    xattn::ops::FlashSoftDeltaKVHeadTileGrouping<2>)

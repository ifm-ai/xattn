#include "softdelta/fwd_dynamic_full_window_instantiation.cuh"

XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_FULL_WINDOW(
    cutlass::half_t, 96, 128, 128, 128, 2,
    xattn::ops::FlashSoftDeltaAllHeadTileGrouping, 128, 128, 2,
    xattn::ops::FlashSoftDeltaKVHeadTileGrouping<2>)

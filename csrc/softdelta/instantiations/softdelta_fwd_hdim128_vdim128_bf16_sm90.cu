#include "softdelta/fwd_dynamic_window_chunk_instantiation.cuh"

XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_DYNAMIC_WINDOW_CHUNK(
    cutlass::bfloat16_t, 128, 128, 128, 128, 2,
    FlashSoftDeltaKVHeadTileGrouping<2>, 128, 128, 2)

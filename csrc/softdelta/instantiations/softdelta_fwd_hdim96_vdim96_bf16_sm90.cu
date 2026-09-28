#include "softdelta/fwd_window_row_pair_tile_instantiation.cuh"

XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_WINDOW_ROW_PAIR_TILE(
    cutlass::bfloat16_t, 96, 96, 128, 128, 3)

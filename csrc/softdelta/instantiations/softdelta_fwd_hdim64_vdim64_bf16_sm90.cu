#include "softdelta/fwd_window_row_pair_dynamic_chunk_instantiation.cuh"

XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_WINDOW_ROW_PAIR_DYNAMIC_CHUNK(
    cutlass::bfloat16_t, 64, 64, 128, 128, 2)

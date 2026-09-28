#include "softdelta/fwd_single_cta_grouped_full_instantiation.cuh"

XATTN_FLASH_SOFTDELTA_FWD_SM90_INSTANTIATE_SINGLE_CTA_GROUPED_FULL(
    cutlass::half_t, 192, 192, 128, 112, true,
    FlashSoftDeltaAllHeadTileGrouping, 2)

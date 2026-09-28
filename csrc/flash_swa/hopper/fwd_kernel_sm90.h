#pragma once

#include "attention/hopper/fwd_kernel_sm90.h"

namespace flash {

template <
    class CollectiveMainloop, class CollectiveEpilogue,
    class TileScheduler>
class FlashSWAFwdSm90
    : public AttentionFwdSm90<
          CollectiveMainloop, CollectiveEpilogue, TileScheduler> {};

}  // namespace flash

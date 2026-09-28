#pragma once

#include "attention/hopper/bwd_kernel_sm90.h"

namespace flash {

template <
    class CollectiveMainloop, class CollectiveEpilogue,
    class TileScheduler>
class FlashSWABwdSm90
    : public AttentionBwdSm90<
          CollectiveMainloop, CollectiveEpilogue, TileScheduler> {};

}  // namespace flash

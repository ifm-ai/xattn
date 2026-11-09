#pragma once

#include "attention/hopper/bwd_kernel_sm90.h"

namespace flash {

// Backward over logical heads l = g * R + r: one negative-query head (r = 0) and
// R - 1 query heads per KV head g (see CollectiveMainloopBwdSm90::NegQReaders).
template <
    class CollectiveMainloop, class CollectiveEpilogue,
    class TileScheduler>
class SoftDeltaBwdSm90
    : public AttentionBwdSm90<
          CollectiveMainloop, CollectiveEpilogue, TileScheduler> {};

}  // namespace flash

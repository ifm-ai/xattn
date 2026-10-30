#pragma once

#include "attention/hopper/fwd_kernel_sm90.h"

namespace flash {

// Forward over logical heads l = g * R + r: one negative-query head (r = 0) and
// R - 1 query heads per KV head g (see CollectiveMainloopFwdSm90::NegQReaders).
template <
    class CollectiveMainloop, class CollectiveEpilogue,
    class TileScheduler>
class SoftDeltaFwdSm90
    : public AttentionFwdSm90<
          CollectiveMainloop, CollectiveEpilogue, TileScheduler> {};

}  // namespace flash

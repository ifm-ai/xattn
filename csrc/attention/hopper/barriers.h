#pragma once

#include "cuda/sync/named_barrier.hpp"

namespace flash {

enum class AttentionFwdNamedBarrier {
    QueryEmpty = 0,
    WarpSchedulerWG1 = 1,
    WarpSchedulerWG2 = 2,
    WarpSchedulerWG3 = 3,
    AppendKV = 4,
    QueryRotated = 5,
    PFull = 6,
    PEmpty = 7,
};

enum class AttentionBwdNamedBarrier {
    KVEmpty = 0,
    PdS = 1,
    dQEmptyWG1 = 2,
    dQEmptyWG2 = 3,
    dQEmptyWG3 = 4,
    dQFullWG1 = 5,
    dQFullWG2 = 6,
    dQFullWG3 = 7,
};

}  // namespace flash

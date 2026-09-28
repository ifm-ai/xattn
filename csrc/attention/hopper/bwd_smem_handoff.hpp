#pragma once

#include <cstdint>

#include <cutlass/cutlass.h>

#include "cuda/sync/named_barrier.hpp"

namespace flash {

enum class AttentionBwdHandoffNamedBarrier : uint32_t {
    DQTail = 0,
};

template <class CollectiveMainloop, class CollectiveEpilogue, uint32_t NumMmaThreads>
struct AttentionBwdSmemHandoff {
    static constexpr bool Required =
        CollectiveMainloop::dQacc_use_TMA &&
        CollectiveEpilogue::NeedsDQTailBarrier;

    CUTLASS_DEVICE
    static void sync() {
        if constexpr (Required) {
            // GQA/MQA epilogue storage aliases the mainloop dQ accumulator.
            // Participants: producer warp 1 and all MMA threads, once per
            // valid scheduler work item.
            flash::named_barrier_sync(
                NumMmaThreads + cutlass::NumThreadsPerWarp,
                static_cast<uint32_t>(
                    AttentionBwdHandoffNamedBarrier::DQTail));
        }
    }
};

} // namespace flash

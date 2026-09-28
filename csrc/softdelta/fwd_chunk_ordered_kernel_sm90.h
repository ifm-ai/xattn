#pragma once

#include "attention/hopper/tile_scheduler.hpp"
#include "softdelta/fwd_kernel_sm90.h"

namespace flash {

template <
    class CollectiveMainloop, class CollectiveEpilogue,
    class TileScheduler>
class FlashSoftDeltaChunkOrderedFwdSm90
    : public AttentionFwdSm90<
          CollectiveMainloop, CollectiveEpilogue, TileScheduler> {
 public:
  using Base = AttentionFwdSm90<
      CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Arguments = typename Base::Arguments;
  using Params = typename Base::Params;

  static Params to_underlying_arguments(Arguments const& args) {
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      sm_count = cutlass::KernelHardwareInfo::
          query_device_multiprocessor_count(args.hw_info.device_id);
    }
    cutlass::KernelHardwareInfo hw_info{
        args.hw_info.device_id, sm_count};
    return {
        CollectiveMainloop::to_underlying_arguments(args.mainloop),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue),
        hw_info,
        TileScheduler::to_underlying_arguments(
            args.scheduler, args.mainloop.attention_chunk, sm_count)};
  }
};

}  // namespace flash

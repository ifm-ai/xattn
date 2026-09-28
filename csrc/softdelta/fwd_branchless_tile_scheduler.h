#pragma once

#include "softdelta/fwd_tile_scheduler.h"

namespace xattn {
namespace ops {

template <
    int NumMmaThreads, int NumProducerThreads,
    typename HeadGrouping>
class FlashSoftDeltaSingleCTABranchlessTileScheduler {
 public:
  using SharedStorage = int;
  using Mapping = detail::FlashSoftDeltaPersistentTileMapping;
  static constexpr bool HasMBlockRange = false;
  static constexpr bool RequiresProducerWarp1 = false;
  static constexpr int NumThreads =
      NumMmaThreads + NumProducerThreads;

  struct Params {
    typename Mapping::Params mapping;
    int* tile_count_semaphore;
  };

  static Params to_underlying_arguments(
      flash::TileSchedulerArguments const& args) {
    assert(args.tile_count_semaphore != nullptr);
    return {
        HeadGrouping::template Make<Mapping>(args),
        args.tile_count_semaphore};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
    const int blocks = params.mapping.total_blocks < num_sm
        ? params.mapping.total_blocks
        : num_sm;
    return {static_cast<uint32_t>(blocks)};
  }

  struct WorkTileInfo {
    int tile_idx;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.mapping.total_blocks;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      const auto coord = Mapping::GetBlockCoord(
          params.mapping, tile_idx);
      return {
          cute::get<0>(coord), 2 * cute::get<1>(coord),
          cute::get<2>(coord), 0};
    }
  };

  CUTLASS_DEVICE
  explicit FlashSoftDeltaSingleCTABranchlessTileScheduler(
      SharedStorage* tile_count_smem)
      : tile_count_smem_(tile_count_smem) {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const&) const {
    return {int(blockIdx.x)};
  }

  CUTLASS_DEVICE
  void init_consumer() const {
    flash::named_barrier_arrive(
        NumThreads,
        cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
  }

  CUTLASS_DEVICE
  void prefetch_next_work(
      Params const& params, WorkTileInfo& current_work) const {
    if (threadIdx.x % NumProducerThreads == 0) {
      const int next_tile =
          atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
      current_work.tile_idx = max(next_tile, int(gridDim.x));
    }
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_next_work(
      Params const&, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      const int new_tile_idx = __shfl_sync(
          0xffffffff, current_work.tile_idx, 0);
      flash::named_barrier_sync(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
      if (threadIdx.x % NumProducerThreads == 0) {
        *tile_count_smem_ = current_work.tile_idx;
      }
      flash::named_barrier_arrive(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      return {new_tile_idx};
    } else {
      flash::named_barrier_sync(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      const int tile_idx = *tile_count_smem_;
      flash::named_barrier_arrive(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
      return {tile_idx};
    }
  }

 private:
  SharedStorage* const tile_count_smem_;
};

}  // namespace ops
}  // namespace xattn

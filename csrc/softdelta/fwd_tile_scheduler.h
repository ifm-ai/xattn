#pragma once

#include <cassert>
#include <cuda_runtime_api.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "attention/hopper/tile_scheduler.hpp"

namespace xattn {
namespace ops {

namespace detail {

struct FlashSoftDeltaPersistentTileMapping {
  struct Params {
    int const total_blocks;
    cutlass::FastDivmod const m_block_divmod;
    cutlass::FastDivmod const head_divmod;
    cutlass::FastDivmod const l2_minor_divmod;
    cutlass::FastDivmod const l2_major_divmod;
    cutlass::FastDivmod const l2_minor_residual_divmod;
    int const num_hb_quotient;
  };

  static Params MakeForHeadGroup(
      flash::TileSchedulerArguments const& args,
      int logical_heads_per_group) {
    return MakeForHeadGroup(
        args, logical_heads_per_group, args.num_blocks);
  }

  static Params MakeForHeadGroup(
      flash::TileSchedulerArguments const& args,
      int logical_heads_per_group,
      int num_blocks) {
    assert(args.qhead_per_khead % 2 == 0);
    assert(logical_heads_per_group > 0);
    const int swizzle = logical_heads_per_group;
    const int num_hb = args.num_head * args.num_batch;
    const int num_hb_remainder = num_hb % swizzle;
    return {
        num_blocks * num_hb,
        cutlass::FastDivmod(num_blocks),
        cutlass::FastDivmod(args.num_head),
        cutlass::FastDivmod(swizzle),
        cutlass::FastDivmod(swizzle * num_blocks),
        cutlass::FastDivmod(
            num_hb_remainder > 0 ? num_hb_remainder : 1),
        num_hb / swizzle,
    };
  }

  static Params MakeForKVHeadGroup(
      flash::TileSchedulerArguments const& args,
      int kv_heads_per_group) {
    return MakeForKVHeadGroup(
        args, kv_heads_per_group, args.num_blocks);
  }

  static Params MakeForKVHeadGroup(
      flash::TileSchedulerArguments const& args,
      int kv_heads_per_group,
      int num_blocks) {
    assert(kv_heads_per_group > 0);
    return MakeForHeadGroup(
        args, kv_heads_per_group * args.qhead_per_khead / 2,
        num_blocks);
  }

  static Params MakeForAllHeads(
      flash::TileSchedulerArguments const& args) {
    return MakeForHeadGroup(args, args.num_head);
  }

  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t, int32_t>
  GetBlockCoord(Params const& params, int tile_idx) {
    int l2_mod;
    int logical_hb;
    int logical_hb_residual;
    logical_hb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
    int block;
    if (logical_hb < params.num_hb_quotient) {
      block = params.l2_minor_divmod.divmod(
          logical_hb_residual, l2_mod);
    } else {
      block = params.l2_minor_residual_divmod.divmod(
          logical_hb_residual, l2_mod);
    }
    int logical_head;
    const int batch = params.head_divmod.divmod(
        logical_head,
        logical_hb * params.l2_minor_divmod.divisor +
            logical_hb_residual);
    block = params.m_block_divmod.divisor - 1 - block;
    return {block, logical_head, batch};
  }

  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t, int32_t>
  GetWaveTailBlockCoord(
      Params const& params, int tile_idx, int resident_blocks) {
    const int num_blocks = params.m_block_divmod.divisor;
    const int num_hb = params.total_blocks / num_blocks;
    const int complete_blocks =
        num_blocks / resident_blocks * resident_blocks;
    const int complete_tiles = complete_blocks * num_hb;
    int block;
    int logical_hb;
    if (tile_idx < complete_tiles) {
      logical_hb = tile_idx / complete_blocks;
      block = num_blocks - 1 - tile_idx % complete_blocks;
    } else {
      const int tail_idx = tile_idx - complete_tiles;
      const int tail_blocks = num_blocks - complete_blocks;
      logical_hb = tail_idx % num_hb;
      block = tail_blocks - 1 - tail_idx / num_hb;
    }
    int logical_head;
    const int batch = params.head_divmod.divmod(
        logical_head, logical_hb);
    return {block, logical_head, batch};
  }

  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t, int32_t>
  GetGroupedWaveTailBlockCoord(
      Params const& params, int tile_idx, int resident_blocks) {
    const int num_blocks = params.m_block_divmod.divisor;
    const int num_hb = params.total_blocks / num_blocks;
    const int group_size = params.l2_minor_divmod.divisor;
    if (num_hb % group_size != 0) {
      return GetWaveTailBlockCoord(
          params, tile_idx, resident_blocks);
    }
    const int blocks_per_wave = resident_blocks / group_size > 0
        ? resident_blocks / group_size
        : 1;
    const int complete_blocks =
        num_blocks / blocks_per_wave * blocks_per_wave;
    const int tiles_per_group = complete_blocks * group_size;
    const int complete_tiles =
        tiles_per_group * (num_hb / group_size);
    int block;
    int logical_hb;
    if (tile_idx < complete_tiles) {
      const int group = tile_idx / tiles_per_group;
      const int group_tile = tile_idx % tiles_per_group;
      logical_hb = group * group_size + group_tile % group_size;
      block = num_blocks - 1 - group_tile / group_size;
    } else {
      const int tail_idx = tile_idx - complete_tiles;
      const int tail_blocks = num_blocks - complete_blocks;
      logical_hb = tail_idx % num_hb;
      block = tail_blocks - 1 - tail_idx / num_hb;
    }
    int logical_head;
    const int batch = params.head_divmod.divmod(
        logical_head, logical_hb);
    return {block, logical_head, batch};
  }

};

}  // namespace detail

struct FlashSoftDeltaAllHeadTileGrouping {
  template <typename Mapping>
  static typename Mapping::Params Make(
      flash::TileSchedulerArguments const& args) {
    return Mapping::MakeForAllHeads(args);
  }

  template <typename Mapping>
  static typename Mapping::Params Make(
      flash::TileSchedulerArguments const& args,
      int num_blocks) {
    return Mapping::MakeForHeadGroup(
        args, args.num_head, num_blocks);
  }
};

struct FlashSoftDeltaL2TileOrder {
  template <typename Mapping>
  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t, int32_t>
  GetBlockCoord(
      typename Mapping::Params const& params,
      int tile_idx, int) {
    return Mapping::GetBlockCoord(params, tile_idx);
  }
};

struct FlashSoftDeltaWaveTailTileOrder {
  template <typename Mapping>
  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t, int32_t>
  GetBlockCoord(
      typename Mapping::Params const& params,
      int tile_idx, int resident_blocks) {
    return Mapping::GetWaveTailBlockCoord(
        params, tile_idx, resident_blocks);
  }
};

struct FlashSoftDeltaGroupedWaveTailTileOrder {
  template <typename Mapping>
  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t, int32_t>
  GetBlockCoord(
      typename Mapping::Params const& params,
      int tile_idx, int resident_blocks) {
    return Mapping::GetGroupedWaveTailBlockCoord(
        params, tile_idx, resident_blocks);
  }
};

template <int KVHeadsPerGroup>
struct FlashSoftDeltaKVHeadTileGrouping {
  static_assert(KVHeadsPerGroup > 0);

  template <typename Mapping>
  static typename Mapping::Params Make(
      flash::TileSchedulerArguments const& args) {
    return Mapping::MakeForKVHeadGroup(args, KVHeadsPerGroup);
  }

  template <typename Mapping>
  static typename Mapping::Params Make(
      flash::TileSchedulerArguments const& args,
      int num_blocks) {
    return Mapping::MakeForKVHeadGroup(
        args, KVHeadsPerGroup, num_blocks);
  }
};

class FlashSoftDeltaPairPersistentTileScheduler {
 public:
  using SharedStorage = int;
  using Mapping = detail::FlashSoftDeltaPersistentTileMapping;
  using Params = Mapping::Params;
  static constexpr bool HasMBlockRange = false;
  static constexpr bool RequiresProducerWarp1 = false;

  static Params to_underlying_arguments(
      flash::TileSchedulerArguments const& args) {
    return Mapping::MakeForAllHeads(args);
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
    const int max_clusters = num_sm > 1 ? num_sm / 2 : 1;
    const int clusters = params.total_blocks < max_clusters
        ? params.total_blocks
        : max_clusters;
    return {static_cast<uint32_t>(2 * clusters)};
  }

  struct WorkTileInfo {
    int tile_idx;
    int wave;
    int branch;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      const auto coord = Mapping::GetBlockCoord(params, tile_idx);
      return {
          cute::get<0>(coord), 2 * cute::get<1>(coord) + branch,
          cute::get<2>(coord), 0};
    }
  };

  CUTLASS_DEVICE
  explicit FlashSoftDeltaPairPersistentTileScheduler(SharedStorage*) {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const&) const {
    return {int(blockIdx.x) / 2, 0, int(blockIdx.x) & 1};
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const&, WorkTileInfo&) const {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_next_work(
      Params const&, WorkTileInfo const& current_work) const {
    const int cluster_count = int(gridDim.x) / 2;
    const int cluster_idx = int(blockIdx.x) / 2;
    const int wave = current_work.wave + 1;
    const int wave_cluster_idx = wave & 1
        ? cluster_count - 1 - cluster_idx
        : cluster_idx;
    return {
        wave * cluster_count + wave_cluster_idx,
        wave,
        current_work.branch,
    };
  }
};

class FlashSoftDeltaSingleCTAPersistentTileScheduler {
 public:
  using SharedStorage = int;
  using Mapping = detail::FlashSoftDeltaPersistentTileMapping;
  using Params = Mapping::Params;
  static constexpr bool HasMBlockRange = false;
  static constexpr bool RequiresProducerWarp1 = false;

  static Params to_underlying_arguments(
      flash::TileSchedulerArguments const& args) {
    return Mapping::MakeForAllHeads(args);
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
    const int blocks = params.total_blocks < num_sm
        ? params.total_blocks
        : num_sm;
    return {static_cast<uint32_t>(blocks)};
  }

  struct WorkTileInfo {
    int tile_idx;
    int wave;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      const auto coord = Mapping::GetBlockCoord(params, tile_idx);
      return {
          cute::get<0>(coord), 2 * cute::get<1>(coord),
          cute::get<2>(coord), 0};
    }
  };

  CUTLASS_DEVICE
  explicit FlashSoftDeltaSingleCTAPersistentTileScheduler(
      SharedStorage*) {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const&) const {
    return {int(blockIdx.x), 0};
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const&, WorkTileInfo&) const {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_next_work(
      Params const&, WorkTileInfo const& current_work) const {
    const int block_count = int(gridDim.x);
    const int block_idx = int(blockIdx.x);
    const int wave = current_work.wave + 1;
    const int wave_block_idx = wave & 1
        ? block_count - 1 - block_idx
        : block_idx;
    return {wave * block_count + wave_block_idx, wave};
  }
};

template <
    int NumMmaThreads, int NumProducerThreads,
    typename HeadGrouping>
class FlashSoftDeltaClusteredRowPairL2TileScheduler {
 public:
  using Mapping = detail::FlashSoftDeltaPersistentTileMapping;
  static constexpr bool HasMBlockRange = false;
  static constexpr bool RequiresProducerWarp1 = false;
  static constexpr int NumThreads =
      NumMmaThreads + NumProducerThreads;

  struct SharedStorage {
    alignas(16) cutlass::arch::ClusterBarrier tile_ready;
    int tile_idx;
  };

  struct Params {
    typename Mapping::Params mapping;
    int num_blocks;
    int* tile_count_semaphore;
  };

  static Params to_underlying_arguments(
      flash::TileSchedulerArguments const& args) {
    assert(args.tile_count_semaphore != nullptr);
    const int pair_blocks = (args.num_blocks + 1) / 2;
    return {
        HeadGrouping::template Make<Mapping>(args, pair_blocks),
        args.num_blocks,
        args.tile_count_semaphore};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
    const int max_clusters = num_sm > 1 ? num_sm / 2 : 1;
    const int clusters = params.mapping.total_blocks < max_clusters
        ? params.mapping.total_blocks
        : max_clusters;
    return {static_cast<uint32_t>(2 * clusters)};
  }

  struct WorkTileInfo {
    int tile_idx;
    int rank;
    int generation;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.mapping.total_blocks;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      const auto coord = Mapping::GetBlockCoord(
          params.mapping, tile_idx);
      const int first_block = 2 * cute::get<0>(coord);
      const int last_block = first_block + 1 < params.num_blocks
          ? first_block + 1
          : first_block;
      return {
          rank == 0 ? last_block : first_block,
          2 * cute::get<1>(coord), cute::get<2>(coord), 0};
    }
  };

  CUTLASS_DEVICE
  explicit FlashSoftDeltaClusteredRowPairL2TileScheduler(
      SharedStorage* storage)
      : storage_(storage) {
    if (threadIdx.x == 0) {
      storage_->tile_ready.init(2);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    cute::cluster_sync();
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const&) const {
    return {
        int(blockIdx.x) / 2,
        int(blockIdx.x) & 1,
        0};
  }

  CUTLASS_DEVICE
  void init_consumer() const {
    flash::named_barrier_arrive(
        NumThreads,
        cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
  }

  CUTLASS_DEVICE
  void prefetch_next_work(
      Params const&, WorkTileInfo&) const {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_next_work(
      Params const& params, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      int next_tile_idx = current_work.tile_idx;
      const int cluster_count = int(gridDim.x) / 2;
      const int next_generation = current_work.generation + 1;
      if (threadIdx.x % NumProducerThreads == 0) {
        if (current_work.rank == 0) {
          next_tile_idx =
              atomicAdd(params.tile_count_semaphore, 1) +
              cluster_count;
          storage_->tile_idx = next_tile_idx;
          cutlass::arch::fence_view_async_shared();
        }
#pragma unroll
        for (uint32_t rank = 0; rank < 2; ++rank) {
          storage_->tile_ready.arrive(rank);
        }
        storage_->tile_ready.wait(current_work.generation & 1);
        if (current_work.rank != 0) {
          const uint32_t local_address =
              __cvta_generic_to_shared(&storage_->tile_idx);
          const uint32_t remote_address =
              cute::set_block_rank(local_address, 0);
          asm volatile(
              "ld.shared::cluster.b32 %0, [%1];"
              : "=r"(next_tile_idx)
              : "r"(remote_address)
              : "memory");
        }
      }
      const int new_tile_idx = __shfl_sync(
          0xffffffff, next_tile_idx, 0);
      flash::named_barrier_sync(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
      if (threadIdx.x % NumProducerThreads == 0) {
        storage_->tile_idx = new_tile_idx;
      }
      flash::named_barrier_arrive(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      return {
          new_tile_idx, current_work.rank,
          next_generation};
    } else {
      flash::named_barrier_sync(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      const int tile_idx = storage_->tile_idx;
      flash::named_barrier_arrive(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
      return {
          tile_idx, current_work.rank,
          current_work.generation + 1};
    }
  }

 private:
  SharedStorage* const storage_;
};

template <
    int NumMmaThreads, int NumProducerThreads,
    typename HeadGrouping,
    typename TileOrder = FlashSoftDeltaL2TileOrder>
class FlashSoftDeltaSingleCTADynamicTileScheduler {
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
      const auto coord = TileOrder::template GetBlockCoord<Mapping>(
          params.mapping, tile_idx, int(gridDim.x));
      return {
          cute::get<0>(coord), 2 * cute::get<1>(coord),
          cute::get<2>(coord), 0};
    }
  };

  CUTLASS_DEVICE
  explicit FlashSoftDeltaSingleCTADynamicTileScheduler(
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
      current_work.tile_idx =
          atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
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

template <
    int NumMmaThreads, int NumProducerThreads,
    typename HeadGrouping,
    typename TileOrder = FlashSoftDeltaL2TileOrder>
class FlashSoftDeltaClusteredRowPairTileScheduler {
 public:
  using Mapping = detail::FlashSoftDeltaPersistentTileMapping;
  static constexpr bool HasMBlockRange = false;
  static constexpr bool RequiresProducerWarp1 = false;
  static constexpr int NumThreads =
      NumMmaThreads + NumProducerThreads;
  static constexpr bool HasEpilogueBlockCoord = true;

  struct SharedStorage {
    alignas(16) cutlass::arch::ClusterBarrier tile_ready;
    int tile_idx;
  };

  struct Params {
    typename Mapping::Params mapping;
    int num_blocks;
    int* tile_count_semaphore;
  };

  static Params to_underlying_arguments(
      flash::TileSchedulerArguments const& args) {
    assert(args.tile_count_semaphore != nullptr);
    const int pair_blocks = (args.num_blocks + 1) / 2;
    return {
        HeadGrouping::template Make<Mapping>(args, pair_blocks),
        args.num_blocks,
        args.tile_count_semaphore};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
    const int max_clusters = num_sm > 1 ? num_sm / 2 : 1;
    const int clusters = params.mapping.total_blocks < max_clusters
        ? params.mapping.total_blocks
        : max_clusters;
    return {static_cast<uint32_t>(2 * clusters)};
  }

  struct WorkTileInfo {
    int tile_idx;
    int rank;
    int generation;
    int epilogue_tile_idx;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.mapping.total_blocks;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      const auto coord = TileOrder::template GetBlockCoord<Mapping>(
          params.mapping, tile_idx, int(gridDim.x) / 2);
      const int first_block = 2 * cute::get<0>(coord);
      const int last_block = first_block + 1 < params.num_blocks
          ? first_block + 1
          : first_block;
      return {
          rank == 0 ? last_block : first_block,
          2 * cute::get<1>(coord), cute::get<2>(coord), 0};
    }
  };

  CUTLASS_DEVICE
  explicit FlashSoftDeltaClusteredRowPairTileScheduler(
      SharedStorage* storage)
      : storage_(storage) {
    if (threadIdx.x == 0) {
      storage_->tile_ready.init(2);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    cute::cluster_sync();
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const&) const {
    return {
        int(blockIdx.x) / 2,
        int(blockIdx.x) & 1,
        0,
        int(blockIdx.x) / 2};
  }

  CUTLASS_DEVICE
  void init_consumer() const {
    flash::named_barrier_arrive(
        NumThreads,
        cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
  }

  CUTLASS_DEVICE
  void prefetch_next_work(
      Params const&, WorkTileInfo&) const {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_next_work(
      Params const& params, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      int next_tile_idx = current_work.tile_idx;
      const int cluster_count = int(gridDim.x) / 2;
      const int next_generation = current_work.generation + 1;
      if (threadIdx.x % NumProducerThreads == 0) {
        if (current_work.rank == 0) {
          next_tile_idx =
              atomicAdd(params.tile_count_semaphore, 1) +
              cluster_count;
          storage_->tile_idx = next_tile_idx;
          cutlass::arch::fence_view_async_shared();
        }
#pragma unroll
        for (uint32_t rank = 0; rank < 2; ++rank) {
          storage_->tile_ready.arrive(rank);
        }
        storage_->tile_ready.wait(current_work.generation & 1);
        if (current_work.rank != 0) {
          const uint32_t local_address =
              __cvta_generic_to_shared(&storage_->tile_idx);
          const uint32_t remote_address =
              cute::set_block_rank(local_address, 0);
          asm volatile(
              "ld.shared::cluster.b32 %0, [%1];"
              : "=r"(next_tile_idx)
              : "r"(remote_address)
              : "memory");
        }
      }
      const int new_tile_idx = __shfl_sync(
          0xffffffff, next_tile_idx, 0);
      flash::named_barrier_sync(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
      if (threadIdx.x % NumProducerThreads == 0) {
        storage_->tile_idx = new_tile_idx;
      }
      flash::named_barrier_arrive(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      return {
          new_tile_idx, current_work.rank,
          next_generation, current_work.tile_idx};
    } else {
      flash::named_barrier_sync(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      const int tile_idx = storage_->tile_idx;
      flash::named_barrier_arrive(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
      return {
          tile_idx, current_work.rank,
          current_work.generation + 1,
          current_work.tile_idx};
    }
  }

  CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t, int32_t, int32_t>
  get_epilogue_block_coord(
      Params const& params, WorkTileInfo const& current_work) const {
    return WorkTileInfo{
        current_work.epilogue_tile_idx,
        current_work.rank,
        current_work.generation - 1,
        current_work.epilogue_tile_idx}
        .get_block_coord(params);
  }

 private:
  SharedStorage* const storage_;
};

template <int NumMmaThreads, int NumProducerThreads>
class FlashSoftDeltaPairDynamicTileScheduler {
 public:
  using Mapping = detail::FlashSoftDeltaPersistentTileMapping;
  static constexpr bool HasMBlockRange = false;
  static constexpr bool RequiresProducerWarp1 = false;
  static constexpr int NumThreads =
      NumMmaThreads + NumProducerThreads;

  struct SharedStorage {
    alignas(16) cutlass::arch::ClusterBarrier tile_ready;
    int tile_idx;
  };

  struct Params {
    typename Mapping::Params mapping;
    int* tile_count_semaphore;
  };

  static Params to_underlying_arguments(
      flash::TileSchedulerArguments const& args) {
    assert(args.tile_count_semaphore != nullptr);
    return {
        Mapping::MakeForAllHeads(args),
        args.tile_count_semaphore};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
    const int max_clusters = num_sm > 1 ? num_sm / 2 : 1;
    const int clusters = params.mapping.total_blocks < max_clusters
        ? params.mapping.total_blocks
        : max_clusters;
    return {static_cast<uint32_t>(2 * clusters)};
  }

  struct WorkTileInfo {
    int tile_idx;
    int branch;
    int generation;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.mapping.total_blocks;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      const int cluster_count = int(gridDim.x) / 2;
      const auto coord = Mapping::GetWaveTailBlockCoord(
          params.mapping, tile_idx, cluster_count);
      return {
          cute::get<0>(coord),
          2 * cute::get<1>(coord) + branch,
          cute::get<2>(coord),
          0};
    }
  };

  CUTLASS_DEVICE
  explicit FlashSoftDeltaPairDynamicTileScheduler(
      SharedStorage* storage)
      : storage_(storage) {
    if (threadIdx.x == 0) {
      storage_->tile_ready.init(2);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    cute::cluster_sync();
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const&) const {
    return {
        int(blockIdx.x) / 2,
        int(blockIdx.x) & 1,
        0};
  }

  CUTLASS_DEVICE
  void init_consumer() const {
    flash::named_barrier_arrive(
        NumThreads,
        cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
  }

  CUTLASS_DEVICE
  void prefetch_next_work(
      Params const&, WorkTileInfo&) const {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_next_work(
      Params const& params, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      int next_tile_idx = current_work.tile_idx;
      const int cluster_count = int(gridDim.x) / 2;
      const int next_generation = current_work.generation + 1;
      if (threadIdx.x % NumProducerThreads == 0) {
        if (current_work.branch == 0) {
          next_tile_idx =
              atomicAdd(params.tile_count_semaphore, 1) +
              cluster_count;
          storage_->tile_idx = next_tile_idx;
          cutlass::arch::fence_view_async_shared();
        }
#pragma unroll
        for (uint32_t rank = 0; rank < 2; ++rank) {
          storage_->tile_ready.arrive(rank);
        }
        storage_->tile_ready.wait(current_work.generation & 1);
        if (current_work.branch != 0) {
          const uint32_t local_address =
              __cvta_generic_to_shared(&storage_->tile_idx);
          const uint32_t remote_address =
              cute::set_block_rank(local_address, 0);
          asm volatile(
              "ld.shared::cluster.b32 %0, [%1];"
              : "=r"(next_tile_idx)
              : "r"(remote_address)
              : "memory");
        }
      }
      const int new_tile_idx = __shfl_sync(
          0xffffffff, next_tile_idx, 0);
      flash::named_barrier_sync(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
      if (threadIdx.x % NumProducerThreads == 0) {
        storage_->tile_idx = new_tile_idx;
      }
      flash::named_barrier_arrive(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      return {
          new_tile_idx, current_work.branch,
          next_generation};
    } else {
      flash::named_barrier_sync(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      const int tile_idx = storage_->tile_idx;
      flash::named_barrier_arrive(
          NumThreads,
          cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
      return {
          tile_idx, current_work.branch,
          current_work.generation + 1};
    }
  }

 private:
  SharedStorage* const storage_;
};

}  // namespace ops
}  // namespace xattn

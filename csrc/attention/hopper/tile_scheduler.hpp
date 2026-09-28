/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdlib>
#include <cstring>

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "cuda/sync/named_barrier.hpp"
#include "attention/cuda/detail/cute_compat.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

inline int
attention_varlen_scheduler_mode_from_env() {
    char const* mode = std::getenv("XATTN_ATTENTION_VARLEN_SCHEDULER");
    // Fallback for callers without the generic scheduler variable.
    if (mode == nullptr) {
        mode = std::getenv("XATTN_FLASH_SCA_VARLEN_SCHEDULER");
    }
    if (mode == nullptr || std::strcmp(mode, "auto") == 0) {
        return 0;
    }
    if (std::strcmp(mode, "persistent") == 0) {
        return 1;
    }
    if (std::strcmp(mode, "ordinary") == 0) {
        return -1;
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////

struct SegmentBwdWorkTile {
    int tile_idx;
    int n_block;
    int bidh;
    int bidb;
    int m_block_min;
    int m_block_max;
};

///////////////////////////////////////////////////////////////////////////////

// Host side kernel arguments
struct TileSchedulerArguments {
    // num_head: num_head_q normally; num_head_k with PackGQA.
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;  // Only used if Varlen and cu_seqlens == nullptr and seqused == nullptr
    int const seqlen_k, headdim, headdim_v, element_size;  // L2 swizzle inputs.
    int* const tile_count_semaphore = nullptr;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    int const* const num_splits_dynamic_ptr = nullptr;
    int const* const num_m_blocks_ptr = nullptr;
    int const* const varlen_batch_idx_ptr = nullptr;
    int const* const num_nheads_in_l2_ptr = nullptr;
    SegmentBwdWorkTile const* const segment_bwd_work_ptr = nullptr;
    // Total packed row count.
    int const total_seqlen = 0;
    // -1=ordinary, 0=auto, 1=persistent.
    int const varlen_scheduler_mode = 0;
};

///////////////////////////////////////////////////////////////////////////////

template<bool Varlen=false, bool Split=false, bool PackGQA=false, int kBlock=128>
class SingleTileScheduler {

public:

    using SharedStorage = int;
    static constexpr bool HasMBlockRange = false;
    static constexpr bool RequiresProducerWarp1 = false;

    // Device side kernel params
    struct Params {
        int const num_blocks, num_head, num_batch, num_splits;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod nsplits_divmod;
        int const* const cu_seqlens;
        int const* const seqused;
        int const* const num_splits_dynamic_ptr = nullptr;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(!Split || !Varlen || args.num_splits_dynamic_ptr != nullptr);
        assert(!Split || !Varlen || args.num_splits < (1 << 16)); // Upper 16 bits encode num_splits.
        return {args.num_blocks, args.num_head, args.num_batch, !Split ? 1 : args.num_splits,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused,
                args.num_splits_dynamic_ptr};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        if constexpr (Varlen) {
            if (params.num_batch > 65535) {
                return {
                    uint32_t(int64_t(params.num_blocks) *
                             int64_t((!Split ? 1 : params.num_splits) *
                                     params.num_head) *
                             int64_t(params.num_batch))};
            }
        }
        return {uint32_t(params.num_blocks), uint32_t((!Split ? 1 : params.num_splits) * params.num_head), uint32_t(params.num_batch)};
    }

    struct WorkTileInfo {
        int block_idx = 0;
        int bidh = 0;
        int bidb = 0;
        int split_idx = 0;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block_idx, bidh, bidb, !Split ? 0 : split_idx};
        }

    };

    CUTLASS_DEVICE
    SingleTileScheduler(SharedStorage* const smem_scheduler) { }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        WorkTileInfo work_info;
        if constexpr (Varlen) {
            if (params.num_batch > 65535) {
                int tile_idx = int(blockIdx.x);
                work_info.block_idx = tile_idx % params.num_blocks;
                tile_idx /= params.num_blocks;
                int const num_head_split =
                    (!Split ? 1 : params.num_splits) * params.num_head;
                work_info.bidh = tile_idx % num_head_split;
                work_info.bidb = tile_idx / num_head_split;
            } else {
                work_info = {
                    int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
            }
        } else {
            work_info = {
                int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
        }
        if constexpr (Split) {
            int split_idx;
            work_info.bidh = params.nsplits_divmod.divmod(split_idx, work_info.bidh);
            work_info.split_idx = split_idx;
        }
        bool is_valid_tile = true;
        if constexpr (Varlen) {
            int seqlen = params.seqused
                ? params.seqused[work_info.bidb]
                : (params.cu_seqlens ? params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb] : params.seqlen);
            if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
            is_valid_tile = work_info.block_idx * kBlock < seqlen;
        }
        if constexpr (Varlen && Split) {
            int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[work_info.bidb] : params.num_splits;
            is_valid_tile &= work_info.split_idx < num_splits_dynamic;
            // Upper 16 bits encode num_splits.
            work_info.split_idx |= (num_splits_dynamic << 16);
        }
        work_info.bidb = is_valid_tile ? work_info.bidb : -1;
        return work_info;
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {0, 0, -1, 0};
    }

};

///////////////////////////////////////////////////////////////////////////////

template<bool Split=false>
class StaticPersistentTileScheduler {

public:

    using SharedStorage = int;
    static constexpr bool HasMBlockRange = false;
    static constexpr bool RequiresProducerWarp1 = false;

    // Device side kernel params
    struct Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
        cutlass::FastDivmod nsplits_divmod;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        return {args.num_blocks * args.num_head * args.num_batch * (!Split ? 1 : args.num_splits),
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head * (!Split ? 1 : args.num_splits)),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits)};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks < num_sm ? params.total_blocks : num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            int split_idx = 0;
            if constexpr (Split) {
                bidh = params.nsplits_divmod.divmod(split_idx, bidh);
            }
            return {block, bidh, bidb, split_idx};
        }

    };

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(SharedStorage* const smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

///////////////////////////////////////////////////////////////////////////////

template<int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
        bool Split=false, bool PackGQA=false, bool WarpSpecialized=true>
class DynamicPersistentTileScheduler {

    // Longest-processing-time-first scheduling with L2-sized head/batch
    // sections. Free SMs claim the longest remaining tile through a semaphore.

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int;
    static constexpr bool HasMBlockRange = false;
    static constexpr bool RequiresProducerWarp1 = false;

protected:
    SharedStorage* const tile_count_smem;

public:

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        long long const size_one_kv_head = long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
        int const size_l2 = 32 * 1024 * 1024;  // 32 MB for K & V
        // Power-of-two L2 section size, scaled by qhead_per_khead outside
        // PackGQA mode.
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        int const swizzle = (size_l2 < size_one_kv_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_kv_head))) * (PackGQA ? 1 : args.qhead_per_khead);
        // Residual sections use their own divisor.
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        int const num_split_blocks = args.num_blocks * (!Split ? 1 : args.num_splits);
        assert(args.tile_count_semaphore != nullptr);
        return {num_split_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * num_split_blocks),
                // FastDivmod divisor is positive.
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks < num_sm ? params.total_blocks : num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            int l2_mod, bidhb, bidhb_residual;
            bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
            // Residual sections use their own divisor.
            if (bidhb < params.num_hb_quotient) {
                block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
            } else {
                block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
            }
            bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
            int split_idx = 0;
            if constexpr (Split) {
                split_idx = params.m_block_divmod.divmod(block, block);
            }
            // Longest-processing-time-first
            block = params.m_block_divmod.divisor - 1 - block;
            return {block, bidh, bidb, split_idx};
        }

    };

    CUTLASS_DEVICE
    DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        if (WarpSpecialized || cutlass::canonical_warp_idx_sync() > 0) {
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
        }
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // Broadcast tile_idx from lane 0.
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % NumProducerThreads == 0) {
                *tile_count_smem = current_work.tile_idx;
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return {new_tile_idx};
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return {tile_idx};
        }
    }

};

///////////////////////////////////////////////////////////////////////////////

class StaticSegmentBwdTileScheduler {

public:
    using SharedStorage = int;
    static constexpr bool HasMBlockRange = true;
    static constexpr bool RequiresProducerWarp1 = false;

    struct Params {
        int const total_blocks;
        SegmentBwdWorkTile const* const work_ptr;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.segment_bwd_work_ptr != nullptr);
        return {args.num_blocks * args.num_head * args.num_batch,
                args.segment_bwd_work_ptr};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks)};
    }

    struct WorkTileInfo {
        int tile_idx;
        SegmentBwdWorkTile work;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {work.n_block, work.bidh, work.bidb, 0 /*split_idx*/};
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t>
        get_m_block_range(Params const& params) const {
            return {work.m_block_min, work.m_block_max};
        }
    };

    CUTLASS_DEVICE
    StaticSegmentBwdTileScheduler(SharedStorage* const smem_scheduler) {}

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int tile_idx) const {
        SegmentBwdWorkTile work =
            tile_idx < params.total_blocks
                ? params.work_ptr[tile_idx]
                : SegmentBwdWorkTile{tile_idx, 0, 0, -1, 0, 0};
        work.tile_idx = tile_idx;
        return {tile_idx, work};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return tile_idx_to_work_tile(params, int(blockIdx.x));
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {params.total_blocks, SegmentBwdWorkTile{params.total_blocks, 0, 0, -1, 0, 0}};
    }

};

///////////////////////////////////////////////////////////////////////////////

class DeterministicSegmentBwdTileScheduler {

public:
    using SharedStorage = int;
    static constexpr bool HasMBlockRange = true;
    static constexpr bool RequiresProducerWarp1 = false;

    struct Params {
        int const total_blocks, num_blocks;
        cutlass::FastDivmod const stream_divmod;
        SegmentBwdWorkTile const* const work_ptr;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.segment_bwd_work_ptr != nullptr);
        int const num_streams = args.num_head * args.num_batch;
        return {
            args.num_blocks * num_streams,
            args.num_blocks,
            cutlass::FastDivmod(num_streams),
            args.segment_bwd_work_ptr};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks)};
    }

    struct WorkTileInfo {
        int tile_idx;
        SegmentBwdWorkTile work;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {work.n_block, work.bidh, work.bidb, 0 /*split_idx*/};
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t>
        get_m_block_range(Params const& params) const {
            return {work.m_block_min, work.m_block_max};
        }
    };

    CUTLASS_DEVICE
    DeterministicSegmentBwdTileScheduler(
        SharedStorage* const smem_scheduler) {}

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int tile_idx) const {
        int stream_idx;
        int const n_block =
            params.stream_divmod.divmod(stream_idx, tile_idx);
        int const dense_tile_idx =
            stream_idx * params.num_blocks + n_block;
        SegmentBwdWorkTile work =
            tile_idx < params.total_blocks
                ? params.work_ptr[dense_tile_idx]
                : SegmentBwdWorkTile{tile_idx, 0, 0, -1, 0, 0};
        work.tile_idx = tile_idx;
        return {tile_idx, work};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return tile_idx_to_work_tile(params, int(blockIdx.x));
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(
        Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(
        Params const& params, WorkTileInfo const& current_work) const {
        return {
            params.total_blocks,
            SegmentBwdWorkTile{
                params.total_blocks, 0, 0, -1, 0, 0}};
    }
};

///////////////////////////////////////////////////////////////////////////////

template <bool Varlen, int kBlock, bool SPT = false, int WorkGroupX = 1,
          bool ContiguousWorkGroup = true>
class SingleTileBwdLPTScheduler {

public:

    static_assert(WorkGroupX >= 1, "BWD scheduler work-group width must be positive");

    using SharedStorage = int;
    static constexpr bool HasMBlockRange = false;
    static constexpr bool RequiresProducerWarp1 = false;

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int const seqlen;
        int const* const cu_seqlens;
        int const* const seqused;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        // BWD argument mapping: args.seqlen=seqlen_k,
        // args.seqlen_k=seqlen_q.
        long long const size_one_qdo_head = long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
        long long const size_one_dqaccum_head = long(args.seqlen_k) * long(args.headdim) * sizeof(float);
        long long const size_one_head = size_one_qdo_head + size_one_dqaccum_head;
        int const size_l2 = 40 * 1024 * 1024;  // 40 MB for Q, dO, and dQaccum
        // Power-of-two L2 section size.
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        int const swizzle = size_l2 < size_one_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_head));
        // Residual sections use their own divisor.
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * args.num_blocks),
                // FastDivmod divisor is positive.
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.seqlen, !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks * WorkGroupX)};
    }

    struct WorkTileInfo {
        int block;
        int bidh;
        int bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block, bidh, bidb, 0 /*split_idx*/};
        }

    };

    CUTLASS_DEVICE
    SingleTileBwdLPTScheduler(SharedStorage* const smem_scheduler) { }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        // CUDA clusters require contiguous reader groups. Independent readers
        // use planes with the per-reader logical tile order.
        int tile_idx;
        if constexpr (ContiguousWorkGroup) {
            tile_idx = blockIdx.x / WorkGroupX;
        } else {
            static_assert(WorkGroupX > 1,
                          "planar work groups require multiple workers");
            tile_idx = blockIdx.x % params.total_blocks;
        }
        int block, bidh, bidb;
        int l2_mod, bidhb, bidhb_residual;
        bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
        // Residual sections use their own divisor.
        if (bidhb < params.num_hb_quotient) {
            block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
        } else {
            block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
        }
        bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
        bool is_valid_tile = true;
        int num_blocks;
        if constexpr (Varlen) {
            int seqlen = params.seqused
                ? params.seqused[bidb]
                : (params.cu_seqlens ? params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb] : params.seqlen);
            num_blocks = cute::ceil_div(seqlen, Int<kBlock>{});
            is_valid_tile = block < num_blocks;
        } else {
            num_blocks = params.block_divmod.divisor;
        }
        if constexpr (SPT) {
            block = num_blocks - block - 1;
        }
        return {block, bidh, is_valid_tile ? bidb : -1};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {0, 0, -1};
    }

};

///////////////////////////////////////////////////////////////////////////////

// Deterministic dense BWD ordering: n-block major, (batch, head) minor.
template <bool ReverseBlocks = false>
class DeterministicDenseBwdTileScheduler {

public:
    using SharedStorage = int;
    static constexpr bool HasMBlockRange = false;
    static constexpr bool RequiresProducerWarp1 = false;

    struct Params {
        int const total_blocks, num_blocks;
        cutlass::FastDivmod const stream_divmod, head_divmod;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        int const num_streams = args.num_head * args.num_batch;
        return {
            args.num_blocks * num_streams,
            args.num_blocks,
            cutlass::FastDivmod(num_streams),
            cutlass::FastDivmod(args.num_head)};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks)};
    }

    struct WorkTileInfo {
        int block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block, bidh, bidb, 0 /*split_idx*/};
        }
    };

    CUTLASS_DEVICE
    DeterministicDenseBwdTileScheduler(
        SharedStorage* const smem_scheduler) {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        int stream_idx;
        int block = params.stream_divmod.divmod(
            stream_idx, int(blockIdx.x));
        if constexpr (ReverseBlocks) {
            block = params.num_blocks - 1 - block;
        }
        int bidh;
        int const bidb = params.head_divmod.divmod(bidh, stream_idx);
        return {block, bidh, bidb};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params,
                       WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params,
                  WorkTileInfo const& current_work) const {
        return {0, 0, -1};
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int NumMmaThreads, int NumProducerThreads>
class DeterministicDensePersistentBwdTileScheduler {
    static constexpr int NumThreads = NumMmaThreads + NumProducerThreads;

public:
    using SharedStorage = int4;
    static constexpr bool HasMBlockRange = false;
    static constexpr bool RequiresProducerWarp1 = false;

    struct Params {
        int const total_blocks;
        int* const tile_count_semaphore;
        cutlass::FastDivmod const stream_divmod, head_divmod;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        int const num_streams = args.num_head * args.num_batch;
        return {
            args.num_blocks * num_streams,
            args.tile_count_semaphore,
            cutlass::FastDivmod(num_streams),
            cutlass::FastDivmod(args.num_head)};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks < num_sm
                             ? params.total_blocks
                             : num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block, bidh, bidb, 0};
        }
    };

private:
    SharedStorage* const work_info_smem;

    CUTLASS_DEVICE
    static WorkTileInfo
    tile_to_work(Params const& params, int tile_idx) {
        if (tile_idx >= params.total_blocks) {
            return {tile_idx, 0, 0, -1};
        }
        int stream_idx;
        int const block = params.stream_divmod.divmod(stream_idx, tile_idx);
        int bidh;
        int const bidb = params.head_divmod.divmod(bidh, stream_idx);
        return {tile_idx, block, bidh, bidb};
    }

public:
    CUTLASS_DEVICE
    DeterministicDensePersistentBwdTileScheduler(
        SharedStorage* const smem_scheduler)
        : work_info_smem(smem_scheduler) {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_to_work(params, int(blockIdx.x));
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(
                    work_info.tile_idx, work_info.block,
                    work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return work_info;
        } else {
            flash::named_barrier_sync(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int4 const work_info = *work_info_smem;
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return {work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx =
                atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params,
                  WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            flash::named_barrier_sync(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::TransformBarrier);
            int const tile_idx =
                __shfl_sync(0xffffffff, current_work.tile_idx, 0);
            WorkTileInfo work_info = tile_to_work(params, tile_idx);
            flash::named_barrier_sync(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(
                    work_info.tile_idx, work_info.block,
                    work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return work_info;
        } else {
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::TransformBarrier);
            flash::named_barrier_sync(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int4 const work_info = *work_info_smem;
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return {work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

// Persistent varlen scheduler with sequential per-stream block traversal and
// dynamic stream assignment. Low-parallelism skew uses ordinary max-grid
// mapping.
template<int kBlock,
         int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup,
         int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool WaitForWorkCompletion=false>
class VarlenSequencePersistentTileScheduler {

    static constexpr int NumThreads = NumMmaThreads + NumProducerThreads;

public:
    using SharedStorage = int4;
    static constexpr bool HasMBlockRange = false;
    static constexpr bool RequiresProducerWarp1 = false;

    struct Params {
        int const num_blocks, num_head, num_batch, seqlen, total_seqlen;
        int* const tile_count_semaphore;
        int const* const cu_seqlens;
        int const* const seqused;
        int const scheduler_mode;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        return {args.num_blocks, args.num_head, args.num_batch, args.seqlen,
                args.total_seqlen,
                args.tile_count_semaphore, args.cu_seqlens, args.seqused,
                args.varlen_scheduler_mode};
    }

    static bool
    use_persistent_grid(int num_head, int num_batch, int seqlen,
                        int total_seqlen, int num_sm,
                        int scheduler_mode = 0) {
        int const num_streams = num_head * num_batch;
        bool const enough_streams = num_streams >= num_sm;
        if (scheduler_mode < 0) {
            return false;
        }
        if (scheduler_mode > 0) {
            return enough_streams;
        }
        bool const low_parallelism = num_streams < 4 * num_sm;
        bool const high_max_to_mean_skew =
            int64_t(seqlen) * num_batch > int64_t(4) * total_seqlen;
        // Ordinary max-grid region: fewer than four stream waves and
        // max-to-mean sequence-length skew greater than four.
        return enough_streams &&
            !(low_parallelism && high_max_to_mean_skew);
    }

    static bool
    use_persistent_grid(Params const& params, int num_sm) {
        return use_persistent_grid(
            params.num_head, params.num_batch, params.seqlen,
            params.total_seqlen, num_sm, params.scheduler_mode);
    }

    static bool
    needs_tile_counter(int num_head, int num_batch, int seqlen,
                       int total_seqlen, int num_sm,
                       int scheduler_mode = 0) {
        int const num_streams = num_head * num_batch;
        return num_streams > num_sm && use_persistent_grid(
            num_head, num_batch, seqlen, total_seqlen, num_sm,
            scheduler_mode);
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        int const num_streams = params.num_head * params.num_batch;
        int const num_ctas = use_persistent_grid(params, num_sm)
            ? num_sm : params.num_blocks * num_streams;
        return {uint32_t(num_ctas)};
    }

    struct WorkTileInfo {
        int block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block, bidh, bidb, 0 /*split_idx*/};
        }
    };

protected:
    SharedStorage* const work_info_smem;

    CUTLASS_DEVICE
    static bool
    use_persistent_mode(Params const& params) {
        int const num_streams = params.num_head * params.num_batch;
        // Counter-free sequential mode: one CTA per multi-block stream.
        return num_streams > int(gridDim.x) ||
            (num_streams == int(gridDim.x) && params.num_blocks > 1);
    }

    CUTLASS_DEVICE
    static int
    sequence_length(Params const& params, int bidb) {
        return params.seqused
            ? params.seqused[bidb]
            : (params.cu_seqlens
                   ? params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb]
                   : params.seqlen);
    }

    CUTLASS_DEVICE
    static WorkTileInfo
    stream_to_work(Params const& params, int stream_idx) {
        int const num_streams = params.num_head * params.num_batch;
        while (stream_idx < num_streams) {
            int const bidb = stream_idx / params.num_head;
            int const bidh = stream_idx - bidb * params.num_head;
            if (sequence_length(params, bidb) > 0) {
                return {0, bidh, bidb};
            }
            if (params.tile_count_semaphore == nullptr) {
                return {0, 0, params.num_batch};
            }
            stream_idx = atomicAdd(params.tile_count_semaphore, 1)
                         + int(gridDim.x);
        }
        return {0, 0, params.num_batch};
    }

public:
    CUTLASS_DEVICE
    VarlenSequencePersistentTileScheduler(
            SharedStorage* const smem_scheduler)
        : work_info_smem(smem_scheduler) {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if (!use_persistent_mode(params)) {
            int const tile_idx = int(blockIdx.x);
            int const stream_idx = tile_idx / params.num_blocks;
            int const block = tile_idx - stream_idx * params.num_blocks;
            int const bidb = stream_idx / params.num_head;
            int const bidh = stream_idx - bidb * params.num_head;
            bool const valid = bidb < params.num_batch &&
                block * kBlock < sequence_length(params, bidb);
            return {block, bidh, valid ? bidb : params.num_batch};
        }
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = stream_to_work(params, int(blockIdx.x));
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(
                    work_info.block, work_info.bidh, work_info.bidb, 0);
            }
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return work_info;
        } else {
            // Initial scheduler metadata synchronization.
            flash::named_barrier_sync(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int4 const work_info = *work_info_smem;
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return {work_info.x, work_info.y, work_info.z};
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params,
                       WorkTileInfo& current_work) const {
        if (!use_persistent_mode(params)) { return; }
        if (threadIdx.x % NumProducerThreads == 0) {
            int const next_block = current_work.block + 1;
            if (next_block * kBlock <
                sequence_length(params, current_work.bidb)) {
                current_work.block = next_block;
            } else if (params.tile_count_semaphore != nullptr) {
                int const next_stream =
                    atomicAdd(params.tile_count_semaphore, 1)
                    + int(gridDim.x);
                current_work = stream_to_work(params, next_stream);
            } else {
                current_work = {0, 0, params.num_batch};
            }
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params,
                  WorkTileInfo const& current_work) const {
        if (!use_persistent_mode(params)) {
            return {0, 0, params.num_batch};
        }
        if constexpr (IsProducerWarp) {
            if constexpr (WaitForWorkCompletion) {
                // Aliased mainloop/epilogue storage is quiescent before the
                // next K/V load.
                flash::named_barrier_sync(
                    NumThreads,
                    cutlass::arch::ReservedNamedBarriers::TransformBarrier);
            }
            WorkTileInfo work_info{
                __shfl_sync(0xffffffff, current_work.block, 0),
                __shfl_sync(0xffffffff, current_work.bidh, 0),
                __shfl_sync(0xffffffff, current_work.bidb, 0)};
            flash::named_barrier_sync(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(
                    work_info.block, work_info.bidh, work_info.bidb, 0);
            }
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return work_info;
        } else {
            if constexpr (WaitForWorkCompletion) {
                flash::named_barrier_arrive(
                    NumThreads,
                    cutlass::arch::ReservedNamedBarriers::TransformBarrier);
            }
            flash::named_barrier_sync(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int4 const work_info = *work_info_smem;
            flash::named_barrier_arrive(
                NumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return {work_info.x, work_info.y, work_info.z};
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template<int kBlockM, int kBlockN, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool Split=false, bool PackGQA=false, bool WarpSpecialized=true, bool LPT = false, bool Sort = false, bool Prepared = true>
class VarlenDynamicPersistentTileScheduler {

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int4;
    static constexpr bool HasMBlockRange = false;
    static constexpr bool RequiresProducerWarp1 = false;

protected:
    SharedStorage* const work_info_smem;

public:

    // Device side kernel params
    struct Params {
        int num_head, num_batch;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod head_divmod;
        cutlass::FastDivmod nsplits_divmod;
        int* const tile_count_semaphore;
        int const* const cu_seqlens;
        int const* const seqused;
        int const* const num_splits_dynamic_ptr;
        int const* const num_m_blocks_ptr;
        int const* const varlen_batch_idx_ptr;
        int const* const num_nheads_in_l2_ptr;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        // Split scheduling treats each split as a logical head.
        assert(args.tile_count_semaphore != nullptr);
        assert(args.num_head < (1 << 16));  // Lower 16 bits encode the head.
        assert(!Split || args.num_splits < (1 << 8)); // Upper 8 bits encode num_splits.
        return {args.num_head, args.num_batch,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                args.tile_count_semaphore, args.cu_seqlens, args.seqused,
                args.num_splits_dynamic_ptr,
                args.num_m_blocks_ptr,
                args.varlen_batch_idx_ptr,
                args.num_nheads_in_l2_ptr};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            auto get_actual_batch = [&](int virtual_batch) {
                if constexpr(Prepared && Sort) {
                    return params.varlen_batch_idx_ptr[virtual_batch];
                } else {
                    return virtual_batch;
                }
            };
            if constexpr (!Split) {
                return {block, bidh, get_actual_batch(bidb), 0 /*split_idx*/};
            } else {
                // bidh packing: head[15:0], split_idx[23:16],
                // num_splits[31:24].
                // Unsigned packing prevents sign extension.
                uint32_t bidh_packed = reinterpret_cast<uint32_t const&>(bidh);
                uint32_t bidh_actual_u = bidh_packed & 0x0000FFFF;
                int bidh_actual = reinterpret_cast<int&>(bidh_actual_u);
                // split_idx packing: split_idx[15:0], num_splits[31:16].
                uint32_t split_idx_u = ((bidh_packed & 0x00FF0000) >> 16) + ((bidh_packed & 0xFF000000) >> 8);
                int split_idx = reinterpret_cast<int&>(split_idx_u);
                return {block, bidh_actual, get_actual_batch(bidb), split_idx};
            }
        }
    };

    CUTLASS_DEVICE
    VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        auto get_num_m_blocks = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            if constexpr (Prepared) {
                return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                    ? params.num_m_blocks_ptr[batch_idx] : 0;
            } else {
                int seqlen;
                if (params.seqused) {
                    seqlen = batch_idx < params.num_batch
                        ? params.seqused[batch_idx] : 0;
                } else if (params.cu_seqlens) {
                    int const cur_cu_seqlen =
                        batch_idx <= params.num_batch
                            ? params.cu_seqlens[batch_idx] : 0;
                    int const next_cu_seqlen = __shfl_down_sync(
                        0xffffffff, cur_cu_seqlen, 1);
                    seqlen = next_cu_seqlen - cur_cu_seqlen;
                } else {
                    seqlen = params.seqlen;
                }
                if constexpr (PackGQA) {
                    seqlen *= params.qhead_per_khead;
                }
                return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                    ? cute::ceil_div(seqlen, kBlockM) : 0;
            }
        };

        auto get_num_splits = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            bool is_valid = batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1;
            if constexpr (!Split) {
                return is_valid ? 1 : 0;
            } else if constexpr(Prepared) {
                return is_valid ? params.num_splits_dynamic_ptr[batch_idx] : 0;
            } else {
                return is_valid ? params.nsplits_divmod.divisor : 0;
            }
        };

        int num_m_blocks = get_num_m_blocks(current_work.bidb);  // Different for each lane
        int num_splits = get_num_splits(current_work.bidb);
        int num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
        // Cumulative number of blocks for the next 31 batches
        int num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
        // Total number of blocks for the next 31 batches
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        // tile_idx stores the starting batch-group offset.
        int group_end_tile = current_work.tile_idx + m_blocks_in_group * params.num_head;  // Same for all lanes
        int bidb = current_work.bidb;
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batch) {
                return {next_tile_idx, 0, 0, params.num_batch};
            }
            num_m_blocks = get_num_m_blocks(bidb);
            num_splits = get_num_splits(bidb);
            num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
            num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += m_blocks_in_group * params.num_head;
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
        // First batch group with end tile greater than next_tile_idx.
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
        bidb += batch_idx_in_group;
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        if constexpr (Split) { num_splits = __shfl_sync(0xffffffff, num_splits, batch_idx_in_group); }
        group_start_tile += (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
        int mh_block = next_tile_idx - group_start_tile;
        int block, bidh;
        if constexpr (LPT) {
            if (!Split || num_splits == 1) {
                auto get_nheads_in_l2 = [&](int batch_idx) {
                    if constexpr(Prepared) {
                        return params.num_nheads_in_l2_ptr[batch_idx];
                    } else {
                        return !PackGQA ? params.qhead_per_khead : 1;
                    }
                };
                int nheads_in_l2 = get_nheads_in_l2(bidb);
                int mh_in_l2 = nheads_in_l2 * num_m_blocks;
                int section_idx = mh_block / mh_in_l2;
                int l2_mod = mh_block - section_idx * mh_in_l2;
                // Tail L2 section.
                int nheads_remainder = params.num_head - section_idx * nheads_in_l2;
                int nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
                block = l2_mod / nheads_in_this_section;
                int bidh_residual = l2_mod - block * nheads_in_this_section;
                bidh = section_idx * nheads_in_l2 + bidh_residual;
                if constexpr(Split) {
                    // Packed work tile: num_splits=1.
                    uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                    bidh = reinterpret_cast<int&>(bidh_packed);
                }
            } else {
                bidh = mh_block / num_m_blocks;
                block = mh_block - bidh * num_m_blocks;
                if constexpr (Split) {
                    int bidh_actual = bidh / num_splits;
                    int split_idx = bidh - bidh_actual * num_splits;
                    uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                    bidh = reinterpret_cast<int&>(bidh_packed);
                }
            }
            block = num_m_blocks - 1 - block;
        } else {
            bidh = mh_block / num_m_blocks;
            block = mh_block - bidh * num_m_blocks;
            if constexpr (Split) {
                int bidh_actual = bidh / num_splits;
                int split_idx = bidh - bidh_actual * num_splits;
                // bidh packing: head[15:0], split_idx[23:16],
                // num_splits[31:24].
                // Unsigned packing prevents sign extension.
                uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                bidh = reinterpret_cast<int&>(bidh_packed);
            }
        }
        return {group_start_tile, block, bidh, bidb};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        // TileCountSmemEmpty arrival is owned by get_initial_work.
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // Broadcast tile_idx from lane 0.
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
            work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int4 work_info = *work_info_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }

};

///////////////////////////////////////////////////////////////////////////////

} // flash

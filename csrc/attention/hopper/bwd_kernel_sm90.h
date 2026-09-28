
/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Modified by: Shicheng Wen (xattn adaptations).

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "bwd_smem_handoff.hpp"
#include "attention/cuda/detail/cute_compat.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class AttentionBwdSm90 {

public:

    // Type Aliases
    static constexpr bool Is_causal = CollectiveMainloop_::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop_::Is_local;
    static_assert(CollectiveMainloop_::Varlen == CollectiveEpilogue_::Varlen);
    static constexpr bool Varlen = CollectiveMainloop_::Varlen;

    // Mainloop derived types
    using CollectiveMainloop = CollectiveMainloop_;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TileShape_MNK_V = typename CollectiveMainloop::TileShape_MNK_V;
    using TiledMmaSdP = typename CollectiveMainloop::TiledMmaSdP;
    using TiledMmadK = typename CollectiveMainloop::TiledMmadK;
    using TiledMmadV = typename CollectiveMainloop::TiledMmadV;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    static constexpr bool dKV_swapAB = CollectiveMainloop::dKV_swapAB;

    // Epilogue derived types
    using CollectiveEpilogue = CollectiveEpilogue_;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaSdP{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaSdP{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    /// Register requirement for Load and Math WGs
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 2 ? 24 : 32;
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 240 : 160;
    // Kernel level shared memory storage
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128> {
            union {
                typename CollectiveMainloop::TensorStorage mainloop;
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;

        struct PipelineStorage : cute::aligned_struct<16> {
            alignas(16) typename CollectiveMainloop::KVSharedStorage kv;
            alignas(16) typename CollectiveMainloop::MainloopPipeline::SharedStorage pipeline_q;
            alignas(16) typename CollectiveMainloop::MainloopPipeline_dO::SharedStorage pipeline_do;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        } pipelines;

    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };

    // Kernel entry point API
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };

    //
    // Methods
    //

    // Underlying arguments alias Params.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        // SM count: caller value or device query.
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }

    // Grid shape from runtime parameters.
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
    }

    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int NumCopyThreads = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        using SmemHandoff = flash::AttentionBwdSmemHandoff<
            CollectiveMainloop, CollectiveEpilogue, NumMmaThreads>;

        using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
        using PipelineParams = typename MainloopPipeline::Params;
        using PipelineState = typename MainloopPipeline::PipelineState;
        using MainloopPipeline_dO = typename CollectiveMainloop::MainloopPipeline_dO;
        using PipelineParams_dO = typename MainloopPipeline_dO::Params;
        using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
        using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
        using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
        using PipelineStateKV = typename CollectiveMainloop::PipelineStateKV;
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();

        // Single-thread TMA descriptor prefetch.
        if (warp_idx == 0 && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }

        // Warp index.
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

        PipelineParams pipeline_params;
        pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ + CollectiveMainloop::TmaTransactionBytesLSE;
        int warp_group_idx = cutlass::canonical_warp_group_idx();
        pipeline_params.role = warp_group_idx == 0
            ? MainloopPipeline::ThreadCategory::Producer
            : MainloopPipeline::ThreadCategory::Consumer;
        pipeline_params.is_leader = warp_group_thread_idx == 0;
        pipeline_params.num_consumers = NumMmaThreads;

        if constexpr (!CollectiveMainloop::HeadPairClusterKVReuse) {
            if (warp_idx == 0 && lane_predicate) {
                shared_storage.pipelines.kv.barrier_KV.init(1 /*numThreads*/);
            }
        }
        // pipeline_q initializes barrier fences.
        MainloopPipeline pipeline_q(shared_storage.pipelines.pipeline_q, pipeline_params, ClusterShape{});
        auto role_dO = warp_group_idx == 0
            ? MainloopPipeline_dO::ThreadCategory::Producer
            : MainloopPipeline_dO::ThreadCategory::Consumer;
        PipelineParams_dO pipeline_params_dO {
            CollectiveMainloop::TmaTransactionBytesdO + CollectiveMainloop::TmaTransactionBytesLSE,
            role_dO, pipeline_params.is_leader, pipeline_params.num_consumers};
        MainloopPipeline_dO pipeline_do(shared_storage.pipelines.pipeline_do, pipeline_params_dO, ClusterShape{});
        MainloopPipelineK pipeline_k = [&] {
            if constexpr (CollectiveMainloop::HeadPairClusterKVReuse) {
                typename MainloopPipelineK::Params params_k;
                params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
                params_k.role = warp_group_idx == 0
                    ? MainloopPipelineK::ThreadCategory::Producer
                    : MainloopPipelineK::ThreadCategory::Consumer;
                params_k.is_leader = pipeline_params.is_leader;
                params_k.num_consumers = NumMmaThreads;
                return MainloopPipelineK(
                    shared_storage.pipelines.kv.pipeline_k,
                    params_k, ClusterShape{});
            } else {
                return nullptr;
            }
        }();
        MainloopPipelineV pipeline_v = [&] {
            if constexpr (CollectiveMainloop::HeadPairClusterKVReuse) {
                typename MainloopPipelineV::Params params_v;
                params_v.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
                params_v.role = warp_group_idx == 0
                    ? MainloopPipelineV::ThreadCategory::Producer
                    : MainloopPipelineV::ThreadCategory::Consumer;
                params_v.is_leader = pipeline_params.is_leader;
                params_v.num_consumers = NumMmaThreads;
                return MainloopPipelineV(
                    shared_storage.pipelines.kv.pipeline_v,
                    params_v, ClusterShape{});
            } else {
                return nullptr;
            }
        }();

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        // Cluster-wide pipeline initialization visibility.
        if constexpr (size(ClusterShape{}) > 1) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }

        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));
        auto get_m_block_range = [&] (auto const& work_tile_info,
                                      cute::tuple<int32_t, int32_t, int32_t> block_coord) {
            if constexpr (TileScheduler::HasMBlockRange) {
                return work_tile_info.get_m_block_range(params.scheduler);
            } else {
                return mainloop.get_m_block_range(params.mainloop, block_coord);
            }
        };

        if (warp_group_idx == 0) {  // Producer
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            if (warp_idx_in_warpgroup == 0) {  // Load K, V, and do TMA on Q and dO
                PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
                PipelineState_dO smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline_dO>();
                PipelineStateKV smem_pipe_write_kv =
                    cutlass::make_producer_start_state<
                        typename CollectiveMainloop::MainloopPipelineKCluster>();
                for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler);
                     work_tile_info.is_valid(params.scheduler);
                     work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)) {
                    auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
                    auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
                    cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};
                    auto m_block_range = get_m_block_range(work_tile_info, block_coord);
                    auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                        scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                    };
                    mainloop.load(params.mainloop, pipeline_q, pipeline_do,
                                  pipeline_k, pipeline_v, smem_pipe_write,
                                  smem_pipe_write_do, smem_pipe_write_kv,
                                  shared_storage, scheduler_prefetch,
                                  block_coord, m_block_range);
                }
                mainloop.load_tail(pipeline_q, pipeline_do, pipeline_k, pipeline_v,
                                   smem_pipe_write, smem_pipe_write_do,
                                   smem_pipe_write_kv);
            } else if (warp_idx_in_warpgroup == 1) {
                if constexpr (CollectiveMainloop::dQacc_use_TMA ||
                              TileScheduler::RequiresProducerWarp1) {
                    for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                         work_tile_info.is_valid(params.scheduler);
                         work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                        auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
                        auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
                        cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};
                        auto m_block_range = get_m_block_range(work_tile_info, block_coord);
                        mainloop.store_dq(params.mainloop, shared_storage, block_coord,
                                          m_block_range);
                        SmemHandoff::sync();
                    }
                }
            }
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
            // Initialize matmul objects.
            TiledMmadK tiled_mma_dK;
            TiledMmadV tiled_mma_dV;

            PipelineState smem_pipe_read;
            PipelineState_dO smem_pipe_read_do;
            PipelineStateKV smem_pipe_read_kv;

            mainloop.mma_init();
            scheduler.init_consumer();

            int work_idx = 0;
            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
                auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
                cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};
                int epilogue_batch = bidb;
                if constexpr (CollectiveMainloop::HeadPairPrivateKVGrad) {
                    int const reader =
                        CollectiveMainloop::head_pair_reader_index();
                    int const logical_batches =
                        CollectiveEpilogue::destination_batch_count(
                            params.epilogue);
                    epilogue_batch = bidb + reader * logical_batches;
                }
                cute::tuple<int32_t, int32_t, int32_t> epilogue_block_coord =
                    {n_block, bidh, epilogue_batch};

                // dK and dV output accumulator.
                Tensor tdKrdK = partition_fragment_C(tiled_mma_dK, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
                Tensor tdVrdV = partition_fragment_C(tiled_mma_dV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK_V{}));
                auto m_block_range = get_m_block_range(work_tile_info, block_coord);
                bool tile_valid = mainloop.mma(
                    params.mainloop, pipeline_q, pipeline_do,
                    pipeline_k, pipeline_v, smem_pipe_read,
                    smem_pipe_read_do, smem_pipe_read_kv,
                    tdKrdK, tdVrdV, threadIdx.x - NumCopyThreads, work_idx,
                    block_coord, m_block_range, shared_storage);
                SmemHandoff::sync();
                if (tile_valid) {
                    epilogue.store(params.epilogue, tdKrdK, tdVrdV, shared_storage,
                                   tiled_mma_dK, tiled_mma_dV,
                                   threadIdx.x - NumCopyThreads,
                                   epilogue_block_coord);
                } else {
                    epilogue.store_zero(params.epilogue,
                                        threadIdx.x - NumCopyThreads,
                                        epilogue_block_coord);
                }

            }
            epilogue.store_tail();
        }

    }

};

} // namespace flash

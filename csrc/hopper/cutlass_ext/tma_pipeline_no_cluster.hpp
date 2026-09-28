/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include<cutlass/pipeline/sm90_pipeline.hpp>

namespace cutlass {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// params.num_consumers % NumThreadsPerWarpGroup == 0
template <int Stages_, class Base=cutlass::PipelineTmaAsync<Stages_>>
class PipelineTmaAsyncNoCluster: public Base {
public:
  using FullBarrier = typename Base::FullBarrier;
  using EmptyBarrier = typename Base::EmptyBarrier;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = typename Base::PipelineState;

  using SharedStorage = typename Base::SharedStorage;
  using ThreadCategory = typename Base::ThreadCategory;
  using Params = typename Base::Params;

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    if (is_initializing_warp) {
      // Full/empty barrier initialization.
      constexpr int producer_arv_cnt = 1;
      uint32_t const num_consumer_warpgroups_per_cluster = (params.num_consumers + NumThreadsPerWarpGroup - 1) / NumThreadsPerWarpGroup;
      uint32_t const multicast_consumer_arrival_count = num_consumer_warpgroups_per_cluster;

      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  template<class ClusterShape, class InitBarriers, class InitMasks>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : Base(storage, params, make_shape(_1{}, _1{}, _1{}) /*cluster_shape*/, cute::false_type{} /*init_barriers*/, cute::false_type{} /*init_masks*/)
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params);
    }

  }

  template<class ClusterShape>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape)
      : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, cute::true_type{}, cute::true_type{}) { }

  template<class ClusterShape, class InitBarriers>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {})
      : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, InitBarriers{}, cute::true_type{}) { }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

private:
  EmptyBarrier* const empty_barrier_ptr_ = nullptr;

  // Consumer completion signal.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(0 /*dst_blockid_*/, uint32_t(threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0) & (!skip) /*is_signaling_thread*/);
  }

};


template <
    int Stages_, int OwnerRank_, uint32_t TransactionBytes_,
    class Base = cutlass::PipelineTmaAsync<Stages_>>
class PipelineTmaAsyncSplitMulticast {
 public:
  using FullBarrier = typename Base::FullBarrier;
  using EmptyBarrier = typename Base::EmptyBarrier;
  using ProducerBarrierType = typename Base::ProducerBarrierType;
  static constexpr uint32_t Stages = Stages_;
  static constexpr int OwnerRank = OwnerRank_;
  static constexpr int PeerRank = 1 - OwnerRank;
  static constexpr uint32_t TransactionBytes = TransactionBytes_;
  using PipelineState = typename Base::PipelineState;
  using SharedStorage = typename Base::SharedStorage;
  using ThreadCategory = typename Base::ThreadCategory;
  using Params = typename Base::Params;

  static_assert(OwnerRank == 0 || OwnerRank == 1);

  template <class ClusterShape>
  CUTLASS_DEVICE
  PipelineTmaAsyncSplitMulticast(
      SharedStorage& storage, Params params,
      ClusterShape cluster_shape)
      : full_barrier_ptr_(&storage.full_barrier_[0]),
        empty_barrier_ptr_(&storage.empty_barrier_[0]) {
    static_assert(cute::size(ClusterShape{}) == 2);
    Base::init_barriers(storage, params, cluster_shape);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    if (cute::block_rank_in_cluster() == OwnerRank) {
      empty_barrier_ptr_[state.index()].wait(state.phase());
      full_barrier_ptr_[state.index()].arrive_and_expect_tx(
          TransactionBytes);
      full_barrier_ptr_[state.index()].arrive_and_expect_tx(
          TransactionBytes, PeerRank);
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return reinterpret_cast<ProducerBarrierType*>(
        &full_barrier_ptr_[state.index()]);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState) {}

  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    if (cute::block_rank_in_cluster() == OwnerRank) {
      CUTLASS_PRAGMA_UNROLL
      for (int count = 0; count < Stages; ++count) {
        empty_barrier_ptr_[state.index()].wait(state.phase());
        ++state;
      }
    }
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(
      PipelineState state, uint32_t skip_wait = false) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    const bool ready =
        full_barrier_ptr_[state.index()].try_wait(state.phase());
    return {static_cast<BarrierStatus>(ready)};
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    full_barrier_ptr_[state.index()].wait(state.phase());
  }

  CUTLASS_DEVICE
  void consumer_wait(
      PipelineState state, ConsumerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[state.index()].wait(state.phase());
    }
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    const uint32_t signal =
        threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0;
    empty_barrier_ptr_[state.index()].arrive(
        OwnerRank, signal);
  }

 private:
  FullBarrier* const full_barrier_ptr_ = nullptr;
  EmptyBarrier* const empty_barrier_ptr_ = nullptr;
};


////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cutlass

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "attention/hopper/epilogue_fwd.hpp"

namespace xattn {
namespace ops {

template <
    class TileShapeMNKPV_, class ClusterShape_, class Element_,
    class ArchTag_, int NumEpilogueThreads_, bool Varlen_,
    bool PackGQA_, bool Split_, bool FP8PermuteCol_>
struct FlashSoftDeltaSingleCTAEpilogueFwd {
  using Base = flash::CollectiveEpilogueFwd<
      TileShapeMNKPV_, ClusterShape_, Element_, ArchTag_,
      NumEpilogueThreads_, Varlen_, PackGQA_, Split_,
      FP8PermuteCol_>;
  using TileShape_MNK_PV = TileShapeMNKPV_;
  using ClusterShape = ClusterShape_;
  using Element = Element_;
  using ElementPartial = float;
  using ArchTag = ArchTag_;
  using SmemLayoutO = typename Base::SmemLayoutO;
  using SmemCopyAtomO = typename Base::SmemCopyAtomO;
  using ShapeO = typename Base::ShapeO;
  using StrideO = typename Base::StrideO;

  static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
  static constexpr int NumOutputThreads = NumEpilogueThreads;
  static constexpr bool Varlen = Varlen_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr bool Split = Split_;
  static constexpr bool Use_smem = true;
  static constexpr bool Use_TMA_O = false;
  static constexpr bool LargeHeadDimV = Base::LargeHeadDimV;
  static constexpr int kBlockM = Base::kBlockM;
  static constexpr int kLogicalBlockM = kBlockM / 2;
  static constexpr int kHeadDimV = Base::kHeadDimV;
  static constexpr int kGateGroupDim = kHeadDimV / 4;
  static constexpr int kSmemElements = cute::cosize_v<SmemLayoutO>;
  static constexpr int kVectorElements = sizeof(uint4) / sizeof(Element);
  static constexpr int kBlockKGmem = Base::kBlockKGmem;
  static constexpr int kGmemThreadsPerRow = Base::kGmemThreadsPerRow;
  static constexpr int kRowsPerPass =
      NumOutputThreads / kGmemThreadsPerRow;
  static constexpr int kRowPasses =
      kLogicalBlockM / kRowsPerPass;
  static constexpr int kColumnPasses =
      kHeadDimV / kBlockKGmem;
  static constexpr bool kUseOutputThreadGuard = kHeadDimV == 32;

  using GmemLayoutAtom = cute::Layout<
      cute::Shape<
          cute::Int<NumOutputThreads / kGmemThreadsPerRow>,
          cute::Int<kGmemThreadsPerRow>>,
      cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>>;
  static_assert(!Varlen);
  static_assert(!PackGQA);
  static_assert(!Split);
  static_assert(
      cute::size(ClusterShape{}) == 1 ||
      cute::size(ClusterShape{}) == 2);
  static_assert(
      NumEpilogueThreads == kBlockM ||
      NumEpilogueThreads == 2 * kBlockM);
  static_assert(
      kBlockM == 128 || kBlockM == 192);
  static_assert(kHeadDimV % 4 == 0);
  static_assert(kGateGroupDim % kVectorElements == 0);
  static_assert(kGmemThreadsPerRow >= 4);
  static_assert(sizeof(Element) == sizeof(uint16_t));
  static_assert(
      kLogicalBlockM %
              cute::size<0>(GmemLayoutAtom{}) ==
          0);
  static_assert(kHeadDimV % kBlockKGmem == 0);

  union OutputVector {
    uint4 packed;
    Element values[kVectorElements];
  };

  struct TensorStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, kSmemElements> smem_o;
  };

  struct Arguments {
    Element* ptr_O;
    ShapeO shape_O;
    StrideO stride_O;
    const Element* ptr_gate;
    int64_t gate_batch_stride;
    int64_t gate_row_stride;
    int64_t gate_head_stride;
    int64_t gate_group_stride;
  };

  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE
  static void initialize_shared_storage(SharedStorage&) {}

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const&) {}

  template <
      typename SharedStorage, typename FrgTensorO,
      typename FrgTensorLSE, typename TiledMma>
  CUTLASS_DEVICE
  void store(
      Params const& params, FrgTensorO& tOrO,
      FrgTensorLSE const&, SharedStorage& shared_storage,
      TiledMma tiled_mma, int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> const&
          block_coord) {
    if constexpr (kHeadDimV == 256) {
      asm volatile("" : "+r"(thread_idx));
    }
    flash::named_barrier_sync(
        NumEpilogueThreads,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    auto sO = SmemTensor(shared_storage);
    auto tOrO_out = cute::make_tensor_like<Element>(tOrO);
    flash::convert_type_out(tOrO, tOrO_out);
    auto smem_tiled_copy = cute::make_tiled_copy_C(
        SmemCopyAtomO{}, tiled_mma);
    auto smem_thread_copy =
        smem_tiled_copy.get_thread_slice(thread_idx);
    auto register_output = smem_thread_copy.retile_S(tOrO_out);
    auto shared_output = smem_thread_copy.partition_D(sO);
    cute::copy(smem_tiled_copy, register_output, shared_output);
    Finish(params, shared_storage, thread_idx, block_coord);
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE
  void store_zero(
      Params const& params, SharedStorage& shared_storage,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> const&
          block_coord) {
    if constexpr (kHeadDimV == 256) {
      asm volatile("" : "+r"(thread_idx));
    }
    flash::named_barrier_sync(
        NumEpilogueThreads,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    auto* storage = shared_storage.tensors.epilogue.smem_o.data();
    for (int index = thread_idx; index < kSmemElements;
         index += NumEpilogueThreads) {
      storage[index] = Element(0);
    }
    Finish(params, shared_storage, thread_idx, block_coord);
  }

  CUTLASS_DEVICE
  void store_tail() {}

 private:
  template <typename SharedStorage>
  CUTLASS_DEVICE
  static auto SmemTensor(SharedStorage& shared_storage) {
    return cute::make_tensor(
        cute::make_smem_ptr(
            shared_storage.tensors.epilogue.smem_o.data()),
        SmemLayoutO{});
  }

  CUTLASS_DEVICE
  static float Sigmoid(Element value) {
    const float input = static_cast<float>(value);
    return 0.5f * (1.0f + __tanhf(0.5f * input));
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE
  void Finish(
      Params const& params, SharedStorage& shared_storage,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> const&
          block_coord) {
    cutlass::arch::fence_view_async_shared();
    flash::named_barrier_sync(
        NumEpilogueThreads,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    bool write_tile = true;
    if constexpr (cute::size(ClusterShape{}) == 2) {
      const int m_block = cute::get<0>(block_coord);
      const int seqlen = cute::get<0>(params.shape_O);
      const int num_blocks =
          (seqlen + kLogicalBlockM - 1) / kLogicalBlockM;
      write_tile = cute::block_rank_in_cluster() == 0 ||
          m_block + 1 < num_blocks;
    }
    if (write_tile &&
        (!kUseOutputThreadGuard || thread_idx < NumOutputThreads)) {
      auto [m_block, packed_head, batch, split] = block_coord;
      static_cast<void>(split);
      const int logical_head = packed_head / 2;
      auto sO = SmemTensor(shared_storage);
      auto primary = cute::local_tile(
          sO,
          cute::Shape<
              cute::Int<kLogicalBlockM>, cute::Int<kHeadDimV>>{},
          cute::make_coord(cute::_0{}, cute::_0{}));
      auto correction = cute::local_tile(
          sO,
          cute::Shape<
              cute::Int<kLogicalBlockM>, cute::Int<kHeadDimV>>{},
          cute::make_coord(cute::_1{}, cute::_0{}));
      const int seqlen = cute::get<0>(params.shape_O);
      const int row_in_pass = thread_idx / kGmemThreadsPerRow;
      const int vector_in_group =
          thread_idx % kGmemThreadsPerRow;
      const bool gate_lane = vector_in_group < 4;
      const int lane = thread_idx % cutlass::NumThreadsPerWarp;
      const int source_lane = lane - vector_in_group;
      const int64_t output_base =
          int64_t(batch) * cute::get<3>(params.stride_O) +
          int64_t(logical_head) * cute::get<2>(params.stride_O);
      int64_t gate_head_offset = 0;
      if (gate_lane) {
        gate_head_offset =
            int64_t(batch) * params.gate_batch_stride +
            int64_t(logical_head) * params.gate_head_stride;
      }
#pragma unroll
      for (int row_pass = 0; row_pass < kRowPasses; ++row_pass) {
        const int row = row_in_pass + row_pass * kRowsPerPass;
        const int sequence = m_block * kLogicalBlockM + row;
        const bool active = sequence < seqlen;
        int64_t gate_row_offset = 0;
        if (active && gate_lane) {
          gate_row_offset = gate_head_offset +
              int64_t(sequence) * params.gate_row_stride;
        }
        const float lane_gate = active && gate_lane
            ? Sigmoid(params.ptr_gate[
                  gate_row_offset +
                  int64_t(vector_in_group) *
                      params.gate_group_stride])
            : 0.0f;
#pragma unroll
        for (int column_pass = 0; column_pass < kColumnPasses;
             ++column_pass) {
          const int column =
              column_pass * kBlockKGmem +
              vector_in_group * kVectorElements;
          const int gate_group = column / kGateGroupDim;
          const float gate = __shfl_sync(
              0xffffffff, lane_gate, source_lane + gate_group);
          if (active) {
            OutputVector read_vector;
            OutputVector correction_vector;
            OutputVector combined_vector;
            read_vector.packed =
                *reinterpret_cast<const uint4*>(&primary(row, column));
            correction_vector.packed =
                *reinterpret_cast<const uint4*>(&correction(row, column));
            if constexpr (
                cute::is_same_v<Element, cutlass::bfloat16_t> ||
                cute::is_same_v<Element, cutlass::half_t>) {
              using PackedVector =
                  cutlass::Array<Element, kVectorElements>;
              const auto& read_packed =
                  reinterpret_cast<const PackedVector&>(
                      read_vector.values);
              const auto& correction_packed =
                  reinterpret_cast<const PackedVector&>(
                      correction_vector.values);
              const PackedVector combined_packed =
                  cutlass::multiply_add<
                      PackedVector, PackedVector, PackedVector>{}(
                      static_cast<Element>(-gate),
                      correction_packed, read_packed);
              combined_vector.packed =
                  reinterpret_cast<const uint4&>(combined_packed);
            } else {
#pragma unroll
              for (int element = 0; element < kVectorElements; ++element) {
                combined_vector.values[element] = static_cast<Element>(
                    static_cast<float>(read_vector.values[element]) -
                    gate * static_cast<float>(
                        correction_vector.values[element]));
              }
            }
            const int64_t output_index =
                output_base +
                int64_t(sequence) * cute::get<0>(params.stride_O) +
                column;
            *reinterpret_cast<uint4*>(params.ptr_O + output_index) =
                combined_vector.packed;
          }
        }
      }
    }

    if constexpr (kHeadDimV == 32) {
      flash::named_barrier_sync(
          NumEpilogueThreads,
          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }
    if constexpr (cute::size(ClusterShape{}) == 1) {
      shared_storage.pipelines.barrier_O.arrive();
    } else {
#pragma unroll
      for (uint32_t cta_id = 0;
           cta_id < cute::size(ClusterShape{}); ++cta_id) {
        shared_storage.pipelines.barrier_O.arrive(cta_id);
      }
    }
  }
};

}  // namespace ops
}  // namespace xattn

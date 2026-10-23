#pragma once

#include "attention/hopper/epilogue_fwd.hpp"

namespace xattn {
namespace ops {

// Forward epilogue over logical heads l = g * R + r with class-major storage:
// head l is stored at readers head g for r = 0 (the negative reads, heads [0, Hkv))
// and at Hkv + g * (R - 1) + r - 1 for r > 0 (the query reads, heads
// [Hkv, Hkv + H)). Output, LSE and FP32 state share the head order.
template <class Base>
struct GroupSoftDeltaReaderEpilogue : Base {
  static_assert(!Base::Varlen && !Base::PackGQA && !Base::Split);

  struct Arguments {
    typename Base::Arguments base;
    int kv_heads;
  };

  struct Params {
    typename Base::Params base;
    int kv_heads;
    cutlass::FastDivmod readers_divmod;
  };

  using BaseEpilogue = Base;

  static Params to_underlying_arguments(Arguments const& args) {
    int const logical_heads = cute::get<2>(args.base.shape_O);
    return {Base::to_underlying_arguments(args.base), args.kv_heads,
            cutlass::FastDivmod(logical_heads / args.kv_heads)};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    Base::prefetch_tma_descriptors(params.base);
  }

  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t, int32_t, int32_t> reader_coord(
      Params const& params, cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
    int group, reader;
    params.readers_divmod.fast_divmod(group, reader, cute::get<1>(block_coord));
    int const rep = params.readers_divmod.divisor - 1;
    int const head = reader == 0 ? group : params.kv_heads + group * rep + reader - 1;
    return {cute::get<0>(block_coord), head, cute::get<2>(block_coord), cute::get<3>(block_coord)};
  }

  template <bool StoreOutputState = true, typename SharedStorage, typename FrgTensorO,
            typename FrgTensorLSE, typename TiledMma>
  CUTLASS_DEVICE void store(Params const& params, FrgTensorO& tOrO, FrgTensorLSE const& lse,
                            SharedStorage& shared_storage, TiledMma tiled_mma, int thread_idx,
                            cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
    Base::store(params.base, tOrO, lse, shared_storage, tiled_mma, thread_idx, reader_coord(params, block_coord));
  }

  template <bool StoreOutputState = true, typename SharedStorage>
  CUTLASS_DEVICE void store_zero(Params const& params, SharedStorage& shared_storage, int thread_idx,
                                 cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
    Base::store_zero(params.base, shared_storage, thread_idx, reader_coord(params, block_coord));
  }
};

template <class TileShape, class ClusterShape, class Element, class ArchTag, int NumThreads, bool Varlen,
          bool PackGQA, bool Split, bool FP8PermuteCol>
using GroupSoftDeltaEpilogueFwd = GroupSoftDeltaReaderEpilogue<flash::CollectiveEpilogueFwd<
    TileShape, ClusterShape, Element, ArchTag, NumThreads, Varlen, PackGQA, Split, FP8PermuteCol>>;

template <class TileShape, class ClusterShape, class Element, class ArchTag, int NumThreads, bool Varlen,
          bool PackGQA, bool Split, bool FP8PermuteCol>
using GroupSoftDeltaEpilogueFwdNoState = GroupSoftDeltaReaderEpilogue<flash::CollectiveEpilogueFwdNoState<
    TileShape, ClusterShape, Element, ArchTag, NumThreads, Varlen, PackGQA, Split, FP8PermuteCol>>;

}  // namespace ops
}  // namespace xattn

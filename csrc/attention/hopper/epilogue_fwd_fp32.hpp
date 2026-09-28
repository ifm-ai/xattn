#pragma once

#include "attention/hopper/epilogue_fwd.hpp"

namespace flash {

// Single FP32 output for internal forward/recomputation.
template <class TileShape, class ClusterShape, class Element_, class ArchTag,
          int NumThreads, bool Varlen_, bool PackGQA, bool Split,
          bool FP8PermuteCol>
struct CollectiveEpilogueFwdFp32 {
    static_assert(!PackGQA && !Split && !FP8PermuteCol);
    static_assert(ArchTag::kMinComputeCapability >= 90);
    using Element = Element_;
    using Base = CollectiveEpilogueFwd<TileShape, ClusterShape, Element,
        ArchTag, NumThreads, Varlen_, PackGQA, Split, FP8PermuteCol>;
    using Arguments = typename Base::Arguments;
    using ShapeO = typename Base::ShapeO;
    using StrideO = typename Base::StrideO;
    using StrideLSE = typename Base::StrideLSE;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool Use_TMA_O = false;
    static constexpr bool LargeHeadDimV = false;
    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kHeadDimV = get<1>(TileShape{});
    static_assert(kHeadDimV <= 256);
    struct TensorStorage : cute::aligned_struct<128> {};

    struct Params {
        float* ptr_O;
        ShapeO shape_O;
        StrideO stride_O;
        float* ptr_LSE;
        StrideLSE stride_LSE;
        int const* cu_seqlens;
        int const* seqused;
    };

    static Params to_underlying_arguments(Arguments const& args) {
        return {reinterpret_cast<float*>(args.ptr_O), args.shape_O,
                args.stride_O, args.ptr_LSE, args.stride_LSE,
                args.cu_seqlens, args.seqused};
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE static void initialize_shared_storage(SharedStorage&) {}
    CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const&) {}
    CUTLASS_DEVICE void store_tail() {}

    template <typename SharedStorage, typename FrgTensorO,
              typename FrgTensorLSE, typename TiledMma>
    CUTLASS_DEVICE void store(
        Params const& params, FrgTensorO& output, FrgTensorLSE const& lse,
        SharedStorage& shared_storage, TiledMma tiled_mma, int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
        auto [m_block, head, batch, split] = block_coord;
        // All MMA threads finish reading V before its reuse.
        flash::named_barrier_sync(
            NumThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        #pragma unroll
        for (uint32_t cta = 0; cta < size(ClusterShape{}); ++cta) {
            shared_storage.pipelines.barrier_O.arrive(cta);
        }
        flash::SeqlenInfo<Varlen, kBlockM> info{
            batch, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
        bool const packed = Varlen && params.cu_seqlens;
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor coords = thread_mma.partition_C(
            cute::make_identity_tensor(select<0, 1>(TileShape{})));
        Tensor coord_rc = make_tensor(
            coords.data(), flash::convert_layout_acc_rowcol(coords.layout()));
        Tensor rows = coord_rc(_, _0{});
        Tensor cols = coord_rc(_0{}, _);
        Tensor mLSE = make_tensor(
            make_gmem_ptr(params.ptr_LSE + info.offset),
            select<0, 2, 3, 4>(params.shape_O), params.stride_LSE)(
                _, head, packed ? 0 : batch, _0{});
        #pragma unroll
        for (int m = 0; m < size(lse); ++m) {
            int const row = m_block * kBlockM + get<0>(rows(m));
            if (get<1>(rows(_0{})) == 0 && row < info.seqlen) {
                mLSE(row) = lse(m);
            }
        }
        Tensor mO = make_tensor(
            make_gmem_ptr(params.ptr_O + info.offset * get<0>(params.stride_O)),
            params.shape_O, params.stride_O)(_, _, head, packed ? 0 : batch, _0{});
        Tensor gO = local_tile(mO, select<0, 1>(TileShape{}), make_coord(m_block, _0{}));
        Tensor target = thread_mma.partition_C(gO);
        Tensor out_rc = make_tensor(
            output.data(), flash::convert_layout_acc_rowcol(output.layout()));
        Tensor target_rc = make_tensor(
            target.data(), flash::convert_layout_acc_rowcol(target.layout()));
        Tensor source_copy = cute::tiled_divide(out_rc, Shape<_1, _2>{});
        Tensor target_copy = cute::tiled_divide(target_rc, Shape<_1, _2>{});
        cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float> copy;
        #pragma unroll
        for (int m = 0; m < size(rows); ++m) {
            if (get<0>(rows(m)) < info.seqlen - m_block * kBlockM) {
                #pragma unroll
                for (int k = 0; k < size(cols) / 2; ++k) {
                    if (get<1>(cols(k * 2)) < get<1>(params.shape_O)) {
                        cute::copy(copy, source_copy(_, m, k), target_copy(_, m, k));
                    }
                }
            }
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void store_zero(
        Params const& params, SharedStorage&, int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
        auto [m_block, head, batch, split] = block_coord;
        flash::SeqlenInfo<Varlen, kBlockM> info{
            batch, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
        bool const packed = Varlen && params.cu_seqlens;
        Tensor mO = make_tensor(
            make_gmem_ptr(params.ptr_O + info.offset * get<0>(params.stride_O)),
            params.shape_O, params.stride_O)(_, _, head, packed ? 0 : batch, _0{});
        Tensor mLSE = make_tensor(
            make_gmem_ptr(params.ptr_LSE + info.offset),
            select<0, 2, 3, 4>(params.shape_O), params.stride_LSE)(
                _, head, packed ? 0 : batch, _0{});
        int const row = m_block * kBlockM + thread_idx;
        if (thread_idx < kBlockM && row < info.seqlen) { mLSE(row) = -INFINITY; }
        #pragma unroll
        for (int linear = thread_idx; linear < kBlockM * kHeadDimV; linear += NumThreads) {
            int const r = m_block * kBlockM + linear / kHeadDimV;
            int const c = linear % kHeadDimV;
            if (r < info.seqlen && c < get<1>(params.shape_O)) { mO(r, c) = 0.0f; }
        }
    }
};

}  // namespace flash

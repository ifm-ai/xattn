/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Modified by: Shicheng Wen (xattn adaptations).

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>  // FastDivMod
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"

#include "attention/sequence/seqlen.h"
#include "cuda/sync/named_barrier.hpp"
#include "attention/cuda/pack_gqa.h"
#include "attention/cuda/detail/cute_compat.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_PV_, class ClusterShape_, class Element_, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool PackGQA_, bool Split_, bool FP8PermuteCol=false>
struct CollectiveEpilogueFwd {

    using TileShape_MNK_PV = TileShape_MNK_PV_;
    using ClusterShape = ClusterShape_;
    using Element = Element_;
    using ElementPartial = float;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;
    static constexpr bool Use_smem = !(Split && !Varlen);
    static constexpr bool Use_TMA_O = ArchTag::kMinComputeCapability >= 90 && !Varlen && !Split && !PackGQA;

    static_assert(ArchTag::kMinComputeCapability >= 80);
    static_assert(ArchTag::kMinComputeCapability >= 90 || CUTE_STATIC_V(size(ClusterShape{})) == 1);
    static_assert(sizeof(Element) <= 2);

    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    static constexpr int kHeadDimV = get<1>(TileShape_MNK_PV{});

    static constexpr bool LargeHeadDimV = kHeadDimV > 256;

    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimV % kGmemElemsPerStore == 0, "Headdim must be a multiple of kGmemElemsPerStore");
    // Each row spans 64 elements; each thread owns 4 M and 2 K elements.
    static constexpr int kBytePerRow = kHeadDimV * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
    // PackGQA pointer computation is shared within a warp.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumEpilogueThreads / kGmemThreadsPerRow");
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomOTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK_PV{})), decltype(cute::get<1>(TileShape_MNK_PV{}))>());
    using SmemLayoutOTMA = decltype(tile_to_shape(SmemLayoutAtomOTMA{}, select<0, 1>(TileShape_MNK_PV{})));
    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));
    using SmemLayoutOSTS = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_MNK_PV{})));
    using SmemLayoutO = std::conditional_t<ArchTag::kMinComputeCapability >= 90, SmemLayoutOTMA, SmemLayoutOSTS>;

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch, num_splits)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t, int64_t>;            // (seqlen_q, head, batch, num_splits)
    // ((qhead_per_khead, seqlen_q), d, nheads_kv, batch, num_splits)
    using ShapeOPacked = std::conditional_t<!PackGQA, ShapeO, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t, int32_t>>;
    using StrideOPacked = std::conditional_t<!PackGQA, StrideO, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t, int64_t>>;
    // ((qhead_per_khead, seqlen_q), nheads_kv, batch, num_splits)
    using ShapeLSEPacked = std::conditional_t<!PackGQA, cute::Shape<int32_t, int32_t, int32_t, int32_t>, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideLSEPacked = std::conditional_t<!PackGQA, StrideLSE, cute::Stride<cute::Stride<int64_t, _1>, int64_t, int64_t, int64_t>>;

    using EpilogueTile_MN = decltype(select<0, 1>(TileShape_MNK_PV{}));
    using CopyOpR2S = std::conditional_t<
        ArchTag::kMinComputeCapability >= 90,
        // cute::SM90_U32x4_STSM_N if Element size is 2 bytes (fp16, bf16)
        decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<StrideO, Element, EpilogueTile_MN>()),
        AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using SmemCopyAtomO = Copy_Atom<CopyOpR2S, Element>;

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> : 0> smem_o;
    };

    using TMA_O = std::conditional_t<
        Use_TMA_O,
        decltype(make_tma_copy(
            GmemTiledCopyOTMA{},
            make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeO{}, StrideO{}),
            SmemLayoutOTMA{},
            select<0, 1>(TileShape_MNK_PV{}),
            _1{})),  // no mcast for O
        std::nullptr_t
    >;

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        ElementPartial* ptr_O_partial;
        StrideO const stride_O_partial;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        float* ptr_LSE_partial;
        StrideLSE const stride_LSE_partial;
        int32_t const nheads_kv;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        // Optional output state extends the fixed argument prefix.
        float* ptr_O_state = nullptr;
        StrideO const stride_O_state = {};
    };

    // Device side kernel params
    struct Params {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        ShapeOPacked const shape_O_packed;
        StrideOPacked const stride_O_packed;
        ElementPartial* ptr_O_partial;
        StrideO const stride_O_partial;
        StrideOPacked const stride_O_partial_packed;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        ShapeLSEPacked const shape_LSE_packed;
        StrideLSEPacked const stride_LSE_packed;
        float* ptr_LSE_partial;
        StrideLSE const stride_LSE_partial;
        StrideLSEPacked const stride_LSE_partial_packed;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_O tma_store_O;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        // Optional high-precision state extends the fixed parameter prefix.
        float* ptr_O_state = nullptr;
        StrideOPacked const stride_O_state_packed = {};
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
        TMA_O tma_store_O = [&]{
            if constexpr (Use_TMA_O) {
                return make_tma_copy(GmemTiledCopyOTMA{}, mO, SmemLayoutO{}, select<0, 1>(TileShape_MNK_PV{}), _1{}); // no mcast
            } else {
                return nullptr;
            }
        }();
        // If PackGQA, reshape O to be ((qhead_per_khead, seqlen_q), head_size, nhead_k, batch_size, num_splits)
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_O), args.nheads_kv);
        auto const shape_O_packed = cute::conditional_return<!PackGQA>(
            args.shape_O,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)), get<1>(args.shape_O), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O))
        );
        auto const stride_O_packed = cute::conditional_return<!PackGQA>(
            args.stride_O,
            make_stride(make_stride(get<2>(args.stride_O), get<0>(args.stride_O)), get<1>(args.stride_O), get<2>(args.stride_O) * qhead_per_khead, get<3>(args.stride_O), get<4>(args.stride_O))
        );
        auto const stride_O_state_packed = cute::conditional_return<!PackGQA>(
            args.stride_O_state,
            make_stride(make_stride(get<2>(args.stride_O_state), get<0>(args.stride_O_state)), get<1>(args.stride_O_state), get<2>(args.stride_O_state) * qhead_per_khead, get<3>(args.stride_O_state), get<4>(args.stride_O_state))
        );
        auto const stride_O_partial_packed = cute::conditional_return<!PackGQA>(
            args.stride_O_partial,
            make_stride(make_stride(get<2>(args.stride_O_partial), get<0>(args.stride_O_partial)), get<1>(args.stride_O_partial), get<2>(args.stride_O_partial) * qhead_per_khead, get<3>(args.stride_O_partial), get<4>(args.stride_O_partial))
        );
        // If PackGQA, Reshape LSE to be ((qhead_per_khead, seqlen_q), nhead_k, batch_size, num_splits)
        auto const shape_LSE_packed = cute::conditional_return<!PackGQA>(
            select<0, 2, 3, 4>(args.shape_O),
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O))
        );
        auto const stride_LSE_packed = cute::conditional_return<!PackGQA>(
            args.stride_LSE,
            make_stride(make_stride(get<1>(args.stride_LSE), get<0>(args.stride_LSE)), get<1>(args.stride_LSE) * qhead_per_khead, get<2>(args.stride_LSE), get<3>(args.stride_LSE))
        );
        auto const stride_LSE_partial_packed = cute::conditional_return<!PackGQA>(
            args.stride_LSE_partial,
            make_stride(make_stride(get<1>(args.stride_LSE_partial), get<0>(args.stride_LSE_partial)), get<1>(args.stride_LSE_partial) * qhead_per_khead, get<2>(args.stride_LSE_partial), get<3>(args.stride_LSE_partial))
        );
        return {args.ptr_O, args.shape_O, args.stride_O, shape_O_packed, stride_O_packed,
                args.ptr_O_partial, args.stride_O_partial, stride_O_partial_packed,
                args.ptr_LSE, args.stride_LSE, shape_LSE_packed, stride_LSE_packed,
                args.ptr_LSE_partial, args.stride_LSE_partial, stride_LSE_partial_packed,
                cutlass::FastDivmod(qhead_per_khead),
                tma_store_O, args.cu_seqlens, args.seqused,
                args.ptr_O_state, stride_O_state_packed};
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void initialize_shared_storage(SharedStorage&) {}

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA_O) {
            cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
        }
    }

    template <bool StoreOutputState=true, typename SharedStorage,
              typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO& tOrO,
          FrgTensorLSE const& lse,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [m_block, bidh, bidb, split_idx] = block_coord;
        int num_splits = get<4>(params.shape_O_packed);
        if constexpr (Split && Varlen) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16; // first 16 bits are for num_splits
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            num_splits = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            split_idx &= 0x0000FFFF;  // Only use the lower 16 bits of split_idx
        }
        bool const is_split = !Split ? false : (!Varlen ? true : num_splits > 1);

        Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});

        static constexpr bool NeedFP8Permute = FP8PermuteCol && (sizeof(Element) == 2 || sizeof(Element) == 4);
        // Permute FP32 state before conversion; otherwise permute the converted output.
        if constexpr (NeedFP8Permute && Split) { flash::permute_output_fp8_Vcolmajor(tOrO); }
        Tensor tOrO_out = make_tensor_like<Element>(tOrO);
        flash::convert_type_out(tOrO, tOrO_out);
        if constexpr (NeedFP8Permute && !Split) { flash::permute_output_fp8_Vcolmajor(tOrO_out); }

        // All epilogue threads synchronize after reading V and before the next Q load.
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        // Step 1: Write O from rmem -> smem
        if constexpr (Use_smem) {
            auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
            auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
            Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
            Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
            if constexpr (Use_TMA_O) {
                cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
                cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            } else {
                flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            }
        } else {
            if constexpr (ArchTag::kMinComputeCapability >= 90) {
                #pragma unroll
                for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                    shared_storage.pipelines.barrier_O.arrive(cta_id);
                }
            }
        }

        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
        bool is_varlen = Varlen && params.cu_seqlens;
        int offset_o = seqlen_info.offset;
        int seqlen_o = seqlen_info.seqlen;
        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);

        // Step 2: Write LSE from rmem -> gmem
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        // (MMA,MMA_M,MMA_K)
        Tensor taccOcO = thread_mma.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
        static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M

        using PackGQA_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, Element>;
        using PackGQApartial_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, ElementPartial>;

        Tensor mLSE = make_tensor(make_gmem_ptr((!is_split ? params.ptr_LSE : params.ptr_LSE_partial) + offset_o * get<0>(!is_split ? params.stride_LSE : params.stride_LSE_partial)),
                                  params.shape_LSE_packed,
                                  !is_split ? params.stride_LSE_packed : params.stride_LSE_partial_packed)(_, bidh, !is_varlen ? bidb : 0, !is_split ? 0 : split_idx);
        if (!LargeHeadDimV || warp_group_idx == 0) {
            if constexpr (!PackGQA) {
                #pragma unroll
                for (int mi = 0; mi < size(lse); ++mi) {
                    int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
                    if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o) { mLSE(row) = lse(mi); }
                }
            } else {
                PackGQA_t::store_LSE(mLSE, lse, tiled_mma, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
            }
        }

        // Store the normalized accumulator before the public fp16/bf16 cast.
        // Backward uses this optional state for the row-wise dot product dO * O.
        if (StoreOutputState && params.ptr_O_state != nullptr && !is_split) {
            if constexpr (!PackGQA) {
                Tensor mO_state = make_tensor(
                    make_gmem_ptr(params.ptr_O_state + offset_o * get<0>(params.stride_O_state_packed)),
                    params.shape_O_packed, params.stride_O_state_packed)(
                        _, _, bidh, !is_varlen ? bidb : 0, _0{});
                Tensor gO_state = local_tile(
                    mO_state, select<0, 1>(TileShape_MNK_PV{}),
                    make_coord(m_block, _0{}));
                static constexpr int kGmemElemsPerStoreState = 2;
                cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>
                    gmem_copy_state;
                Tensor tOrO_rowcol = make_tensor(
                    tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
                Tensor tOrO_copy = cute::tiled_divide(
                    tOrO_rowcol,
                    Shape<_1, Int<kGmemElemsPerStoreState>>{});
                Tensor tOgO_state = thread_mma.partition_C(gO_state);
                Tensor tOgO_state_rowcol = make_tensor(
                    tOgO_state.data(),
                    flash::convert_layout_acc_rowcol(tOgO_state.layout()));
                Tensor tOgO_state_copy = cute::tiled_divide(
                    tOgO_state_rowcol,
                    Shape<_1, Int<kGmemElemsPerStoreState>>{});
                Tensor taccOcO_col = taccOcO_rowcol(_0{}, _);
                #pragma unroll
                for (int m = 0; m < size(taccOcO_row); ++m) {
                    if (get<0>(taccOcO_row(m)) <
                        seqlen_o - m_block * kBlockM) {
                        #pragma unroll
                        for (int k = 0;
                             k < size(taccOcO_col) /
                                     kGmemElemsPerStoreState;
                             ++k) {
                            if (get<1>(taccOcO_col(
                                    k * kGmemElemsPerStoreState)) <
                                get<1>(params.shape_O)) {
                                cute::copy(
                                    gmem_copy_state, tOrO_copy(_, m, k),
                                    tOgO_state_copy(_, m, k));
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Write O from smem -> gmem
        if constexpr (Use_TMA_O) {
            Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh, bidb, split_idx);
            Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));  // (M, K)
            auto block_tma_O = params.tma_store_O.get_slice(_0{});
            Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
            Tensor tOsO = block_tma_O.partition_S(sO); // (TMA, TMA_M, TMA_K)
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                  cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (cute::elect_one_sync()) {
                    cute::copy(params.tma_store_O, tOsO, tOgO);
                    tma_store_arrive();
                    tma_store_wait<0>();
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                        shared_storage.pipelines.barrier_O.arrive(cta_id);
                    }
                }
            }
        } else {  // Bounded stores for variable-length sequences.
            if (!is_split) {
                Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !is_varlen ? bidb : 0, _0{});
                Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));  // (M, K)
                GmemTiledCopyO gmem_tiled_copy_O;
                auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
                Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
                Tensor tOrO = make_fragment_like(tOsO);
                cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
                if constexpr (ArchTag::kMinComputeCapability >= 90) {
                    cutlass::arch::fence_view_async_shared(); // ensure smem reads are done before next TMA to smem_v
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                        shared_storage.pipelines.barrier_O.arrive(cta_id);
                    }
                }
                if constexpr (!PackGQA) {
                    // (BLK_M,BLK_K) -> (blk_m,blk_k)
                    Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
                    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));
                    #pragma unroll
                    for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
                    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
                    // Preserve out-of-range global-memory destinations.
                    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
                    );
                } else {
                    // PackGQA shares output-pointer computation across each row.
                    PackGQA_t::store_O(mO, tOrO, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
                }
            } else {
                Tensor mOpartial = make_tensor(make_gmem_ptr(params.ptr_O_partial + offset_o * get<0>(params.stride_O_partial)), params.shape_O_packed, params.stride_O_partial_packed)(_, _, bidh, !is_varlen ? bidb : 0, split_idx);
                Tensor gOpartial = local_tile(mOpartial, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));  // (M, K)
                // The !Use_smem path has already arrived on barrier_O.
                if constexpr (Use_smem) {
                    if constexpr (ArchTag::kMinComputeCapability >= 90) {
                        #pragma unroll
                        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                            shared_storage.pipelines.barrier_O.arrive(cta_id);
                        }
                    }
                }
                if constexpr (!PackGQA) {
                    static constexpr int kGmemElemsPerStoreDirect = 2;
                    cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial> gmem_copy_direct;
                    // Reshape acc from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
                    Tensor tOrO_rowcol = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
                    Tensor tOrO_copy = cute::tiled_divide(tOrO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
                    Tensor tOgO = thread_mma.partition_C(gOpartial);
                    Tensor tOgO_rowcol = make_tensor(tOgO.data(), flash::convert_layout_acc_rowcol(tOgO.layout()));
                    Tensor tOgO_copy = cute::tiled_divide(tOgO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
                    Tensor taccOcO_col = taccOcO_rowcol(_0{}, _);
                    #pragma unroll
                    for (int m = 0; m < size(taccOcO_row); ++m) {
                        if (get<0>(taccOcO_row(m)) < seqlen_o - m_block * kBlockM) {
                            #pragma unroll
                            for (int k = 0; k < size(taccOcO_col) / kGmemElemsPerStoreDirect; ++k) {
                                if (get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect)) < get<1>(params.shape_O)) {
                                    cute::copy(gmem_copy_direct, tOrO_copy(_, m, k), tOgO_copy(_, m, k));
                                }
                            }
                        }
                    }
                } else {
                    PackGQApartial_t::store_O_direct(mOpartial, tOrO, tiled_mma, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
                }
            }
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        // The store method has already waited for TMA completion.
    }

    // Write 0 to output and -inf to LSE
    template <bool StoreOutputState=true, typename SharedStorage>
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         SharedStorage&,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        int num_splits = get<4>(params.shape_O_packed);
        if constexpr (Split && Varlen) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16; // first 16 bits are for num_splits
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            num_splits = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            split_idx &= 0x0000FFFF;  // Only use the lower 16 bits of split_idx
        }
        bool const is_split = !Split ? false : (!Varlen ? true : num_splits > 1);

        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
        bool const is_varlen = Varlen && params.cu_seqlens;
        int offset_o = seqlen_info.offset;
        int seqlen_o = seqlen_info.seqlen;
        int qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
        Tensor mLSE = make_tensor(make_gmem_ptr((!is_split ? params.ptr_LSE : params.ptr_LSE_partial) + offset_o * get<0>(!is_split ? params.stride_LSE : params.stride_LSE_partial)),
                                  params.shape_LSE_packed,
                                  !is_split ? params.stride_LSE_packed : params.stride_LSE_partial_packed)(_, bidh, !is_varlen ? bidb : 0, !is_split ? 0 : split_idx);
        Tensor gLSE = local_tile(mLSE, Shape<Int<kBlockM>>{}, make_coord(m_block));

        static_assert(kBlockM <= NumEpilogueThreads);
        if (thread_idx < kBlockM) {
            const int row = m_block * kBlockM + thread_idx;
            if constexpr (!PackGQA) {
                if (row < seqlen_o) { mLSE(row) = -INFINITY; }
            } else {
                if (row < seqlen_o * qhead_per_khead) {
                    int m_idx, h_idx;
                    m_idx = params.qhead_per_khead_divmod.divmod(h_idx, row);
                    // mLSE has shape ((qhead_per_khead, seqlen_q)) and it's unhappy with just 1 "make_coord"
                    mLSE(make_coord(make_coord(h_idx, m_idx))) = -INFINITY;
                }
            }
        }

        // The split-output combine kernel ignores O when LSE is -inf.
        if (!is_split) {
            Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !is_varlen ? bidb : 0, _0{});

            GmemTiledCopyO gmem_tiled_copy_O;
            auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
            Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
            if constexpr (!PackGQA) {
                Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
                #pragma unroll
                for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
                Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));  // (M, K)
                Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
                Tensor tOrO = make_fragment_like(tOgO);
                cute::clear(tOrO);
                // Preserve out-of-range global-memory destinations.
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
                );
            } else {
                // PackGQA shares output-pointer computation across each row.
                using PackGQA_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, Element>;
                Tensor tOrO = make_tensor<Element>(make_shape(Shape<_1, Int<kGmemElemsPerStore>>{}, size<1>(tOcO), size<2>(tOcO)));
                cute::clear(tOrO);
                PackGQA_t::store_O(mO, tOrO, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
            }

            if (StoreOutputState && params.ptr_O_state != nullptr) {
                if constexpr (!PackGQA) {
                    Tensor mO_state = make_tensor(
                        make_gmem_ptr(params.ptr_O_state +
                                      offset_o * get<0>(params.stride_O_state_packed)),
                        params.shape_O_packed,
                        params.stride_O_state_packed)(
                            _, _, bidh, !is_varlen ? bidb : 0, _0{});
                    #pragma unroll
                    for (int linear = thread_idx;
                         linear < kBlockM * kHeadDimV;
                         linear += NumEpilogueThreads) {
                        int const row_in_block = linear / kHeadDimV;
                        int const col = linear % kHeadDimV;
                        int const row = m_block * kBlockM + row_in_block;
                        if (row < seqlen_o &&
                            col < get<1>(params.shape_O)) {
                            mO_state(row, col) = 0.0f;
                        }
                    }
                }
            }
        }

    }

};

// Low-precision epilogue specialization with FP32 state stores disabled.
template <typename Base>
struct AttentionEpilogueWithoutOutputState : Base {
    static constexpr bool UseFastReciprocal = true;

    template <typename... Args>
    CUTLASS_DEVICE void store(Args&&... args) {
        Base::template store<false>(static_cast<Args&&>(args)...);
    }

    template <typename... Args>
    CUTLASS_DEVICE void store_zero(Args&&... args) {
        Base::template store_zero<false>(static_cast<Args&&>(args)...);
    }
};

template <class TileShape, class ClusterShape, class Element, class ArchTag,
          int NumThreads, bool Varlen, bool PackGQA, bool Split,
          bool FP8PermuteCol>
using CollectiveEpilogueFwdNoState = AttentionEpilogueWithoutOutputState<
    CollectiveEpilogueFwd<TileShape, ClusterShape, Element, ArchTag,
                         NumThreads, Varlen, PackGQA, Split, FP8PermuteCol>>;

} // namespace flash

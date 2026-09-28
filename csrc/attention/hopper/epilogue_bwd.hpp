/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Modified by: Shicheng Wen (xattn adaptations).

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/barrier.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "attention/sequence/seqlen.h"
#include "cuda/sync/named_barrier.hpp"
#include "attention/cuda/detail/cute_compat.h"
#include "hopper/cute_ext/bulk_reduce_add.hpp"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, int kHeadDimV_, class Element_, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool dKV_swapAB_, int AtomLayoutKdKV=1>
struct CollectiveEpilogueBwd {

    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool Use_TMA = !Varlen && ArchTag::kMinComputeCapability >= 90;
    static constexpr bool NeedsDQTailBarrier = false;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    using GmemTiledCopydKVTMA = cute::SM90_TMA_STORE;

    // Non-TMA output storage, including zero-fill.
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static_assert(kHeadDimV_ % kGmemElemsPerLoad == 0, "V headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kHeadDimV = kHeadDimV_;
    using TileShape_NK = decltype(select<1, 2>(TileShape_MNK{}));
    using TileShape_NK_V = Shape<Int<kBlockN>, Int<kHeadDimV>>;

    static constexpr int kGmemThreadsPerRowK = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
    static constexpr int kGmemThreadsPerRowV = cutlass::gcd(kHeadDimV / kGmemElemsPerLoad, NumEpilogueThreads);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRowK == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRowK");
    static_assert(NumEpilogueThreads % kGmemThreadsPerRowV == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRowV");
    using GmemLayoutAtomdK = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRowK>, Int<kGmemThreadsPerRowK>>,
                                   Stride<Int<kGmemThreadsPerRowK>, _1>>;
    using GmemLayoutAtomdV = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRowV>, Int<kGmemThreadsPerRowV>>,
                                   Stride<Int<kGmemThreadsPerRowV>, _1>>;
    using GmemTiledCopydK = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomdK{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store
    using GmemTiledCopydV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomdV{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomdKTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                          decltype(cute::get<1>(TileShape_MNK{})), Int<CUTE_STATIC_V(cute::get<2>(TileShape_MNK{})) / AtomLayoutKdKV>>());
    using SmemLayoutdKTMA = decltype(tile_to_shape(SmemLayoutAtomdKTMA{}, TileShape_NK{}));
    using SmemLayoutdKtTMA =
        decltype(cute::composition(SmemLayoutdKTMA{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));
    using SmemLayoutAtomdVTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                          Int<kBlockN>, Int<kHeadDimV / AtomLayoutKdKV>>());
    using SmemLayoutdVTMA = decltype(tile_to_shape(SmemLayoutAtomdVTMA{}, TileShape_NK_V{}));
    using SmemLayoutdVtTMA =
        decltype(cute::composition(SmemLayoutdVTMA{},
                                   make_layout(make_shape(Int<kHeadDimV>{}, Int<kBlockN>{}),
                                               make_stride(Int<kBlockN>{}, _1{}))));

    // Non-TMA shared-memory layouts.
    static constexpr int kBlockKSmemK = kHeadDim % 64 == 0 ? 64 : (kHeadDim % 32 == 0 ? 32 : 16);
    static constexpr int kSwizzleK = kBlockKSmemK == 64 ? 3 : (kBlockKSmemK == 32 ? 2 : 1);
    using SmemLayoutAtomdKSTG =
        decltype(composition(Swizzle<kSwizzleK, 3, 3>{},
                             Layout<Shape<Int<8>, Int<kBlockKSmemK>>,
                             Stride<Int<kBlockKSmemK>, _1>>{}));
    static constexpr int kBlockKSmemV = kHeadDimV % 64 == 0 ? 64 : (kHeadDimV % 32 == 0 ? 32 : 16);
    static constexpr int kSwizzleV = kBlockKSmemV == 64 ? 3 : (kBlockKSmemV == 32 ? 2 : 1);
    using SmemLayoutAtomdVSTG =
        decltype(composition(Swizzle<kSwizzleV, 3, 3>{},
                             Layout<Shape<Int<8>, Int<kBlockKSmemV>>,
                             Stride<Int<kBlockKSmemV>, _1>>{}));

    using SmemLayoutAtomdK = std::conditional_t<Use_TMA, SmemLayoutAtomdKTMA, SmemLayoutAtomdKSTG>;
    using SmemLayoutAtomdV = std::conditional_t<Use_TMA, SmemLayoutAtomdVTMA, SmemLayoutAtomdVSTG>;
    using SmemLayoutdK = decltype(tile_to_shape(SmemLayoutAtomdK{}, TileShape_NK{}));
    using SmemLayoutdV = decltype(tile_to_shape(SmemLayoutAtomdV{}, TileShape_NK_V{}));
    using SmemLayoutdKt =
        decltype(cute::composition(SmemLayoutdK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));
    using SmemLayoutdVt =
        decltype(cute::composition(SmemLayoutdV{},
                                   make_layout(make_shape(Int<kHeadDimV>{}, Int<kBlockN>{}),
                                               make_stride(Int<kBlockN>{}, _1{}))));

    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 90,
            std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
            AutoVectorizingCopyWithAssumedAlignment<128>
        >,
        Element>;

    static constexpr size_t SmemAlignmentdK = ArchTag::kMinComputeCapability >= 90 ? cutlass::detail::alignment_for_swizzle(SmemLayoutdK{}) : 128;
    static constexpr size_t SmemAlignmentdV = ArchTag::kMinComputeCapability >= 90 ? cutlass::detail::alignment_for_swizzle(SmemLayoutdV{}) : 128;
    static_assert(SmemAlignmentdK >= 128 && SmemAlignmentdV >= 128, "Require at least 128B alignment");
    static constexpr size_t SmemAlignmentdKV = SmemAlignmentdK >= SmemAlignmentdV ? SmemAlignmentdK : SmemAlignmentdV;

    struct TensorStorage : cute::aligned_struct<SmemAlignmentdKV> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>, SmemAlignmentdK> smem_dk;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>, SmemAlignmentdV> smem_dv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_k, d, head, batch)
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

    using TMA_dK = std::conditional_t<
        Use_TMA,
        decltype(make_tma_copy(
            GmemTiledCopydKVTMA{},
            make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
            SmemLayoutdKTMA{},
            TileShape_NK{},
            _1{})),  // no mcast for dK
        std::nullptr_t
        >;
    using TMA_dV = std::conditional_t<
        Use_TMA,
        decltype(make_tma_copy(
            GmemTiledCopydKVTMA{},
            make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
            SmemLayoutdVTMA{},
            TileShape_NK_V{},
            _1{})),  // no mcast for dKV
        std::nullptr_t
        >;

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        ShapedKV const shape_dV;
        StridedKV const stride_dV;
        int const num_batch;
        int const num_heads_q;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens;
        int const* seqused;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        ShapedKV const shape_dV;
        StridedKV const stride_dV;
        TMA_dK tma_store_dK;
        TMA_dV tma_store_dV;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
        Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dV, args.stride_dV);
        TMA_dK tma_store_dK = [&] {
            if constexpr (Use_TMA) {
                return make_tma_copy(GmemTiledCopydKVTMA{}, mdK, SmemLayoutdKTMA{}, TileShape_NK{}, _1{}); // no mcast for dK
            } else {
                return nullptr;
            }
        }();
        TMA_dV tma_store_dV = [&] {
            if constexpr (Use_TMA) {
                return make_tma_copy(GmemTiledCopydKVTMA{}, mdV, SmemLayoutdVTMA{}, TileShape_NK_V{}, _1{}); // no mcast for dV
            } else {
                return nullptr;
            }
        }();
        return {args.ptr_dK, args.shape_dK, args.stride_dK, args.ptr_dV, args.shape_dV, args.stride_dV,
                tma_store_dK, tma_store_dV, args.cu_seqlens, args.seqused};
    }

    /// Single-thread TMA descriptor prefetch.
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA) {
            cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
        }
    }

    template <typename SharedStorage, typename FrgTensorK, typename FrgTensorV,
              typename TiledMmaK, typename TiledMmaV>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorK const& tdKrdK,
          FrgTensorV const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMmaK tiled_mma_dK,
          TiledMmaV tiled_mma_dV,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb] = block_coord;
        Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdK{}));
        Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdV{}));
        Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKt{}));
        Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdVt{}));
        auto smem_tiled_copy_dK = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma_dK);
        auto smem_thr_copy_dK = smem_tiled_copy_dK.get_thread_slice(thread_idx);
        auto smem_tiled_copy_dV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma_dV);
        auto smem_thr_copy_dV = smem_tiled_copy_dV.get_thread_slice(thread_idx);

        Tensor tdVrdV_out = make_tensor_like<Element>(tdVrdV);
        flash::convert_type_out(tdVrdV, tdVrdV_out);
        Tensor tdKrdK_out = make_tensor_like<Element>(tdKrdK);
        flash::convert_type_out(tdKrdK, tdKrdK_out);
        Tensor taccdKrdK = smem_thr_copy_dK.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdKsdK = smem_thr_copy_dK.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // K/V shared-memory read completion.
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        cute::copy(smem_tiled_copy_dV, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dK, taccdKrdK, taccdKsdK);
        if constexpr (Use_TMA) {
            cutlass::arch::fence_view_async_shared(); // Shared-memory visibility for TMA.
            cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

            Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK);
            Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dV);
            Tensor gdK = local_tile(mdK(_, _, bidh, bidb), TileShape_NK{}, make_coord(n_block, _0{}));  // (M, K)
            Tensor gdV = local_tile(mdV(_, _, bidh, bidb), TileShape_NK_V{}, make_coord(n_block, _0{}));  // (M, K)
            auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
            auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
            Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
            Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
            Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (cute::elect_one_sync()) {
                    cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
                    cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
                    tma_store_arrive();
                }
            }
            tma_store_wait<0>();

        } else {
            flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            flash::SeqlenInfo<Varlen, kBlockN> seqlen_info{bidb, size<0>(params.shape_dK), params.cu_seqlens, params.seqused};
            bool const is_varlen = Varlen && params.cu_seqlens;
            Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdK = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdK), TileShape_NK{}, make_coord(n_block, _0{}));  // (M, K)
            Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdV = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdV), TileShape_NK_V{}, make_coord(n_block, _0{}));  // (M, K)

            GmemTiledCopydK gmem_tiled_copy_dK;
            auto gmem_thr_copy_dK = gmem_tiled_copy_dK.get_thread_slice(thread_idx);
            GmemTiledCopydV gmem_tiled_copy_dV;
            auto gmem_thr_copy_dV = gmem_tiled_copy_dV.get_thread_slice(thread_idx);
            Tensor tdKgdK = gmem_thr_copy_dK.partition_D(gdK);
            Tensor tdKsdK = gmem_thr_copy_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdVgdV = gmem_thr_copy_dV.partition_D(gdV);
            Tensor tdVsdV = gmem_thr_copy_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            Tensor tdKrdK = make_fragment_like(tdKgdK);
            Tensor tdVrdV = make_fragment_like(tdVgdV);
            Tensor cdK = cute::make_identity_tensor(TileShape_NK{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
            Tensor cdV = cute::make_identity_tensor(TileShape_NK_V{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
            Tensor tdKcdK = gmem_thr_copy_dK.partition_D(cdK);
            Tensor tdVcdV = gmem_thr_copy_dV.partition_D(cdV);
            Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
            Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVgdV)));
            #pragma unroll
            for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
            #pragma unroll
            for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(_0{}, _0{}, k)) < get<1>(params.shape_dV); }
            // Partial kBlockN tiles require shared-memory bounds checks.
            static constexpr bool EvenNK = kBlockN % CUTE_STATIC_V(size<0>(GmemLayoutAtomdK{})) == 0;
            static constexpr bool EvenNV = kBlockN % CUTE_STATIC_V(size<0>(GmemLayoutAtomdV{})) == 0;
            flash::copy</*Is_even_MN=*/EvenNV, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
                gmem_tiled_copy_dV, tdVsdV, tdVrdV, tdVcdV, tdVpdV, kBlockN);
            flash::copy</*Is_even_MN=*/EvenNK, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
                gmem_tiled_copy_dK, tdKsdK, tdKrdK, tdKcdK, tdKpdK, kBlockN);
            // gdKV identity layout; Clear_OOB_K=false preserves out-of-range
            // global values.
            int const n_valid = seqlen_info.seqlen - n_block * int(kBlockN);
            int const n_limit = n_valid < int(kBlockN) ? n_valid : int(kBlockN);
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, n_limit
            );
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dK, tdKrdK, tdKgdK, tdKcdK, tdKpdK, n_limit
            );
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
    }

    // Zero dK/dV.
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        auto [n_block, bidh, bidb] = block_coord;
        flash::SeqlenInfo<Varlen, kBlockN> seqlen_info{bidb, size<0>(params.shape_dK), params.cu_seqlens, params.seqused};
        bool const is_varlen = Varlen && params.cu_seqlens;
        Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdK), TileShape_NK{}, make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdV), TileShape_NK_V{}, make_coord(n_block, _0{}));  // (M, K)

        GmemTiledCopydK gmem_tiled_copy_dK;
        auto gmem_thr_copy_dK = gmem_tiled_copy_dK.get_thread_slice(thread_idx);
        GmemTiledCopydV gmem_tiled_copy_dV;
        auto gmem_thr_copy_dV = gmem_tiled_copy_dV.get_thread_slice(thread_idx);
        Tensor tdKgdK = gmem_thr_copy_dK.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dV.partition_D(gdV);
        Tensor tdKrdK = make_fragment_like(tdKgdK);
        Tensor tdVrdV = make_fragment_like(tdVgdV);
        clear(tdKrdK);
        clear(tdVrdV);
        // gdKV identity layout.
        Tensor cdK = cute::make_identity_tensor(TileShape_NK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor cdV = cute::make_identity_tensor(TileShape_NK_V{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dK.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVgdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(_0{}, _0{}, k)) < get<1>(params.shape_dV); }
        // Clear_OOB_K=false preserves out-of-range global values.
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dK, tdKrdK, tdKgdK, tdKcdK, tdKpdK, seqlen_info.seqlen - n_block * kBlockN
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, seqlen_info.seqlen - n_block * kBlockN
        );
    }

};

// Decode each batch group into final or private dK/dV destinations while
// reusing the direct-store epilogue.
template <class TileShape_MNK_, int kHeadDimV_, class Element_, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool dKV_swapAB_,
          int AtomLayoutKdKV = 1>
struct CollectiveEpilogueBwdSplitBatchDestinations
    : CollectiveEpilogueBwd<
          TileShape_MNK_, kHeadDimV_, Element_, ArchTag_,
          NumEpilogueThreads_, Varlen_, dKV_swapAB_, AtomLayoutKdKV> {
    using Base = CollectiveEpilogueBwd<
        TileShape_MNK_, kHeadDimV_, Element_, ArchTag_,
        NumEpilogueThreads_, Varlen_, dKV_swapAB_, AtomLayoutKdKV>;
    using BaseArguments = typename Base::Arguments;
    using BaseParams = typename Base::Params;

    struct Arguments {
        BaseArguments primary;
        BaseArguments secondary;
    };

    struct Params {
        BaseParams primary;
        BaseParams secondary;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {
            Base::to_underlying_arguments(args.primary),
            Base::to_underlying_arguments(args.secondary)};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        Base::prefetch_tma_descriptors(params.primary);
        Base::prefetch_tma_descriptors(params.secondary);
    }

    CUTLASS_DEVICE
    static int destination_batch_count(Params const& params) {
        return int(get<3>(params.primary.shape_dK));
    }

    template <typename SharedStorage, typename FrgTensorK,
              typename FrgTensorV, typename TiledMmaK, typename TiledMmaV>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorK const& tdKrdK,
          FrgTensorV const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMmaK tiled_mma_dK,
          TiledMmaV tiled_mma_dV,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
        auto [n_block, bidh, encoded_bidb] = block_coord;
        int const batch_count = destination_batch_count(params);
        bool const use_secondary = encoded_bidb >= batch_count;
        BaseParams const& destination =
            use_secondary ? params.secondary : params.primary;
        cute::tuple<int32_t, int32_t, int32_t> destination_coord{
            n_block, bidh,
            use_secondary ? encoded_bidb - batch_count : encoded_bidb};
        Base::store(destination, tdKrdK, tdVrdV, shared_storage,
                    tiled_mma_dK, tiled_mma_dV, thread_idx,
                    destination_coord);
    }

    CUTLASS_DEVICE void
    store_zero(
        Params const& params,
        int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
        auto [n_block, bidh, encoded_bidb] = block_coord;
        int const batch_count = destination_batch_count(params);
        bool const use_secondary = encoded_bidb >= batch_count;
        BaseParams const& destination =
            use_secondary ? params.secondary : params.primary;
        cute::tuple<int32_t, int32_t, int32_t> destination_coord{
            n_block, bidh,
            use_secondary ? encoded_bidb - batch_count : encoded_bidb};
        Base::store_zero(destination, thread_idx, destination_coord);
    }
};

template <class TileShape_MNK_, int kHeadDimV_, class ElementAccum,
          class ArchTag_, int NumEpilogueThreads_, bool Varlen_,
          bool Deterministic, bool dKV_swapAB_>
struct CollectiveEpilogueBwdGQA {

    using TileShape_MNK = TileShape_MNK_;
    using Element = ElementAccum;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool Use_TMA = ArchTag::kMinComputeCapability >= 90;
    static_assert(
        !Deterministic || Use_TMA,
        "Deterministic grouped attention BWD requires semaphore-ordered "
        "TMA reductions instead of unordered floating-point atomicAdd");

    static_assert(ArchTag::kMinComputeCapability >= 80);

    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kMaxHeadDim =
        kHeadDim >= kHeadDimV ? kHeadDim : kHeadDimV;
    // Mixed-D/V GQA/MQA FP32 epilogue storage aliases smem_dqacc.
    static constexpr bool NeedsDQTailBarrier =
        Use_TMA && kHeadDim != kHeadDimV;
    using TileShape_NK = Shape<Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_NV = Shape<Int<kBlockN>, Int<kHeadDimV>>;
    using SmemLayoutdK = Layout<
        Shape<Int<kBlockN>, Int<kHeadDim>>,
        Stride<Int<kHeadDim>, _1>>;
    using SmemLayoutdV = Layout<
        Shape<Int<kBlockN>, Int<kHeadDimV>>,
        Stride<Int<kHeadDimV>, _1>>;
    struct TensorStorageTMA : cute::aligned_struct<128> {
        cute::array_aligned<ElementAccum, kBlockN * kMaxHeadDim, 128>
            smem_dkv;
    };
    struct TensorStorageSTG {
        cute::array<ElementAccum, 0> smem_dkv;
    };
    using TensorStorage =
        std::conditional_t<Use_TMA, TensorStorageTMA, TensorStorageSTG>;

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using TMA_dK = std::conditional_t<
        Use_TMA,
        decltype(make_tma_copy(
            cute::SM90_TMA_REDUCE_ADD{},
            make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)),
                        ShapedKV{}, StridedKV{}),
            SmemLayoutdK{}, TileShape_NK{}, _1{})),
        std::nullptr_t>;
    using TMA_dV = std::conditional_t<
        Use_TMA,
        decltype(make_tma_copy(
            cute::SM90_TMA_REDUCE_ADD{},
            make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)),
                        ShapedKV{}, StridedKV{}),
            SmemLayoutdV{}, TileShape_NV{}, _1{})),
        std::nullptr_t>;

    // Host side kernel arguments
    struct Arguments {
        ElementAccum* ptr_dKaccum;
        ShapedKV const shape_dKaccum;
        StridedKV const stride_dKaccum;
        ElementAccum* ptr_dVaccum;
        ShapedKV const shape_dVaccum;
        StridedKV const stride_dVaccum;
        int const num_batch;
        int const num_heads_q;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens;
        int const* seqused;
    };

    // Device side kernel params
    struct Params {
        ElementAccum* ptr_dKaccum;
        ShapedKV const shape_dKaccum;
        StridedKV const stride_dKaccum;
        ElementAccum* ptr_dVaccum;
        ShapedKV const shape_dVaccum;
        StridedKV const stride_dVaccum;
        TMA_dK tma_reduce_dK;
        TMA_dV tma_reduce_dV;
        cutlass::FastDivmod qhead_per_khead_divmod;
        int* dk_semaphore;
        int* dv_semaphore;
        int const num_batch;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Deterministic) {
            assert(args.dk_semaphore != nullptr);
            assert(args.dv_semaphore != nullptr);
        }
        Tensor mdKaccum = make_tensor(make_gmem_ptr(args.ptr_dKaccum),
                                      args.shape_dKaccum,
                                      args.stride_dKaccum);
        Tensor mdVaccum = make_tensor(make_gmem_ptr(args.ptr_dVaccum),
                                      args.shape_dVaccum,
                                      args.stride_dVaccum);
        TMA_dK tma_reduce_dK = [&] {
            if constexpr (Use_TMA) {
                return make_tma_copy(cute::SM90_TMA_REDUCE_ADD{}, mdKaccum,
                                     SmemLayoutdK{}, TileShape_NK{}, _1{});
            } else {
                return nullptr;
            }
        }();
        TMA_dV tma_reduce_dV = [&] {
            if constexpr (Use_TMA) {
                return make_tma_copy(cute::SM90_TMA_REDUCE_ADD{}, mdVaccum,
                                     SmemLayoutdV{}, TileShape_NV{}, _1{});
            } else {
                return nullptr;
            }
        }();
        return {args.ptr_dKaccum, args.shape_dKaccum, args.stride_dKaccum, args.ptr_dVaccum, args.shape_dVaccum, args.stride_dVaccum,
                tma_reduce_dK, tma_reduce_dV,
                cutlass::FastDivmod(args.num_heads_q / get<2>(args.shape_dKaccum)),
                args.dk_semaphore, args.dv_semaphore,
                args.num_batch, args.cu_seqlens, args.seqused};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA) {
            cute::prefetch_tma_descriptor(
                params.tma_reduce_dK.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                params.tma_reduce_dV.get_tma_descriptor());
        }
    }

    template <typename SharedStorage, typename FrgTensorK, typename FrgTensorV,
              typename TiledMmaK, typename TiledMmaV>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorK const& tdKrdK,
          FrgTensorV const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMmaK tiled_mma_dK,
          TiledMmaV tiled_mma_dV,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb] = block_coord;
        int bidh_idx_in_group;
        int bidh_kv = params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh);
        flash::SeqlenInfo<Varlen, kBlockN> seqlen_info{bidb, size<0>(params.shape_dKaccum), params.cu_seqlens, params.seqused};
        bool const is_varlen = Varlen && params.cu_seqlens;
        Tensor mdKaccum = make_tensor(
            make_gmem_ptr(params.ptr_dKaccum), params.shape_dKaccum,
            params.stride_dKaccum)(_, _, bidh_kv, !is_varlen ? bidb : 0);
        Tensor mdVaccum = make_tensor(
            make_gmem_ptr(params.ptr_dVaccum), params.shape_dVaccum,
            params.stride_dVaccum)(_, _, bidh_kv, !is_varlen ? bidb : 0);
        Tensor gdKaccum = local_tile(
            domain_offset(make_coord(seqlen_info.offset, _0{}), mdKaccum),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            make_coord(n_block, _0{}));
        Tensor gdVaccum = local_tile(
            domain_offset(make_coord(seqlen_info.offset, _0{}), mdVaccum),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
            make_coord(n_block, _0{}));

        auto thr_mma_dK = tiled_mma_dK.get_thread_slice(thread_idx);
        auto thr_mma_dV = tiled_mma_dV.get_thread_slice(thread_idx);
        auto cK = cute::make_identity_tensor(
            select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(
                TileShape_MNK{}));
        using TileShape_MNK_V = cute::Shape<
            decltype(get<0>(TileShape_MNK{})),
            decltype(get<1>(TileShape_MNK{})), Int<kHeadDimV>>;
        auto cV = cute::make_identity_tensor(
            select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(
                TileShape_MNK_V{}));
        auto tdKcK = thr_mma_dK.partition_C(cK);
        auto tdVcV = thr_mma_dV.partition_C(cV);

        // K/V shared-memory lifetime extends through dQ finalization.
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        int const num_batch = params.num_batch;
        int const num_head_kv = get<2>(params.shape_dKaccum);
        int const flag_idx = n_block * num_batch * num_head_kv;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
        int *lock_ptr = !Deterministic
            ? nullptr
            : params.dv_semaphore + bidb * num_head_kv + bidh_kv;
        if constexpr (Use_TMA) {
            Tensor sdV = make_tensor(
                make_smem_ptr(
                    shared_storage.tensors.epilogue.smem_dkv.data()),
                SmemLayoutdV{});
            #pragma unroll
            for (int i = 0; i < size(tdVrdV); ++i) {
                int const axis0 = get<0>(tdVcV(i));
                int const axis1 = get<1>(tdVcV(i));
                int const row = !dKV_swapAB ? axis0 : axis1;
                int const col = !dKV_swapAB ? axis1 : axis0;
                sdV(row, col) =
                    n_block * kBlockN + row < seqlen_info.seqlen &&
                            col < kHeadDimV
                        ? tdVrdV(i)
                        : ElementAccum(0);
            }
            cutlass::arch::fence_view_async_shared();
            flash::named_barrier_sync(
                NumEpilogueThreads,
                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

            if constexpr (Deterministic) {
                if (thread_idx < cutlass::NumThreadsPerWarp) {
                    Barrier::wait_eq(lock_ptr, thread_idx, flag_idx,
                                     bidh_idx_in_group);
                }
                flash::named_barrier_sync(
                    NumEpilogueThreads,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            }

            Tensor mdVaccumTma = params.tma_reduce_dV.get_tma_tensor(
                params.shape_dVaccum);
            Tensor gdVaccumTma = local_tile(
                domain_offset(
                    make_coord(seqlen_info.offset, _0{}),
                    mdVaccumTma(_, _, bidh_kv, !is_varlen ? bidb : 0)),
                TileShape_NV{}, make_coord(n_block, _0{}));
            auto block_tma_dV = params.tma_reduce_dV.get_slice(_0{});
            Tensor tdVgdVaccum = block_tma_dV.partition_D(gdVaccumTma);
            Tensor tdVsdVaccum = block_tma_dV.partition_S(sdV);
            if (thread_idx == 0) {
                cute::copy(params.tma_reduce_dV, tdVsdVaccum,
                           tdVgdVaccum);
                tma_store_arrive();
                tma_store_wait<0>();
            }
            flash::named_barrier_sync(
                NumEpilogueThreads,
                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            if constexpr (Deterministic) {
                if (thread_idx < cutlass::NumThreadsPerWarp) {
                    Barrier::arrive_inc(lock_ptr, thread_idx, flag_idx);
                }
            }
        } else {
            if constexpr (Deterministic) {
                if (thread_idx < cutlass::NumThreadsPerWarp) {
                    Barrier::wait_eq(lock_ptr, thread_idx, flag_idx,
                                     bidh_idx_in_group);
                }
                flash::named_barrier_sync(
                    NumEpilogueThreads,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            }
            #pragma unroll
            for (int i = 0; i < size(tdVrdV); ++i) {
                int const axis0 = get<0>(tdVcV(i));
                int const axis1 = get<1>(tdVcV(i));
                int const row = !dKV_swapAB ? axis0 : axis1;
                int const col = !dKV_swapAB ? axis1 : axis0;
                if (n_block * kBlockN + row < seqlen_info.seqlen &&
                    col < kHeadDimV) {
                    atomicAdd(&gdVaccum(row, col), tdVrdV(i));
                }
            }
            if constexpr (Deterministic) {
                flash::named_barrier_sync(
                    NumEpilogueThreads,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (thread_idx < cutlass::NumThreadsPerWarp) {
                    Barrier::arrive_inc(lock_ptr, thread_idx, flag_idx);
                }
            }
        }

        lock_ptr = !Deterministic ? nullptr : params.dk_semaphore + bidb * num_head_kv + bidh_kv;
        if constexpr (Use_TMA) {
            Tensor sdK = make_tensor(
                make_smem_ptr(
                    shared_storage.tensors.epilogue.smem_dkv.data()),
                SmemLayoutdK{});
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) {
                int const axis0 = get<0>(tdKcK(i));
                int const axis1 = get<1>(tdKcK(i));
                int const row = !dKV_swapAB ? axis0 : axis1;
                int const col = !dKV_swapAB ? axis1 : axis0;
                sdK(row, col) =
                    n_block * kBlockN + row < seqlen_info.seqlen &&
                            col < kHeadDim
                        ? tdKrdK(i)
                        : ElementAccum(0);
            }
            cutlass::arch::fence_view_async_shared();
            flash::named_barrier_sync(
                NumEpilogueThreads,
                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

            if constexpr (Deterministic) {
                if (thread_idx < cutlass::NumThreadsPerWarp) {
                    Barrier::wait_eq(lock_ptr, thread_idx, flag_idx,
                                     bidh_idx_in_group);
                }
                flash::named_barrier_sync(
                    NumEpilogueThreads,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            }

            Tensor mdKaccumTma = params.tma_reduce_dK.get_tma_tensor(
                params.shape_dKaccum);
            Tensor gdKaccumTma = local_tile(
                domain_offset(
                    make_coord(seqlen_info.offset, _0{}),
                    mdKaccumTma(_, _, bidh_kv, !is_varlen ? bidb : 0)),
                TileShape_NK{}, make_coord(n_block, _0{}));
            auto block_tma_dK = params.tma_reduce_dK.get_slice(_0{});
            Tensor tdKgdKaccum = block_tma_dK.partition_D(gdKaccumTma);
            Tensor tdKsdKaccum = block_tma_dK.partition_S(sdK);
            if (thread_idx == 0) {
                cute::copy(params.tma_reduce_dK, tdKsdKaccum,
                           tdKgdKaccum);
                tma_store_arrive();
                tma_store_wait<0>();
            }
            if constexpr (Deterministic) {
                flash::named_barrier_sync(
                    NumEpilogueThreads,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (thread_idx < cutlass::NumThreadsPerWarp) {
                    Barrier::arrive_inc(lock_ptr, thread_idx, flag_idx);
                }
            }
        } else {
            if constexpr (Deterministic) {
                if (thread_idx < cutlass::NumThreadsPerWarp) {
                    Barrier::wait_eq(lock_ptr, thread_idx, flag_idx,
                                     bidh_idx_in_group);
                }
                flash::named_barrier_sync(
                    NumEpilogueThreads,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            }
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) {
                int const axis0 = get<0>(tdKcK(i));
                int const axis1 = get<1>(tdKcK(i));
                int const row = !dKV_swapAB ? axis0 : axis1;
                int const col = !dKV_swapAB ? axis1 : axis0;
                if (n_block * kBlockN + row < seqlen_info.seqlen &&
                    col < kHeadDim) {
                    atomicAdd(&gdKaccum(row, col), tdKrdK(i));
                }
            }
            if constexpr (Deterministic) {
                flash::named_barrier_sync(
                    NumEpilogueThreads,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (thread_idx < cutlass::NumThreadsPerWarp) {
                    Barrier::arrive_inc(lock_ptr, thread_idx, flag_idx);
                }
            }
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
    }

    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        // dKaccum and dVaccum are zero-initialized.
    }

};

} // namespace flash

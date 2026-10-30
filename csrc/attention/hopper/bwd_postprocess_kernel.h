/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/arch/barrier.h"

#include "attention/sequence/seqlen.h"
#include "attention/cuda/detail/cute_compat.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class Element, class ElementAccum, class ArchTag_, int kNThreads, class TiledMma, bool dQ_swapAB, bool SplitDQ=false, bool NegQReaders=false>
class AttentionBwdPostprocessDQ {

public:

    // Type Aliases
    using TileShape_MK = TileShape_MK_;
    using ArchTag = ArchTag_;

    static_assert(ArchTag::kMinComputeCapability >= 75);
    static constexpr bool IsSm90 = ArchTag::kMinComputeCapability >= 90;

    static constexpr uint32_t MaxThreadsPerBlock = kNThreads;

    static constexpr int kBlockM = get<0>(TileShape_MK{});
    static constexpr int kHeadDim = get<1>(TileShape_MK{});
    static_assert(!IsSm90 || kNThreads % cutlass::NumThreadsPerWarpGroup == 0, "kNThreads must be a multiple of NumThreadsPerWarpGroup");
    static constexpr int NumdQWarpGgroups = kNThreads / cutlass::NumThreadsPerWarpGroup;
    using R2SLayoutAtomdQaccum = std::conditional_t<
        IsSm90,
        Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumdQWarpGgroups>>>,
        Layout<Shape<Int<kNThreads>>>
    >;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape<Int<IsSm90 ? 4 : 1>>>{}));  // Val layout, 1 or 4 vals per read
    using G2SLayoutAtomdQaccum = Layout<Shape<Int<kNThreads>>>;
    // Synchronous global-to-shared copy.
    using G2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, ElementAccum>{}, G2SLayoutAtomdQaccum{},
                                                         Layout<Shape<_4>>{}));  // Val layout, 4 vals per read
    // Global-to-shared loads require full tiles.
    static_assert(IsSm90 || (kBlockM * kHeadDim) % (kNThreads * 4) == 0);
    static constexpr int SmemdQaccumSize = size(TileShape_MK{});
    using SmemLayoutdQaccumFlat = Layout<Shape<Int<SmemdQaccumSize>>>;
    using SmemLayoutdQaccum = std::conditional_t<
        IsSm90,
        Layout<Shape<Int<kBlockM * kHeadDim / NumdQWarpGgroups>, Int<NumdQWarpGgroups>>>,
        Layout<Shape<Int<kBlockM * kHeadDim>>>
    >;

    // Shared-memory K width is divisible by the per-warpgroup MMA width.
    static constexpr int MmaShapeN = get<1>(typename TiledMma::AtomShape_MNK{});
    static constexpr int kBlockKSmem = MmaShapeN % 64 == 0 ? 64 : (MmaShapeN % 32 == 0 ? 32 : 16);
    static constexpr int kSwizzle = kBlockKSmem == 64 ? 3 : (kBlockKSmem == 32 ? 2 : 1);
    using SmemLayoutAtomdQ =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                 Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                 Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(SmemLayoutAtomdQ{}, TileShape_MK{}));
    using SmemLayoutdQt =
        decltype(cute::composition(SmemLayoutdQ{},
                                   make_layout(make_shape(get<1>(TileShape_MK{}), get<0>(TileShape_MK{})),
                                               make_stride(Int<get<0>(TileShape_MK{})>{}, _1{}))));

    using SmemCopyAtomdQ = Copy_Atom<
        std::conditional_t<
            IsSm90 && sizeof(Element) == 2,
            std::conditional_t<!dQ_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
            AutoVectorizingCopyWithAssumedAlignment<128>
        >,
        Element>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, int(MaxThreadsPerBlock));
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    struct SharedStorage : cute::aligned_struct<128> {
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccum>> smem_dqacc;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdQ>> smem_dq;
        alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_dQaccum;
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    // Use a one-CTA launch bound above 48 KiB of shared storage. The negative-query
    // fold's dK registers would otherwise halve the occupancy of the readers'
    // launch: two CTAs whenever two fit.
    static constexpr uint32_t MinBlocksPerMultiprocessor =
        NegQReaders && SharedStorageSize <= 113 * 1024 ? 2
        : SharedStorageSize > 48 * 1024 ? 1 : 2;

    using ShapedQ = cute::Shape<int32_t, int32_t, int32_t, int32_t>;   // (seqlen_q, d, head, batch)
    using StridedQ = cute::Stride<int64_t, _1, int64_t, int64_t>;
    // int64 extent: varlen workspaces can exceed 2^31 / d padded rows times d.
    using ShapedQaccum = cute::Shape<int64_t, int32_t, int32_t>;  // (seqlen_q * d, head, batch)
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    // Negative-query readers (softdelta): dQaccum head l = g * readers + r. r > 0
    // writes dQ head g * (readers - 1) + r - 1; r = 0 writes head g of dNegQ, plus
    // the FP32 dK accumulator rows (same head and rows, rounded once) when
    // ptr_dK_accum is set; dNegQ is then dK itself.
    struct NegQArgs {
        Element* ptr_dNegQ = nullptr;
        StridedQ stride_dNegQ = {};
        float const* ptr_dK_accum = nullptr;
        StridedQ stride_dK_accum = {};
        int readers = 1;
    };

    // Device side arguments
    struct Arguments {
        ElementAccum const* ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        Element* ptr_dQ;
        ShapedQ const shape_dQ;
        StridedQ const stride_dQ;
        float const softmax_scale;
        int const num_batch;
        int const num_m_blocks;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        int const dq_slots = 1;
        int64_t const dq_slot_stride = 0;
        int const* dq_exponent = nullptr;  // fixed-point deterministic dQ
        NegQArgs neg_q = {};
    };

    // Kernel entry point API
    struct Params {
        ElementAccum const* ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        Element* ptr_dQ;
        ShapedQ const shape_dQ;
        StridedQ const stride_dQ;
        float const softmax_scale;
        int const num_batch;
        int const num_m_blocks;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        int const dq_slots = 1;
        int64_t const dq_slot_stride = 0;
        int const* dq_exponent = nullptr;  // fixed-point deterministic dQ
        NegQArgs neg_q = {};
    };

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        return {
            args.ptr_dQaccum,
            args.shape_dQaccum,
            args.stride_dQaccum,
            args.ptr_dQ,
            args.shape_dQ,
            args.stride_dQ,
            args.softmax_scale,
            args.num_batch,
            args.num_m_blocks,
            args.cu_seqlens,
            args.seqused,
            args.dq_slots,
            args.dq_slot_stride,
            args.dq_exponent,
            args.neg_q
        };
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {
        // Batches past the grid z limit are strided: blockIdx.z, + gridDim.z, ...
        for (int bidb = int(blockIdx.z); bidb < params.num_batch; bidb += int(gridDim.z)) {
            if (bidb != int(blockIdx.z)) { __syncthreads(); }  // shared memory reuse
            run_tile(params, smem_buf, int(blockIdx.x), int(blockIdx.y), bidb);
        }
    }

    CUTLASS_DEVICE
    void
    run_tile(Params const& params, char* smem_buf, int const m_block, int const bidh, int const bidb) {

        static constexpr int kBlockM = get<0>(TileShape_MK{});
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        Tensor sdQaccum = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccum{});
        Tensor sdQaccum_flat = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccumFlat{});
        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.smem_dq.data()), SmemLayoutdQ{});
        Tensor sdQt = make_tensor(make_smem_ptr(shared_storage.smem_dq.data()), SmemLayoutdQt{});

        int const thread_idx = threadIdx.x;

        flash::SeqlenInfo<true /*Varlen*/, kBlockM> seqlen_info(bidb, size<0>(params.shape_dQ), params.cu_seqlens, params.seqused);
        bool const is_varlen = params.cu_seqlens;
        if (is_varlen && m_block * kBlockM >= seqlen_info.seqlen) { return; }

        // Step 1: load dQaccum from gmem to smem
        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum const*>(params.ptr_dQaccum)),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdQaccum = local_tile(domain_offset(make_coord(int64_t(seqlen_info.offset_padded) * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(m_block));  // (M * K)
        // Reduce distinct key-tile partial slots in fixed order before conversion.
        if constexpr (SplitDQ) {
            if (params.dq_slots > 1) {
                auto const* src = reinterpret_cast<float4 const*>(
                    raw_pointer_cast(gdQaccum.data()));
                auto* dst = reinterpret_cast<float4*>(
                    raw_pointer_cast(sdQaccum_flat.data()));
                for (int i = thread_idx; i < kBlockM * kHeadDim / 4;
                     i += MaxThreadsPerBlock) {
                    float4 sum = src[i];
                    for (int slot = 1; slot < params.dq_slots; ++slot) {
                        float4 value =
                            src[i + int64_t(slot) * params.dq_slot_stride / 4];
                        sum.x += value.x;
                        sum.y += value.y;
                        sum.z += value.z;
                        sum.w += value.w;
                    }
                    dst[i] = sum;
                }
                __syncthreads();
            }
        }
        if (!SplitDQ || params.dq_slots == 1) {
        if constexpr (IsSm90) {  // Use BulkCopy
            static constexpr uint32_t TmaTransactionBytesdQaccum = static_cast<uint32_t>(size(SmemLayoutdQaccumFlat{}) * cute::sizeof_bits_v<ElementAccum> / 8);
            auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
            if (thread_idx == 0) {
                shared_storage.barrier_dQaccum.init(1 /*numThreads*/);
                shared_storage.barrier_dQaccum.arrive_and_expect_tx(TmaTransactionBytesdQaccum);
                copy(bulk_copy.with(*reinterpret_cast<uint64_t*>(&shared_storage.barrier_dQaccum)), gdQaccum, sdQaccum_flat);
            }
            __syncthreads();
            shared_storage.barrier_dQaccum.wait(0);
        } else {
            G2STiledCopydQaccum g2s_tiled_copy_dQaccum;
            auto g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
            Tensor tdQgdQaccumg2s = g2s_thr_copy_dQaccum.partition_S(gdQaccum);
            Tensor tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum);
            cute::copy(g2s_tiled_copy_dQaccum, tdQgdQaccumg2s, tdQsdQaccumg2s);
            __syncthreads();
        }
        }


        // Step 2: Load dQaccum from smem to register, then convert fp32 -> fp16/bf16
        R2STiledCopydQaccum s2r_tiled_copy_dQaccum;
        auto s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        Tensor tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum);
        TiledMma tiled_mma_dQ;
        Tensor taccdQrdQaccum = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 1, !dQ_swapAB ? 1 : 0>(TileShape_MK{}));
        CUTE_STATIC_ASSERT_V(size(taccdQrdQaccum) == size(tdQsdQaccum));
        Tensor tdQrdQaccum = s2r_thr_copy_dQaccum.retile_D(taccdQrdQaccum);
        cute::copy(s2r_tiled_copy_dQaccum, tdQsdQaccum, tdQrdQaccum);
        if (params.dq_exponent != nullptr) {
            // int32 sums on the row grid 2^(e-30); rows index like dPsum.
            auto thr_mma_dQ = tiled_mma_dQ.get_thread_slice(thread_idx);
            Tensor cdQ = thr_mma_dQ.partition_C(cute::make_identity_tensor(select<!dQ_swapAB ? 0 : 1, !dQ_swapAB ? 1 : 0>(TileShape_MK{})));
            int const rounded = int(get<0>(params.shape_dQaccum) / kHeadDim);
            int const* const exp_row = params.dq_exponent
                + (int64_t(!is_varlen ? bidb : 0) * get<1>(params.shape_dQaccum) + bidh) * rounded
                + seqlen_info.offset_padded + m_block * kBlockM;
            #pragma unroll
            for (int i = 0; i < size(taccdQrdQaccum); ++i) {
                int const row = !dQ_swapAB ? int(get<0>(cdQ(i))) : int(get<1>(cdQ(i)));
                taccdQrdQaccum(i) = float(__float_as_int(taccdQrdQaccum(i))) * __int_as_float((97 + exp_row[row]) << 23) * params.softmax_scale;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < size(taccdQrdQaccum); ++i) { taccdQrdQaccum(i) *= params.softmax_scale; }
        }
        Element* ptr_out = params.ptr_dQ;
        StridedQ stride_out = params.stride_dQ;
        int out_head = bidh;
        if constexpr (NegQReaders) {
            int const readers = params.neg_q.readers;
            int const group = bidh / readers;
            int const reader = bidh - group * readers;
            out_head = reader == 0 ? group : group * (readers - 1) + reader - 1;
            if (reader == 0 && params.neg_q.ptr_dK_accum != nullptr) {
                // All loads before any use: one memory round trip per thread.
                auto thr_mma_dQ = tiled_mma_dQ.get_thread_slice(thread_idx);
                Tensor cdQ = thr_mma_dQ.partition_C(cute::make_identity_tensor(select<!dQ_swapAB ? 0 : 1, !dQ_swapAB ? 1 : 0>(TileShape_MK{})));
                StridedQ const stride = params.neg_q.stride_dK_accum;
                float const* const dk = params.neg_q.ptr_dK_accum + int64_t(bidb) * get<3>(stride)
                    + int64_t(group) * get<2>(stride) + int64_t(m_block) * kBlockM * get<0>(stride);
                int const rows = std::min(seqlen_info.seqlen - m_block * kBlockM, kBlockM);
                float dk_values[decltype(size(taccdQrdQaccum))::value];
                if constexpr (!dQ_swapAB) {
                    // GMMA accumulator values 2j, 2j + 1 are adjacent columns of one row.
                    static_assert(decltype(size(taccdQrdQaccum))::value % 2 == 0);
                    #pragma unroll
                    for (int i = 0; i < size(taccdQrdQaccum); i += 2) {
                        int const row = int(get<0>(cdQ(i)));
                        int const col = int(get<1>(cdQ(i)));
                        // Unconditional from a clamped row, masked at use (a guarded
                        // load becomes a branch that serializes the round trips).
                        float2 const pair = *reinterpret_cast<float2 const*>(
                            dk + (row < rows ? row : 0) * get<0>(stride) + col);
                        dk_values[i] = row < rows ? pair.x : 0.0f;
                        dk_values[i + 1] = row < rows ? pair.y : 0.0f;
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < size(taccdQrdQaccum); ++i) {
                        int const row = int(get<1>(cdQ(i)));
                        int const col = int(get<0>(cdQ(i)));
                        dk_values[i] = row < rows ? dk[row * get<0>(stride) + col] : 0.0f;
                    }
                }
                #pragma unroll
                for (int i = 0; i < size(taccdQrdQaccum); ++i) { taccdQrdQaccum(i) += dk_values[i]; }
            }
            if (reader == 0) {
                ptr_out = params.neg_q.ptr_dNegQ;
                stride_out = params.neg_q.stride_dNegQ;
            }
        }
        // Convert tdQrdQ from fp32 to fp16
        Tensor rdQ = make_tensor_like<Element>(taccdQrdQaccum);
        flash::convert_type_out(taccdQrdQaccum, rdQ);

        // Step 3: Copy dQ from register to smem
        auto smem_tiled_copy_dQ = make_tiled_copy_C(SmemCopyAtomdQ{}, tiled_mma_dQ);
        auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(thread_idx);
        Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
        Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(cute::conditional_return<!dQ_swapAB>(sdQ, sdQt));  // ((Atom,AtomNum),PIPE_M,PIPE_N)
        cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
        __syncthreads();

        // Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
        Tensor mdQ = make_tensor(make_gmem_ptr(ptr_out), params.shape_dQ, stride_out)(_, _, out_head, !is_varlen ? bidb : 0);
        Tensor gdQ = local_tile(domain_offset(make_coord(seqlen_info.offset, _0{}), mdQ), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)
        GmemTiledCopy gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(thread_idx);
        Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

        Tensor tdQrdQ = make_fragment_like(tdQsdQ);
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cute::make_identity_tensor(TileShape_MK{}));
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(_0{}, _0{}, k)) < get<1>(params.shape_dQ); }
        // Need to check OOB when reading from smem if kBlockM isn't evenly tiled
        static constexpr bool EvenM = kBlockM % CUTE_STATIC_V(size<0>(GmemLayoutAtom{})) == 0;
        flash::copy</*Is_even_MN=*/EvenM, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
            gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ, tdQcdQ, tdQpdQ, kBlockM);

        // Copy dQ to global memory, preserving out-of-range destinations.
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, std::min(seqlen_info.seqlen - m_block * kBlockM, kBlockM)
        );
    }

};

} // namespace flash

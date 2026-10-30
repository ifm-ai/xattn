/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "attention/sequence/seqlen.h"
#include "attention/cuda/detail/cute_compat.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, int kHeadDimQ_, class Element, class ElementAccum,
          class ArchTag_, bool Clear_dQaccum, bool Varlen>
class AttentionBwdPreprocess {

public:

    // Type Aliases
    using TileShape_MK = TileShape_MK_;
    using ArchTag = ArchTag_;

    static_assert(std::is_same_v<Element, cutlass::half_t> && ArchTag::kMinComputeCapability >= 75 ||
                  std::is_same_v<Element, cutlass::bfloat16_t> && ArchTag::kMinComputeCapability >= 80 ||
                  std::is_same_v<Element, cutlass::float_e4m3_t> && ArchTag::kMinComputeCapability >= 89);

    static constexpr uint32_t MaxThreadsPerBlock = 256;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
    static constexpr int SharedStorageSize = 0;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<1>(TileShape_MK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBlockM = get<0>(TileShape_MK{});
    static constexpr int kHeadDim = get<1>(TileShape_MK{});
    static constexpr int kHeadDimQ = kHeadDimQ_;
    // Power-of-two K width keeps reductions within a warp.
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static_assert((kBlockM * kHeadDimQ / kGmemElemsPerLoadAccum) % MaxThreadsPerBlock == 0, "MaxThreadsPerBlock must divide kBlockM * kHeadDimQ / kGmemElemsPerLoadAccum");
    using GmemLayoutAtomAccum = Layout<Shape<Int<MaxThreadsPerBlock>>>;
    using GmemTiledCopyAccum = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        GmemLayoutAtomAccum{},
                        Layout<Shape<Int<kGmemElemsPerLoadAccum>>>{}));  // Val layout, 4 vals per store

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapedPsum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_q, head, batch)
    using StridedPsum = cute::Stride<_1, int64_t, int64_t>;
    // int64 extent: varlen workspaces can exceed 2^31 / d padded rows times d.
    using ShapedQaccum = cute::Shape<int64_t, int32_t, int32_t>;  // (seqlen_q * d, head, batch)
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    // Device side arguments
    struct Arguments {
        Element const* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        Element const* ptr_dO;
        StrideO const stride_dO;
        float* ptr_dPsum;
        ShapedPsum const shape_dPsum;
        StridedPsum const stride_dPsum;
        float const* ptr_LSE;
        StridedPsum const stride_LSE;
        float *ptr_LSE_log2;
        StridedPsum const stride_LSE_log2;
        ElementAccum* ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        int num_batch;
        int num_m_blocks;
        int* dq_semaphore;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        // Fixed-point deterministic dQ: max_j |V_j|, max |K| in, row exponents out.
        float const* ptr_fixed_bounds = nullptr;
        int* ptr_dq_exponent = nullptr;
    };

    // Kernel entry point API
    struct Params {
        Element const* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        Element const* ptr_dO;
        StrideO const stride_dO;
        float* ptr_dPsum;
        ShapedPsum const shape_dPsum;
        StridedPsum const stride_dPsum;
        float const* ptr_LSE;
        StridedPsum const stride_LSE;
        float* ptr_LSE_log2;
        StridedPsum const stride_LSE_log2;
        ElementAccum* ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        int num_batch;
        int num_m_blocks;
        int* dq_semaphore;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        float const* ptr_fixed_bounds = nullptr;
        int* ptr_dq_exponent = nullptr;
    };

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        return {
            args.ptr_O,
            args.shape_O,
            args.stride_O,
            args.ptr_dO,
            args.stride_dO,
            args.ptr_dPsum,
            args.shape_dPsum,
            args.stride_dPsum,
            args.ptr_LSE,
            args.stride_LSE,
            args.ptr_LSE_log2,
            args.stride_LSE_log2,
            args.ptr_dQaccum,
            args.shape_dQaccum,
            args.stride_dQaccum,
            args.num_batch,
            args.num_m_blocks,
            args.dq_semaphore,
            args.cu_seqlens,
            args.seqused,
            args.ptr_fixed_bounds,
            args.ptr_dq_exponent
        };
    }

    // Batches past the grid z limit are strided: blockIdx.z, + gridDim.z, ...
    CUTLASS_DEVICE
    void
    operator()(Params const& params, [[maybe_unused]] char* smem_buf) {
        for (int bidb = int(blockIdx.z); bidb < params.num_batch; bidb += int(gridDim.z)) {
            run_tile(params, int(blockIdx.x), int(blockIdx.y), bidb);
        }
    }

    CUTLASS_DEVICE
    void
    run_tile(Params const& params, int const m_block, int const bidh, int const bidb) {

        static constexpr int kBlockM = get<0>(TileShape_MK{});

        int const thread_idx = threadIdx.x;

        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info(bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused);
        bool const is_varlen = Varlen && params.cu_seqlens;
        int const seqlen_o = seqlen_info.seqlen;
        if (is_varlen && m_block * kBlockM >= seqlen_o) { return; }

        Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_O, params.stride_dO)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdO = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdO), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)

        auto shape_LSE = select<0, 2, 3>(params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), shape_LSE, params.stride_LSE)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gLSE = local_tile(cute::domain_offset(make_coord(seqlen_info.offset), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block));
        static_assert(kBlockM <= MaxThreadsPerBlock);
        float lse = thread_idx < seqlen_o - m_block * kBlockM && thread_idx < kBlockM ? gLSE(thread_idx) : INFINITY;

        GmemTiledCopy gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

        Tensor tOgdO = gmem_thr_copy_O.partition_S(gdO);
        // Construct identity layout for gO
        Tensor cO = cute::make_identity_tensor(TileShape_MK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgdO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }

        // (8, kBlockM / 32, kHeadDim / 64) or (8, kBlockM / 16, kHeadDim / 128)
        Tensor tOrO = make_fragment_like(tOgdO);
        Tensor tOrdO = make_fragment_like(tOgdO);
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
            gmem_tiled_copy_O, tOgdO, tOrdO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );

        // Reshape from e.g. (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, (8, kHeadDim / 64))
        Layout l = make_layout(get<1>(tOrO.layout()), make_layout(get<0>(tOrO.layout()), get<2>(tOrO.layout())));
        Tensor tOrO_l = make_tensor(tOrO.data(), l);
        Tensor mO = make_tensor(
            make_gmem_ptr(params.ptr_O), params.shape_O,
            params.stride_O)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gO = local_tile(
            cute::domain_offset(
                make_coord(seqlen_info.offset, _0{}), mO),
            TileShape_MK{}, make_coord(m_block, _0{}));
        Tensor tOgO = gmem_thr_copy_O.partition_S(gO);
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false,
                    /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
            gmem_tiled_copy_O, tOgO, tOrO, tOcO, tOpO,
            seqlen_o - m_block * kBlockM);
        Tensor o_fp32 = make_tensor_like<float>(tOrO_l);
        flash::convert_type_out(tOrO_l, o_fp32);
        Tensor tOrdO_l = make_tensor(tOrdO.data(), l);
        Tensor do_fp32 = make_tensor_like<float>(tOrdO_l);
        flash::convert_type_out(tOrdO_l, do_fp32);
        // Sum across the last dimension
        Tensor dP_sum = make_tensor<float>(make_shape(size<0>(o_fp32)));
        #pragma unroll
        for (int mi = 0; mi < size<0>(o_fp32); ++mi) {
            float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
            #pragma unroll
            for (int ni = 1; ni < size<1>(o_fp32); ni++) {
                dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
            }
            flash::SumOp<float> sum_op;
            dP_sum(mi) = flash::Allreduce<kGmemThreadsPerRow>::run(dP_sum_cur, sum_op);
        }

        // Row exponent e with 2 |dO_i| max|V| max|K| (1 + 1/64) < 2^e.
        Tensor dq_exp = make_tensor<int>(make_shape(size<0>(o_fp32)));
        if (params.ptr_dq_exponent != nullptr) {
            float const bound_vk = params.ptr_fixed_bounds[0] * params.ptr_fixed_bounds[1];
            #pragma unroll
            for (int mi = 0; mi < size<0>(do_fp32); ++mi) {
                float sq = 0.f;
                #pragma unroll
                for (int ni = 0; ni < size<1>(do_fp32); ++ni) { sq += do_fp32(mi, ni) * do_fp32(mi, ni); }
                flash::SumOp<float> sum_op;
                sq = flash::Allreduce<kGmemThreadsPerRow>::run(sq, sum_op);
                int e = 0;
                frexpf(2.f * sqrtf(sq) * bound_vk * 1.015625f, &e);
                // 2^(30-e) and 2^(e-30) stay normal FP32 for e in [-90, 120].
                dq_exp(mi) = max(-90, min(120, e));
            }
        }
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_dPsum, params.stride_dPsum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdPsum = local_tile(cute::domain_offset(make_coord(seqlen_info.offset_padded), mdPsum), Shape<Int<kBlockM>>{}, make_coord(m_block));
        if (get<1>(tOcO(_0{}, _0{}, _0{})) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(dP_sum); ++mi) {
                int const row = get<0>(tOcO(_0{}, mi, _0{}));
                gdPsum(row) = row < seqlen_o - m_block * kBlockM ? dP_sum(mi) : 0;
            }
            if (params.ptr_dq_exponent != nullptr) {
                Tensor mExp = make_tensor(make_gmem_ptr(params.ptr_dq_exponent), params.shape_dPsum, params.stride_dPsum)(_, bidh, !is_varlen ? bidb : 0);
                Tensor gExp = local_tile(cute::domain_offset(make_coord(seqlen_info.offset_padded), mExp), Shape<Int<kBlockM>>{}, make_coord(m_block));
                #pragma unroll
                for (int mi = 0; mi < size(dq_exp); ++mi) {
                    int const row = get<0>(tOcO(_0{}, mi, _0{}));
                    gExp(row) = row < seqlen_o - m_block * kBlockM ? dq_exp(mi) : 0;
                }
            }
        }

        int const seqlen_rounded = cute::round_up(seqlen_o, kBlockM);
        Tensor mLSElog2 = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_dPsum, params.stride_LSE_log2)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gLSElog2 = local_tile(cute::domain_offset(make_coord(seqlen_info.offset_padded), mLSElog2), Shape<Int<kBlockM>>{}, make_coord(m_block));
        if (thread_idx < seqlen_rounded - m_block * kBlockM && thread_idx < kBlockM) {
            gLSElog2(thread_idx) = lse == -INFINITY ? 0.f : lse * float(M_LOG2E);
        }

        if constexpr (Clear_dQaccum) {
            Tensor mdQaccum = make_tensor(make_gmem_ptr(params.ptr_dQaccum), params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0);
            Tensor gdQaccum = local_tile(cute::domain_offset(make_coord(int64_t(seqlen_info.offset_padded) * kHeadDimQ), mdQaccum), Shape<Int<kBlockM * kHeadDimQ>>{}, make_coord(m_block));
            GmemTiledCopyAccum gmem_tiled_copy_dQaccum;
            auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
            Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
            Tensor zero = make_fragment_like(tdQgdQaccum);
            clear(zero);
            cute::copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, zero, tdQgdQaccum);
        }

        if (params.dq_semaphore != nullptr && thread_idx == 0) {
            int const num_batch = params.num_batch;
            int const num_head = get<2>(params.shape_O);
            params.dq_semaphore[bidh + bidb * num_head + m_block * num_head * num_batch] = 0;
        }

    }

};

} // namespace flash

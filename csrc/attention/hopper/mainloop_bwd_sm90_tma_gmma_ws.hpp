/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Modified by: Shicheng Wen (xattn adaptations).

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/barrier.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "attention/hopper/semantics/segment.h"
#include "attention/hopper/semantics/position.h"
#include "attention/hopper/semantics/mask.h"
#include "attention/hopper/semantics/visibility.h"

#include "attention/hopper/barriers.h"
#include "attention/sequence/seqlen.h"
#include "attention/cuda/softmax.h"
#include "attention/cuda/detail/cute_compat.h"
#include "hopper/cute_ext/bulk_reduce_add.hpp"
#include "hopper/cutlass_ext/tma_pipeline_no_cluster.hpp"

namespace flash {

using namespace cute;

template <int Stages, int Stages_dO, int Stages_dS, class ClusterShape_, class TileShape_MNK_, int kHeadDimV_, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool Deterministic, bool HasSegment_,
        bool IsSlidingChunk_, bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
        int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
        bool Mma_dP_is_RS=false, bool TwoComponentDV_=false,
        bool ReaderPairKVReuse_=false,
        bool DirectChunkTile_=IsSlidingChunk_,
        bool HeadPairParallel_=false,
        bool HeadPairClusterKVReuse_=false,
        bool HeadPairPrivateKVGrad_=false, bool SplitDQ_=false>
struct CollectiveMainloopBwdSm90 {

    static constexpr int kStages = Stages;
    static constexpr bool SplitDQ = SplitDQ_;
    static constexpr int kStages_dO = Stages_dO;
    static constexpr int kStages_dS = Stages_dS;
    static_assert(kStages >= kStages_dO);
    static_assert(Stages_dS == 1 || Stages_dS == kStages);
    static_assert(!Mma_dP_is_RS || SdP_swapAB_);  // RS dP requires SdP_swapAB.
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool Has_segment = HasSegment_;
    static constexpr bool Is_sliding_chunk = IsSlidingChunk_;
    static constexpr bool TwoComponentDV = TwoComponentDV_;
    static constexpr bool ReaderPairKVReuse = ReaderPairKVReuse_;
    static constexpr bool HeadPairParallel = HeadPairParallel_;
    static constexpr bool HeadPairClusterKVReuse =
        HeadPairClusterKVReuse_;
    static constexpr bool HeadPairPrivateKVGrad =
        HeadPairPrivateKVGrad_;

    static_assert(!ReaderPairKVReuse || (!Varlen_ && !HasSegment_),
                  "reader-pair K/V reuse requires dense inputs");
    static_assert(!HeadPairParallel || (!Varlen_ && !HasSegment_),
                  "parallel head-pair execution requires dense inputs");
    static_assert(!HeadPairClusterKVReuse || HeadPairParallel,
                  "cluster K/V reuse requires parallel head-pair execution");
    static_assert(!HeadPairPrivateKVGrad || HeadPairParallel,
                  "private K/V gradients require parallel head-pair execution");
    static_assert(!(ReaderPairKVReuse && HeadPairParallel),
                  "same-CTA and parallel head-pair execution are exclusive");

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;

    static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kHeadDimV = kHeadDimV_;
    // Match dense full forward's exponent approximation on equal-D/V kernels.
    static constexpr bool UseFastExp2 =
        Is_causal && !Varlen && !Has_segment && kHeadDim == kHeadDimV;
    static constexpr bool UseDirectChunkMask =
        Is_sliding_chunk && kHeadDim == 256 && kHeadDimV == 256;
    static constexpr bool UseDirectChunkTile = DirectChunkTile_;
    using TileShape_MNK_V = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDimV>>;

    using SeqlenInfo_t = flash::SeqlenInfoQK<Varlen, kBlockM>;
    using AttentionRange = std::conditional_t<
        Is_sliding_chunk,
        xattn::ops::attention::hopper::semantics::SlidingChunkRangeAdapter<
            SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local,
            false, false, Has_segment>,
        xattn::ops::attention::hopper::semantics::StandardRangeAdapter<
            SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local>>;
    using SegmentPartition =
        xattn::ops::attention::hopper::semantics::SegmentPartitionAdapter<
            kBlockM, kBlockN>;

    CUTLASS_DEVICE
    static int q_head_for_reader(int const kv_head, int const reader,
                                 int const readers_per_kv_head) {
        if constexpr (ReaderPairKVReuse || HeadPairParallel) {
            return kv_head * readers_per_kv_head + reader;
        } else {
            return kv_head;
        }
    }

    CUTLASS_DEVICE
    static int head_pair_reader_index() {
        if constexpr (HeadPairClusterKVReuse) {
            return int(cute::block_rank_in_cluster());
        } else if constexpr (HeadPairParallel) {
            // Non-cluster head pairs use two complete reader planes.
            return int(blockIdx.x / (gridDim.x / 2));
        } else {
            return 0;
        }
    }

    static_assert(ArchTag::kMinComputeCapability >= 90);
    static_assert(get<2>(ClusterShape{}) == 1);
    static_assert(
        get<0>(ClusterShape{}) == 1 ||
        (HeadPairClusterKVReuse && get<0>(ClusterShape{}) == 2 &&
         get<1>(ClusterShape{}) == 1));

    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp * 2;

    static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0);
    static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 && AtomLayoutNdKV == NumMmaWarpGroups && SdP_swapAB && !dKV_swapAB;
    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarpGroups && AtomLayoutMdQ == NumMmaWarpGroups && !SdP_swapAB && !dQ_swapAB;  // RS dQ requires !dQ_swapAB.

    static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
    static constexpr GMMA::Major PdSt_Major = PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<NumMmaWarpGroups / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>, Int<kHeadDimV>>,
        Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDimV>>
    >;
    using TiledMmadPSS = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdP>(),
        AtomLayoutSdP{}));
    using TiledMmadPRS = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdK = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>
    >;
    using TileShapeAtomdV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDimV / (NumMmaWarpGroups / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDimV>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<NumMmaWarpGroups / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadK = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Mma_dKV_is_RS,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdK, GMMA::Major::K, GMMA::Major::MN>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdK, !dKV_swapAB ? PdSt_Major : GMMA::Major::MN, !dKV_swapAB ? GMMA::Major::MN : PdSt_Major>())
        >{},
        AtomLayoutdKV{}));
    using TiledMmadV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Mma_dKV_is_RS,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdV, GMMA::Major::K, GMMA::Major::MN>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdV, !dKV_swapAB ? PdSt_Major : GMMA::Major::MN, !dKV_swapAB ? GMMA::Major::MN : PdSt_Major>())
        >{},
        AtomLayoutdKV{}));
    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim>, Int<kBlockM / AtomLayoutMdQ>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<NumMmaWarpGroups / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Mma_dQ_is_RS,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, !dQ_swapAB ? PdS_Major : GMMA::Major::MN, !dQ_swapAB ? GMMA::Major::MN : PdS_Major>())
        >{},
        AtomLayoutdQ{}));

    // Shared storage supports Q/dO for SdP and their transposes for dKV.
    // GMMA::Major::K layout depends on the K dimension.
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>>()); // for dKV_Mma
    using SmemLayoutAtomdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDimV / (NumMmaWarpGroups / AtomLayoutNdKV)>>()); // for dV_Mma
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutdO =
        decltype(tile_to_shape(SmemLayoutAtomdO{},
                 make_shape(Int<kBlockM>{}, Int<kHeadDimV>{}, Int<kStages_dO>{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), Int<kHeadDimV>>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK_V{})));

    using SmemLayoutAtomPdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<PdS_Major, Element,
                                       Int<kBlockM / AtomLayoutMSdP>,
                                       Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
        std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    // Stride is a multiple of 64 for TMA alignment and padded ShuffleLSE reads.
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<cute::round_up(kBlockM, 64)>>>;
    using SmemLayoutLSEMma = std::conditional_t<
        SdP_swapAB,
        cute::Layout<cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 64)>>>,
        cute::Layout<cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>, cute::Stride<_1, _0, Int<cute::round_up(kBlockM, 64)>>>
    >;

    // Transposed tensor view.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(Int<kHeadDimV>{}, get<0>(TileShape_MNK{}), Int<kStages_dO>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDimV>{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    using SmemLayoutPdSt =
        decltype(cute::composition(SmemLayoutPdS{},
                                   make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

    // Thread layout: 256 or 384 threads per row, partitioned by warpgroup.
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumMmaWarpGroups>>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim / NumMmaWarpGroups>, Int<NumMmaWarpGroups>>>;

    static constexpr int kNumPdSStore = kBlockM * kBlockN / NumMmaThreads;
    // Accumulators hold P/dS for !SdP_swapAB and Pt/dSt otherwise.
    // MN-major PdS requires transposed writes.
    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<(!SdP_swapAB) ^ (PdS_Major == GMMA::Major::MN),
            std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U32x4_STSM_N, cute::SM90_U32x2_STSM_N>,
            std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>
        >,
        Element
    >;

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
    using GmemTiledCopyKV = std::conditional_t<
        HeadPairClusterKVReuse,
        decltype(cutlass::gemm::collective::detail::
                     sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{}))),
        cute::SM90_TMA_LOAD>;
    using TmaCopyClusterShape = std::conditional_t<
        HeadPairClusterKVReuse, Shape<_1, _1, _1>, ClusterShape>;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;  // (seqlen, head, batch)
    using ShapedQaccum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_q * d, head, batch)
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    using TMA_Q = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}),
        TileShape_MNK{},
        ClusterShape{})); // mcast along N mode for this M load, if any
    using TMA_dO = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutdO{}),
        TileShape_MNK_V{},
        ClusterShape{})); // mcast along N mode for this M load, if any

    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK{},
        TileShape_MNK{},
        TmaCopyClusterShape{}));

    using TMA_V = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutV{},
        TileShape_MNK_V{},
        TmaCopyClusterShape{}));

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
    using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesdO = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutdO{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);

    // Clustered paired readers keep one K/V tile per CTA and use one pipeline
    // stage. Owner-ranked K and V pipelines arm every multicast destination
    // before issue. Non-paired kernels use one combined transaction barrier.
    static constexpr int kStagesKV = 1;
    using MainloopPipelineKCluster =
        cutlass::PipelineTmaAsyncSplitMulticast<
            kStagesKV, 0, TmaTransactionBytesK>;
    using MainloopPipelineVCluster =
        cutlass::PipelineTmaAsyncSplitMulticast<
            kStagesKV, 1, TmaTransactionBytesV>;
    using MainloopPipelineK = std::conditional_t<
        HeadPairClusterKVReuse, MainloopPipelineKCluster, decltype(nullptr)>;
    using MainloopPipelineV = std::conditional_t<
        HeadPairClusterKVReuse, MainloopPipelineVCluster, decltype(nullptr)>;
    using PipelineStateKV = cutlass::PipelineState<kStagesKV>;

    struct CombinedKVSharedStorage : cute::aligned_struct<16> {
        alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
    };
    struct ClusterKVSharedStorage : cute::aligned_struct<16> {
        alignas(16) typename MainloopPipelineKCluster::SharedStorage pipeline_k;
        alignas(16) typename MainloopPipelineVCluster::SharedStorage pipeline_v;
    };
    using KVSharedStorage = std::conditional_t<
        HeadPairClusterKVReuse,
        ClusterKVSharedStorage,
        CombinedKVSharedStorage>;

    // Separate causal-mask iterations for selected head dimensions.
    static constexpr bool SeparateMaskingIterations = kHeadDim <= 64;
    // ShuffleLSE distributes row statistics across eight threads.
    static constexpr bool ShuffleLSE = SdP_swapAB && kHeadDim <= 64;
    static constexpr bool ShuffledPsum = SdP_swapAB && kHeadDim <= 64;
    static constexpr bool dQacc_use_TMA = kHeadDim < 256 || Deterministic;
    static_assert(
        !Deterministic || dQacc_use_TMA,
        "Deterministic attention BWD must not use unordered dQ atomicAdd");
    // Split hdim256 dQ MMA into two 128-column slices with separate atomic adds.
    static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && kHeadDimV == 256 && !dQacc_use_TMA && dQ_swapAB && AtomLayoutMdQ == 1 && NumMmaWarpGroups == 2;
    static_assert(!(Deterministic && Slice_dQKV_Mma), "Deterministic mode not supported with Slice_dQKV_Mma");

    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    static constexpr size_t SmemAlignmentdS = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    // TMA shared-memory alignment.
    static constexpr size_t SmemAlignmentQKVdO =
        (kHeadDim % 256 == 0 || kHeadDimV % 256 == 0) ? 256 : 128;
    static constexpr size_t SmemAlignmentV = !Mma_dP_is_RS ? SmemAlignmentQKVdO : cutlass::detail::alignment_for_swizzle(SmemLayoutV{});
    static_assert(SmemAlignmentP >= 128 && SmemAlignmentdS >= 128, "Require at least 128B alignment");

    using SmemdQacc_t = std::conditional_t<!dQacc_use_TMA, cute::array<ElementAccum, 0>, cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccum>>>;
    using SmemP_t = std::conditional_t<Mma_dKV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP>>;
    using SmemPResidual_t = std::conditional_t<
        Mma_dKV_is_RS || !TwoComponentDV,
        cute::array<Element, 0>,
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP>>;
    struct TensorStorage : cute::aligned_struct<cute::max(SmemAlignmentP, SmemAlignmentdS, SmemAlignmentQKVdO)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
        SmemdQacc_t smem_dqacc;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_dpsum;
        SmemP_t smem_p;
        SmemPResidual_t smem_p_residual;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
    };

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* const ptr_V;
        ShapeQKV const shape_V;
        StrideQKV const stride_V;
        Element const* const ptr_dO;
        ShapeQKV const shape_dO;
        StrideQKV const stride_dO;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        int const window_size_left, window_size_right;
        int const attention_chunk;
        float const softcap_val;
        int const num_batch;
        int* const dq_semaphore;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int64_t const* const q_segment_idx = nullptr;
        int64_t const* const k_segment_idx = nullptr;
        int const k_segment_len = 0;
        int const* const q_position_offsets = nullptr;
        int const* const q_chunk_positions = nullptr;
        int const reset_attention_chunk = 0;
        int const* const deterministic_segment_ranges = nullptr;
        // Paired-reader state extends the fixed SWA/SCA argument prefix.
        int const odd_head_window_right_delta = 0;
        int const dq_slots = 1;
        int64_t const dq_slot_stride = 0;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const shape_V;
        ShapeQKV const shape_dO;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum stride_dQaccum;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;
        TMA_dO tma_load_dO;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale, softmax_scale_log2;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        float const softcap_val;
        int const num_batch;
        int* const dq_semaphore;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int64_t const* const q_segment_idx = nullptr;
        int64_t const* const k_segment_idx = nullptr;
        int const k_segment_len = 0;
        int const* const q_position_offsets = nullptr;
        int const* const q_chunk_positions = nullptr;
        cutlass::FastDivmod reset_attention_chunk_divmod;
        int const* const deterministic_segment_ranges = nullptr;
        // Paired-reader state extends the fixed device-parameter prefix and is
        // compiled out of ordinary kernels.
        int const odd_head_window_right_delta = 0;
        int const dq_slots = 1;
        int64_t const dq_slot_stride = 0;

        CUTLASS_DEVICE
        int get_chunk_position_offset(
                int const bidb, int const seqlen_q,
                int const seqlen_k) const {
            return xattn::ops::attention::hopper::semantics::SlidingChunkPositionAdapter::
                chunk_position_offset(
                    q_position_offsets, bidb, seqlen_q, seqlen_k);
        }

        CUTLASS_DEVICE
        int window_size_right_for_head(int const bidh) const {
            if constexpr (ReaderPairKVReuse || HeadPairParallel) {
                return window_size_right +
                    ((bidh & 1) ? odd_head_window_right_delta : 0);
            } else {
                return window_size_right;
            }
        }
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQdO{},
            mQ,
            SmemLayoutQ{}(_, _, _0{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along N mode for this M load, if any
        Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_dO, args.stride_dO);
        TMA_dO tma_load_dO = make_tma_copy_A_sm90(
            GmemTiledCopyQdO{},
            mdO,
            SmemLayoutdO{}(_, _, _0{}),
            TileShape_MNK_V{},
            ClusterShape{}); // mcast along N mode for this M load, if any
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{},
            TileShape_MNK{},
            TmaCopyClusterShape{});
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_V, args.stride_V);
        TMA_V tma_load_V = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutV{},
            TileShape_MNK_V{},
            TmaCopyClusterShape{});
        if constexpr (Deterministic) { assert(args.dq_semaphore != nullptr); }
        if constexpr (Deterministic && Has_segment) {
            assert(args.deterministic_segment_ranges != nullptr);
        }
        // Avoid dividing by zero
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        cutlass::FastDivmod reset_attention_chunk_divmod(
            args.reset_attention_chunk >= 1 ? args.reset_attention_chunk : 1);
        reset_attention_chunk_divmod.divisor = args.reset_attention_chunk;
        // Softcap uses softmax_scale / softcap_val before tanh and
        // softcap_val * log2(e) before exp2. Backward applies (1 - tanh^2)
        // in the loop and the original softmax_scale in the gradient epilogue.
        return {args.shape_Q, args.shape_K,
                args.shape_V, args.shape_dO,
                args.ptr_dQaccum, args.shape_dQaccum, args.stride_dQaccum,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_dO, tma_load_K, tma_load_V,
                args.ptr_LSE_log2, args.shape_LSE, args.stride_LSE_log2, args.ptr_dPsum, args.stride_dPsum,
                args.softmax_scale,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.window_size_left, args.window_size_right,
                attention_chunk_divmod,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.num_batch, args.dq_semaphore,
                args.cu_seqlens_q, args.cu_seqlens_k, args.seqused_q, args.seqused_k,
                args.q_segment_idx, args.k_segment_idx, args.k_segment_len,
                args.q_position_offsets, args.q_chunk_positions,
                reset_attention_chunk_divmod,
                args.deterministic_segment_ranges,
                args.odd_head_window_right_delta,
                args.dq_slots, args.dq_slot_stride};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t>
    get_m_block_range_for_q_head(
                      Params const& params,
                      cute::tuple<int32_t, int32_t, int32_t> block_coord,
                      int const q_head) const {
        int const n_block = get<0>(block_coord);
        int const bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = AttentionRange::bwd_m_block_range(
            seqlen_info, n_block, bidb, params.window_size_left,
            params.window_size_right_for_head(q_head),
            0 /*sink_token_length*/,
            params.attention_chunk_divmod,
            params.get_chunk_position_offset(
                bidb, seqlen_info.seqlen_q, seqlen_info.seqlen_k));
        if constexpr (Has_segment) {
            auto [segment_m_block_min, segment_m_block_max] =
                SegmentPartition::bwd_m_block_range(
                    m_block_min, m_block_max, n_block,
                    seqlen_info.seqlen_q, seqlen_info.seqlen_k,
                    params.q_segment_idx, params.k_segment_idx,
                    params.k_segment_len, bidb);
            m_block_min = segment_m_block_min;
            m_block_max = segment_m_block_max;
        }
        return {m_block_min, m_block_max};
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t>
    get_m_block_range(Params const& params,
                      cute::tuple<int32_t, int32_t, int32_t> block_coord) const {
        int const kv_head = get<1>(block_coord);
        int const reader = HeadPairParallel
            ? head_pair_reader_index()
            : 0;
        // Grouped readers share the coarse M-tile range; their element masks
        // differ by one position. BWD tile widths are multiples of 16.
        return get_m_block_range_for_q_head(
            params, block_coord,
            q_head_for_reader(
                kv_head, reader,
                params.qhead_per_khead_divmod.divisor));
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load_reader_group(Params const& params,
         MainloopPipeline pipeline_q,
         MainloopPipeline_dO pipeline_do,
         PipelineState& smem_pipe_write,
         PipelineState_dO& smem_pipe_write_do,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         cute::tuple<int32_t, int32_t> m_block_range) {
        int const n_block = get<0>(block_coord);
        int const kv_head = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k,
            params.seqused_q, params.seqused_k};
        auto const [m_block_min, m_block_max] = m_block_range;
        if (m_block_max <= m_block_min) {
            scheduler_prefetch();
            return;
        }

        Tensor sQ = make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
            SmemLayoutQ{});
        Tensor sdO = make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
            SmemLayoutdO{});
        Tensor sK = make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
            SmemLayoutK{});
        Tensor sV = make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
            SmemLayoutV{});
        Tensor sLSE = make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()),
            SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()),
            SmemLayoutLSE{});

        int const readers_per_kv_head =
            params.qhead_per_khead_divmod.divisor;
        int const first_q_head =
            q_head_for_reader(kv_head, 0, readers_per_kv_head);
        int const bidh_kv = kv_head;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(
            _, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_V)(
            _, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor gK = local_tile(
            domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK),
            select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));
        Tensor gV = local_tile(
            domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV),
            select<1, 2>(TileShape_MNK_V{}),
            make_coord(n_block, _0{}));
        Tensor sK_x = make_tensor(
            sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(
            gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(
            sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(
            gV.data(), make_layout(gV.layout(), Layout<_1>{}));
        auto [tKgK, tKsK] = tma_partition(
            params.tma_load_K, _0{}, Layout<_1>{},
            group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));
        auto [tVgV, tVsV] = tma_partition(
            params.tma_load_V, _0{}, Layout<_1>{},
            group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));

        uint32_t const block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 const cluster_local_block_id = {
            block_rank_in_cluster % cluster_shape_x,
            block_rank_in_cluster / cluster_shape_x};
        auto block_tma_Q =
            params.tma_load_Q.get_slice(cluster_local_block_id.y);
        auto block_tma_dO =
            params.tma_load_dO.get_slice(cluster_local_block_id.y);
        uint16_t mcast_mask_qdo = 0;
        if constexpr (
            cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{};
            for (int n = 0; n < size<1>(block_layout); ++n) {
                mcast_mask_qdo |= uint16_t(1)
                    << block_layout(cluster_local_block_id.x, n, _0{});
            }
        }
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
        int const lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            shared_storage.pipelines.kv.barrier_KV.arrive_and_expect_tx(
                TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(
                     reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                         shared_storage.pipelines.kv.barrier_KV),
                     0 /*mcast_mask*/),
                 tKgK, tKsK);
            copy(params.tma_load_V.with(
                     reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                         shared_storage.pipelines.kv.barrier_KV),
                     0 /*mcast_mask*/),
                 tVgV, tVsV);

            CUTLASS_PRAGMA_NO_UNROLL
            for (int reader = 0; reader < readers_per_kv_head; ++reader) {
                int const q_head = q_head_for_reader(
                    kv_head, reader, readers_per_kv_head);
                bool const is_varlen_q = Varlen && params.cu_seqlens_q;
                Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(
                    _, _, q_head, !is_varlen_q ? bidb : 0);
                Tensor mdO =
                    params.tma_load_dO.get_tma_tensor(params.shape_dO)(
                        _, _, q_head, !is_varlen_q ? bidb : 0);
                Tensor mLSE = make_tensor(
                    make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE,
                    params.stride_LSE_log2)(
                        _, q_head, !is_varlen_q ? bidb : 0);
                Tensor mdPsum = make_tensor(
                    make_gmem_ptr(params.ptr_dPsum), params.shape_LSE,
                    params.stride_dPsum)(
                        _, q_head, !is_varlen_q ? bidb : 0);
                Tensor gQ = local_tile(
                    domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ),
                    select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
                Tensor gdO = local_tile(
                    domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO),
                    select<0, 2>(TileShape_MNK_V{}), make_coord(_, _0{}));
                Tensor gLSE = local_tile(
                    domain_offset(
                        make_coord(seqlen_info.offset_q_padded), mLSE),
                    select<0>(TileShape_MNK{}), make_coord(_));
                Tensor gdPsum = local_tile(
                    domain_offset(
                        make_coord(seqlen_info.offset_q_padded), mdPsum),
                    select<0>(TileShape_MNK{}), make_coord(_));
                Tensor tQgQ = group_modes<0, 3>(
                    block_tma_Q.partition_S(gQ));
                Tensor tQsQ = group_modes<0, 3>(
                    block_tma_Q.partition_D(sQ));
                Tensor tdOgdO = group_modes<0, 3>(
                    block_tma_dO.partition_S(gdO));
                Tensor tdOsdO = group_modes<0, 3>(
                    block_tma_dO.partition_D(sdO));

                CUTLASS_PRAGMA_NO_UNROLL
                for (int m_block = m_block_min;
                     m_block < m_block_max; ++m_block) {
                    pipeline_q.producer_acquire(smem_pipe_write);
                    copy(params.tma_load_Q.with(
                             *pipeline_q.producer_get_barrier(smem_pipe_write),
                             mcast_mask_qdo,
                             TMA::CacheHintSm90::EVICT_LAST),
                         tQgQ(_, m_block),
                         tQsQ(_, smem_pipe_write.index()));
                    copy(bulk_copy.with(
                             *pipeline_q.producer_get_barrier(smem_pipe_write)),
                         gLSE(_, m_block),
                         sLSE(_, smem_pipe_write.index()));

                    PipelineState_dO smem_pipe_write_do_cur =
                        cute::conditional_return<Q_dO_same_stages>(
                            smem_pipe_write, smem_pipe_write_do);
                    pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                    copy(params.tma_load_dO.with(
                             *pipeline_do.producer_get_barrier(
                                 smem_pipe_write_do_cur),
                             mcast_mask_qdo,
                             TMA::CacheHintSm90::EVICT_LAST),
                         tdOgdO(_, m_block),
                         tdOsdO(_, smem_pipe_write_do_cur.index()));
                    copy(bulk_copy.with(
                             *pipeline_do.producer_get_barrier(
                                 smem_pipe_write_do_cur)),
                         gdPsum(_, m_block),
                         sdPsum(_, smem_pipe_write_do_cur.index()));
                    if constexpr (!Q_dO_same_stages) {
                        ++smem_pipe_write_do;
                    }
                    ++smem_pipe_write;
                }
            }
        }
        scheduler_prefetch();
        if constexpr (Q_dO_same_stages) {
            smem_pipe_write_do = smem_pipe_write;
        }
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipeline pipeline_q,
         MainloopPipeline_dO pipeline_do,
         MainloopPipelineK pipeline_k,
         MainloopPipelineV pipeline_v,
         PipelineState& smem_pipe_write,
         PipelineState_dO& smem_pipe_write_do,
         PipelineStateKV& smem_pipe_write_kv,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         cute::tuple<int32_t, int32_t> m_block_range
         ) {

        if constexpr (ReaderPairKVReuse) {
            load_reader_group(
                params, pipeline_q, pipeline_do, smem_pipe_write,
                smem_pipe_write_do, shared_storage, scheduler_prefetch,
                block_coord, m_block_range);
            return;
        }

        int const n_block = get<0>(block_coord);
        int const scheduled_head = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        int const m_block_min = get<0>(m_block_range);
        int const m_block_max = get<1>(m_block_range);
        // It's possible to have m_block_max <= m_block_min. Loading Q, K can cause illegal memory access.
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) {
                scheduler_prefetch();
                return;
            }
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        int const bidh = HeadPairParallel
            ? q_head_for_reader(
                  scheduled_head, head_pair_reader_index(),
                  params.qhead_per_khead_divmod.divisor)
            : scheduled_head;
        int const bidh_kv = HeadPairParallel
            ? scheduled_head
            : params.qhead_per_khead_divmod.divide(bidh);
        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_dO)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_V)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, bidh, !is_varlen_q ? bidb : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gdO = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK_V{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK_V{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gLSE = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mLSE), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)

        Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
        auto block_tma_Q = params.tma_load_Q.get_slice(cluster_local_block_id.y);
        auto block_tma_dO = params.tma_load_dO.get_slice(cluster_local_block_id.y);
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
        Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
        Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
        auto [tKgK, tKsK] = tma_partition(
            params.tma_load_K, _0{}, Layout<_1>{},
            group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));
        auto [tVgV, tVsV] = tma_partition(
            params.tma_load_V, _0{}, Layout<_1>{},
            group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};

        uint16_t mcast_mask_qdo = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int n = 0; n < size<1>(block_layout); ++n) {
                mcast_mask_qdo |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
            }
        }
        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{};
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= uint16_t(1) <<
                    block_layout(m, cluster_local_block_id.y, _0{});
            }
        }

        int m_block = m_block_min;

        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            pipeline_q.producer_acquire(smem_pipe_write);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
            copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
                 gLSE(_, m_block), sLSE(_, smem_pipe_write.index()));
        }

        if (lane_predicate) {
            // Copy K tile and V tile from GMEM to SMEM. The clustered path
            // arms both the local and peer full barriers before issuing the
            // complete-box multicast transaction owned by each rank.
            if constexpr (HeadPairClusterKVReuse) {
                if (cluster_local_block_id.x == 0) {
                    pipeline_k.producer_acquire(smem_pipe_write_kv);
                    copy(params.tma_load_K.with(
                             *pipeline_k.producer_get_barrier(
                                 smem_pipe_write_kv),
                             mcast_mask_kv),
                         tKgK, tKsK);
                }
                if (cluster_local_block_id.x == 1) {
                    pipeline_v.producer_acquire(smem_pipe_write_kv);
                    copy(params.tma_load_V.with(
                             *pipeline_v.producer_get_barrier(
                                 smem_pipe_write_kv),
                             mcast_mask_kv),
                         tVgV, tVsV);
                }
                ++smem_pipe_write_kv;
            } else {
                shared_storage.pipelines.kv.barrier_KV.arrive_and_expect_tx(
                    TmaTransactionBytesK + TmaTransactionBytesV);
                copy(params.tma_load_K.with(
                         reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                             shared_storage.pipelines.kv.barrier_KV),
                         mcast_mask_kv),
                     tKgK, tKsK);
                copy(params.tma_load_V.with(
                         reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                             shared_storage.pipelines.kv.barrier_KV),
                         mcast_mask_kv),
                     tVgV, tVsV);
            }

            #pragma unroll (kHeadDim < 256 && kHeadDimV < 256 ? 2 : 1)
            for (; m_block < m_block_max - 1; ++m_block) {
                // Equal Q/dO stage counts share a pipeline state.
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
                copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                     gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                ++smem_pipe_write;
                pipeline_q.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
                copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
                     gLSE(_, m_block + 1), sLSE(_, smem_pipe_write.index()));
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
            pipeline_do.producer_acquire(smem_pipe_write_do_cur);
            copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
            copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                 gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
            ++smem_pipe_write;
        }
        if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write) {
        static_assert(Q_dO_same_stages, "Q and dO must have the same number of stages");
        // producer_tail increments its pipeline-state argument.
        PipelineState smem_pipe_write_do = smem_pipe_write;
        // Issue the epilogue waits
        if (cute::elect_one_sync()) {
            // Wait for every pipeline stage to be released or unused.
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
              MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v,
              PipelineState& smem_pipe_write,
              PipelineState_dO& smem_pipe_write_do,
              PipelineStateKV& smem_pipe_write_kv) {
        // Issue the epilogue waits
        if (cute::elect_one_sync()) {
            // Wait for every pipeline stage to be released or unused.
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
            if constexpr (HeadPairClusterKVReuse) {
                if (cute::block_rank_in_cluster() == 0) {
                    pipeline_k.producer_tail(smem_pipe_write_kv);
                }
                if (cute::block_rank_in_cluster() == 1) {
                    pipeline_v.producer_tail(smem_pipe_write_kv);
                }
            }
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_dq_for_q_head(Params const& params,
             SharedStorage &shared_storage,
             cute::tuple<int32_t, int32_t, int32_t> block_coord,
             cute::tuple<int32_t, int32_t> m_block_range,
             int const bidh
             ) {
        if constexpr (!dQacc_use_TMA) { return; }

        int const n_block = get<0>(block_coord);
        int const bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        int const m_block_min = get<0>(m_block_range);
        int const m_block_max = get<1>(m_block_range);
        // It's possible to have m_block_max <= m_block_min. In the no-seg
        // local deterministic path the semaphore order still follows the dense
        // n-block range, but the segment path compresses that order below and
        // can skip empty ranges entirely.
        if constexpr ((Is_causal || Is_local || Varlen) &&
                      !(Is_local && Deterministic && !Has_segment)) {
            if (m_block_max <= m_block_min) { return; }
        }

        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        static constexpr int dQ_TMA_num_bytes = CUTE_STATIC_V(size<0>(sdQ)) * sizeof(ElementAccum);

        bool const is_varlen = Varlen && params.cu_seqlens_q;
        ElementAccum* dq_accum_ptr = params.ptr_dQaccum;
        if constexpr (SplitDQ) {
            dq_accum_ptr += int64_t(n_block % params.dq_slots) * params.dq_slot_stride;
        }
        Tensor mdQaccum = make_tensor(make_gmem_ptr(dq_accum_ptr),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdQaccum_ = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum = cute::flat_divide(gdQaccum_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)

        int const num_batch = params.num_batch;
        int const num_head = get<2>(params.shape_Q);
        int *lock_ptr = !Deterministic ? nullptr : params.dq_semaphore + bidb * num_head + bidh;
        if constexpr (!Is_local && !Has_segment && !Varlen) {
            using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
            bool const lane_predicate = cute::elect_one_sync();
            constexpr int kBlockMStore = get<0>(TileShape_MNK{});
            constexpr int kBlockNStore = get<1>(TileShape_MNK{});
            int const n_block_global_max =
                cute::ceil_div(seqlen_info.seqlen_k, kBlockNStore);
            #pragma unroll 2
            for (int m_block = m_block_min; m_block < m_block_max; ++m_block) {
                if constexpr (Deterministic) {
                    if constexpr(Is_causal) {
                        int const n_block_max_for_m_block = std::min(
                            n_block_global_max,
                            cute::ceil_div(
                                (m_block + 1) * kBlockMStore +
                                    seqlen_info.seqlen_k - seqlen_info.seqlen_q +
                                    params.window_size_right_for_head(bidh),
                                kBlockNStore));
                        Barrier::wait_eq(
                            lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp,
                            m_block * num_batch * num_head,
                            n_block_max_for_m_block - 1 - n_block);
                    } else {
                        Barrier::wait_eq(
                            lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp,
                            m_block * num_batch * num_head, n_block);
                    }
                }
                #pragma unroll
                for (int warpgroup_idx = 0;
                     warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
                    cutlass::arch::NamedBarrier::sync(
                        cutlass::NumThreadsPerWarpGroup +
                            cutlass::NumThreadsPerWarp,
                        static_cast<uint32_t>(AttentionBwdNamedBarrier::dQFullWG1) +
                            warpgroup_idx);
                    if (lane_predicate) {
                        SM90_BULK_REDUCE_ADD::copy(
                            raw_pointer_cast(sdQ(_, warpgroup_idx).data()),
                            raw_pointer_cast(
                                gdQaccum(_, warpgroup_idx, m_block).data()),
                            dQ_TMA_num_bytes,
                            static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
                        tma_store_arrive();
                    }
                }
                for_each(
                    make_int_sequence<NumMmaWarpGroups>{},
                    [&] (auto warpgroup_idx) {
                        if (lane_predicate) {
                            tma_store_wait<
                                NumMmaWarpGroups - 1 -
                                CUTE_STATIC_V(warpgroup_idx)>();
                        }
                        cutlass::arch::NamedBarrier::arrive(
                            cutlass::NumThreadsPerWarpGroup +
                                cutlass::NumThreadsPerWarp,
                            static_cast<uint32_t>(
                                AttentionBwdNamedBarrier::dQEmptyWG1) +
                                warpgroup_idx);
                    });
                if constexpr (Deterministic) {
                    Barrier::arrive_inc(
                        lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp,
                        m_block * num_batch * num_head);
                }
            }
        } else {
        xattn::ops::attention::hopper::semantics::AttentionMaskAdapter<
            UseDirectChunkMask, kBlockM, kBlockN,
            false /*PackGQA*/, TiledMmaSdP,
            SdP_swapAB, Has_segment>
            segment_mask(
                0 /*thread_idx*/, seqlen_info.seqlen_q,
                seqlen_info.seqlen_k, params.window_size_left,
                params.window_size_right_for_head(bidh),
                0 /*sink_token_length*/,
                params.attention_chunk_divmod,
                params.qhead_per_khead_divmod, params.q_segment_idx,
                params.k_segment_idx, params.k_segment_len, bidb,
                params.get_chunk_position_offset(
                    bidb, seqlen_info.seqlen_q, seqlen_info.seqlen_k),
                Has_segment ? params.q_chunk_positions : nullptr,
                params.reset_attention_chunk_divmod);
        auto segment_relation_for_m_block = [&](int const m_block_cur) {
            if constexpr (Has_segment) {
                return segment_mask.segment_tile_relation_valid(
                    m_block_cur, n_block);
            } else {
                return int(SegmentPartition::kFull);
            }
        };
        constexpr int kBlockMStore = get<0>(TileShape_MNK{});
        constexpr int kBlockNStore = get<1>(TileShape_MNK{});
        auto deterministic_order_count_for_m_block = [&](int const m_block_cur) {
            if constexpr (Is_local && Deterministic && Has_segment) {
                int const num_m_blocks =
                    cute::ceil_div(seqlen_info.seqlen_q, kBlockMStore);
                int const range_idx =
                    (bidb * num_m_blocks + m_block_cur) * 2;
                int const n_block_min_for_m =
                    params.deterministic_segment_ranges[range_idx];
                int const n_block_max_for_m =
                    params.deterministic_segment_ranges[range_idx + 1];
                return cute::make_tuple(
                    n_block - n_block_min_for_m,
                    n_block_max_for_m - n_block_min_for_m);
            } else if constexpr (Is_local) {
                auto [n_block_min_for_m, n_block_max_for_m] =
                    AttentionRange::fwd_n_block_range(
                        seqlen_info, m_block_cur, bidb, 0 /*split_idx*/,
                        1 /*num_splits*/, params.window_size_left,
                        params.window_size_right_for_head(bidh),
                        params.attention_chunk_divmod,
                        params.qhead_per_khead_divmod,
                        params.get_chunk_position_offset(
                            bidb, seqlen_info.seqlen_q,
                            seqlen_info.seqlen_k));
                if constexpr (Has_segment) {
                    auto [segment_n_block_min, segment_n_block_max] =
                        SegmentPartition::fwd_n_block_range(
                            n_block_min_for_m, n_block_max_for_m,
                            m_block_cur, seqlen_info.seqlen_q,
                            seqlen_info.seqlen_k,
                            params.q_segment_idx, params.k_segment_idx,
                            params.k_segment_len, bidb);
                    n_block_min_for_m = segment_n_block_min;
                    n_block_max_for_m = segment_n_block_max;
                }
                return cute::make_tuple(
                    n_block - n_block_min_for_m,
                    n_block_max_for_m - n_block_min_for_m);
            } else {
                return cute::make_tuple(n_block, n_block + 1);
            }
        };
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
        bool const lane_predicate = cute::elect_one_sync();
        int m_block = m_block_min;
        int n_block_global_max =
            cute::ceil_div(seqlen_info.seqlen_k, kBlockNStore);
        #pragma unroll 2
        for (; m_block < m_block_max; ++m_block) {
            int const segment_relation = segment_relation_for_m_block(m_block);
            if constexpr (Has_segment) {
                if (segment_relation == SegmentPartition::kNoOverlap) {
                    continue;
                }
            }
            int deterministic_order = 0;
            int deterministic_count = 0;
            if constexpr (Deterministic) {
                auto deterministic_order_count =
                    deterministic_order_count_for_m_block(m_block);
                deterministic_order = get<0>(deterministic_order_count);
                deterministic_count = get<1>(deterministic_order_count);
                if constexpr (Has_segment) {
                    if (deterministic_order < 0 ||
                        deterministic_order >= deterministic_count ||
                        deterministic_count <= 0) {
                        continue;
                    }
                }
                bool const needs_deterministic_sync =
                    (!SplitDQ || params.dq_slots == 1) &&
                    !(Has_segment && deterministic_count <= 1);
                if (needs_deterministic_sync) {
                    if constexpr(Is_causal) {
                        int n_block_max_for_m_block = std::min(
                            n_block_global_max,
                            cute::ceil_div(
                                (m_block + 1) * kBlockMStore +
                                    seqlen_info.seqlen_k - seqlen_info.seqlen_q +
                                    params.window_size_right_for_head(bidh),
                                kBlockNStore));
                        Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block_max_for_m_block - 1 - n_block);
                    } else {
                        Barrier::wait_eq(
                            lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp,
                            m_block * num_batch * num_head,
                            deterministic_order);
                    }
                } else {
                    (void)n_block_global_max;
                }
            }
            if (segment_relation != SegmentPartition::kNoOverlap) {
                #pragma unroll
                for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
                    cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(AttentionBwdNamedBarrier::dQFullWG1) + warpgroup_idx /*id*/);  // sdQ full, to be written to gmem
                    if (lane_predicate) {
                        SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdQ(_, warpgroup_idx).data()), raw_pointer_cast(gdQaccum(_, warpgroup_idx, m_block).data()), dQ_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
                        tma_store_arrive();
                    }
                }
                // for_each keeps warpgroup_idx typed as Int<x>.
                for_each(make_int_sequence<NumMmaWarpGroups>{}, [&] (auto warpgroup_idx) {
                    if (lane_predicate) { tma_store_wait<NumMmaWarpGroups - 1 - CUTE_STATIC_V(warpgroup_idx)>(); }
                    cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(AttentionBwdNamedBarrier::dQEmptyWG1) + warpgroup_idx /*id*/);  // sdQ empty, ready to be written to
                });
            }
            if constexpr (Deterministic) {
                bool const needs_deterministic_sync =
                    (!SplitDQ || params.dq_slots == 1) &&
                    !(Has_segment && deterministic_count <= 1);
                if (needs_deterministic_sync) {
                    Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
                }
            }
        }
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_dq(Params const& params,
             SharedStorage &shared_storage,
             cute::tuple<int32_t, int32_t, int32_t> block_coord,
             cute::tuple<int32_t, int32_t> m_block_range) {
        if constexpr (!dQacc_use_TMA) {
            return;
        } else if constexpr (ReaderPairKVReuse) {
            int const kv_head = get<1>(block_coord);
            int const readers_per_kv_head =
                params.qhead_per_khead_divmod.divisor;
            CUTLASS_PRAGMA_NO_UNROLL
            for (int reader = 0; reader < readers_per_kv_head; ++reader) {
                int const q_head = q_head_for_reader(
                    kv_head, reader, readers_per_kv_head);
                store_dq_for_q_head(
                    params, shared_storage, block_coord, m_block_range,
                    q_head);
            }
        } else if constexpr (HeadPairParallel) {
            int const kv_head = get<1>(block_coord);
            int const q_head = q_head_for_reader(
                kv_head, head_pair_reader_index(),
                params.qhead_per_khead_divmod.divisor);
            store_dq_for_q_head(
                params, shared_storage, block_coord, m_block_range,
                q_head);
        } else {
            store_dq_for_q_head(
                params, shared_storage, block_coord, m_block_range,
                get<1>(block_coord));
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if constexpr (dQacc_use_TMA) {
            if (warp_idx_in_warpgroup == 0) {
                cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(AttentionBwdNamedBarrier::dQEmptyWG1) - 1 + flash::canonical_warp_group_idx_nosync() /*id*/);  // sdQ empty, ready to be written to
            }
        }
    }

    template <bool InitializeAccumulators, bool FinalizeAccumulators,
              typename SharedStorage, typename FrgTensordK,
              typename FrgTensordV>
    CUTLASS_DEVICE bool
    mma_for_q_head(Params const& params,
        MainloopPipeline pipeline_q,
        MainloopPipeline_dO pipeline_do,
        PipelineState& smem_pipe_read,
        PipelineState_dO& smem_pipe_read_do,
        FrgTensordK& tdKrdK,
        FrgTensordV& tdVrdV,
        int thread_idx,
        int &work_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        cute::tuple<int32_t, int32_t> m_block_range,
        SharedStorage& shared_storage,
        int const bidh
        ) {
        static_assert(is_rmem<FrgTensordK>::value, "dK tensor must be rmem resident.");
        static_assert(is_rmem<FrgTensordV>::value, "dV tensor must be rmem resident.");

        int n_block = get<0>(block_coord);
        int bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = m_block_range;
        // It's possible to have m_block_max <= m_block_min. Exit early
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) { return false; }
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdOt{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutKt{});
        Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdS{});
        Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
        Tensor sPt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdSt{});
        Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
        Tensor sPResidual = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p_residual.data()), SmemLayoutPdS{});
        Tensor sPResidual_pi = cute::as_position_independent_swizzle_tensor(sPResidual);
        Tensor sPResidualt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p_residual.data()), SmemLayoutPdSt{});
        Tensor sPResidualt_pi = cute::as_position_independent_swizzle_tensor(sPResidualt);
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS{});
        Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdSt{});
        Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});

        static_assert(stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
                      stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
                      size<0>(typename TiledMmaSdP::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                      size<0>(typename TiledMmaSdP::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        constexpr int MmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        Layout warp_group_thread_layout_dq = make_layout(make_shape(Int<NumMmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaSdP tiled_mma_SdP;
        using TiledMmadP = std::conditional_t<!Mma_dP_is_RS, TiledMmadPSS, TiledMmadPRS>;
        TiledMmadP tiled_mma_dP;
        TiledMmadK tiled_mma_dK;
        TiledMmadV tiled_mma_dV;
        TiledMmadQ tiled_mma_dQ;

        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout_dq(warp_group_idx));

        auto smem_tiled_copy_PdS = make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
        auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);

        R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);

        // Template-dependent fragment partitioning.
        Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
        Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
        Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_dP, sdO);
        Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);
        Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dV, sdOt);
        Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dK, sQt);
        Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
        Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

        Tensor tPsP = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP_pi, sPt_pi));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor tPsPResidual = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sPResidual_pi, sPResidualt_pi));
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_pi, sdSt_pi));      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Select row or column indices from ((2, 2, V), MMA_M, MMA_N, PIPE)
        // according to SdP_swapAB.
        Tensor tLSEsLSE = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),  // (2, MMA_M, PIPE)
            group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _, _)));  // (2, V, MMA_N, PIPE)
        Tensor tLSEsdPsum = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),
            group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _, _)));
        // Distribute row statistics across eight threads.
        static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tLSEsLSE))::value, 8);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        // Direct atomicAdd path for gdQaccum.
        bool const is_varlen = Varlen && params.cu_seqlens_q;
        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdQaccum_ = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum = cute::flat_divide(gdQaccum_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)
        // Reuse the dQ shared-memory copy partition.
        Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);

        xattn::ops::attention::hopper::semantics::AttentionMaskAdapter<
            UseDirectChunkMask, kBlockM, kBlockN,
            false /*PackGQA*/, TiledMmaSdP,
            SdP_swapAB, Has_segment,
            !Varlen && !Has_segment && !Deterministic &&
                !SeparateMaskingIterations,
            UseDirectChunkTile> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left,
            params.window_size_right_for_head(bidh),
            0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod,
            Has_segment ? params.q_segment_idx : nullptr,
            Has_segment ? params.k_segment_idx : nullptr,
            Has_segment ? params.k_segment_len : 0,
            bidb,
            params.get_chunk_position_offset(bidb, seqlen_q, seqlen_k),
            Has_segment ? params.q_chunk_positions : nullptr,
            params.reset_attention_chunk_divmod
        );

        int m_block = m_block_min;

        if constexpr (InitializeAccumulators) {
            clear(tdKrdK);
            clear(tdVrdV);
            if constexpr (!HeadPairClusterKVReuse) {
                cutlass::ConsumerToken barrier_token =
                    static_cast<cutlass::BarrierStatus>(
                        shared_storage.pipelines.kv.barrier_KV.try_wait(
                            work_idx % 2));
                if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
                    shared_storage.pipelines.kv.barrier_KV.wait(work_idx % 2);
                }
            }
        }

        if constexpr (Mma_dP_is_RS) {
            using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
            auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
            auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
            Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
            Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(cute::as_position_independent_swizzle_tensor(sV));
            cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
        }

        auto bwd_step = [&](int m_block, auto mask_fn) {
            int segment_relation = SegmentPartition::kFull;
            if constexpr (Has_segment) {
                segment_relation = mask.segment_tile_relation_valid(
                    m_block, n_block);
                if (segment_relation == SegmentPartition::kNoOverlap) {
                    consumer_wait(pipeline_q, smem_pipe_read);
                    PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
                    consumer_wait(pipeline_do, smem_pipe_read_do_cur);
                    pipeline_do.consumer_release(smem_pipe_read_do_cur);
                    pipeline_q.consumer_release(smem_pipe_read);
                    ++smem_pipe_read;
                    if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
                    return;
                }
            }
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            consumer_wait(pipeline_q, smem_pipe_read);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
            Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(make_fragment_like(tLSEsLSE(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            if constexpr (!ShuffleLSE) {
                cute::copy(tLSEsLSE(_, smem_pipe_read.index()), tLSErLSE);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    // Padded sLSE storage covers unused out-of-range reads.
                    tLSErLSE(i) = tLSEsLSE((thread_idx % 32) / 4 + i * 8, smem_pipe_read.index());
                }
            }
            Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            consumer_wait(pipeline_do, smem_pipe_read_do_cur);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_dP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);
            warpgroup_wait<1>();
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }

            // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
            // Apply dtanh before masking.
            auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();
            mask_fn(tSrS, m_block, segment_relation);
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                float const lse_scaled = [&] {
                    if constexpr (!ShuffleLSE) return tLSErLSE(mi);
                    else return __shfl_sync(0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    scores(mi, ni) = flash::attention_exp2f<UseFastExp2>(
                        scores(mi, ni) * params.softmax_scale_log2 - lse_scaled);
                }
            }

            Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(make_fragment_like(tLSEsdPsum(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            if constexpr (!ShuffledPsum) {
                cute::copy(tLSEsdPsum(_, smem_pipe_read_do_cur.index()), tLSErdPsum);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    tLSErdPsum(i) = tLSEsdPsum((thread_idx % 32) / 4 + i * 8, smem_pipe_read_do_cur.index());
                }
            }

            warpgroup_wait<0>();
            // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                float const dP_sum_cur = [&] {
                    if constexpr (!ShuffledPsum) return tLSErdPsum(mi);
                    else return __shfl_sync(0xffffffff, tLSErdPsum(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum_cur);
                    if constexpr (Has_softcap) { dS(mi, ni) *= dtanh(mi, ni); }
                }
            }

            // Convert scores from fp32 to fp16/bf16
            Tensor rP = make_tensor_like<Element>(tSrS);
            flash::convert_type_out(tSrS, rP);
            Tensor rPResidual = make_tensor_like<Element>(tSrS);
            if constexpr (TwoComponentDV) {
                #pragma unroll
                for (int i = 0; i < size(rP); ++i) {
                    tSrS(i) -= static_cast<ElementAccum>(rP(i));
                }
                flash::convert_type_out(tSrS, rPResidual);
            }
            if constexpr (!Mma_dKV_is_RS) {
                // Wait for the preceding P read before overwriting shared storage.
                if constexpr (kStages_dS == 1) {
                    cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(AttentionBwdNamedBarrier::PdS) /*id*/);
                }
                Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);     // ((Atom,AtomNum), MMA_N, MMA_N)
                cute::copy(smem_tiled_copy_PdS, tPaP, tPsP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));
                if constexpr (TwoComponentDV) {
                    Tensor tPaPResidual = smem_thr_copy_PdS.retile_S(rPResidual);
                    cute::copy(smem_tiled_copy_PdS, tPaPResidual, tPsPResidual(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));
                }
            }
            Tensor rdS = make_tensor_like<Element>(tdPrdP);
            flash::convert_type_out(tdPrdP, rdS);
            // Synchronize P writes for !Mma_dKV_is_RS and preceding dS reads
            // for Mma_dKV_is_RS. Double-buffered dS uses the loop-end barrier.
            if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(AttentionBwdNamedBarrier::PdS) /*id*/);
            }
            // hdim64 stores dS before the dV GEMM.
            Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));

            if constexpr (!Slice_dQKV_Mma) {
                // Unsliced dQ MMA path.
                if constexpr (Mma_dKV_is_RS) {
                    Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadV>(tSrS.layout()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                    if constexpr (TwoComponentDV) {
                        Tensor tdVrPResidual = make_tensor(rPResidual.data(), convert_layout_acc_Aregs<TiledMmadV>(tSrS.layout()));
                        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dV, tdVrPResidual, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                    }
                } else {
                    Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dV, sPt);
                    Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                    if constexpr (TwoComponentDV) {
                        Tensor tdVrPResidual = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dV, sPResidualt);
                        Tensor tdVrPResidual_cur = tdVrPResidual(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dV, tdVrPResidual_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                    }
                }
                // SMEM fence to make sure sdS is written before it's read by WGMMA
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(AttentionBwdNamedBarrier::PdS) /*id*/);
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO

                if constexpr (Mma_dKV_is_RS) {
                    Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadK>(tdPrdP.layout()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dK, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                } else {
                    Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dK, sdSt);
                    Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dK, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                }
                if constexpr (dQacc_use_TMA) {
                    int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
                    cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(AttentionBwdNamedBarrier::dQEmptyWG1) + warp_group_idx /*id*/);  // sdQ full, to be written to gmem
                    Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
                    cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(AttentionBwdNamedBarrier::dQFullWG1) + warp_group_idx /*id*/);  // sdQ full, to be written to gmem
                } else {
                    // Reuse the dQ shared-memory copy partition.
                    Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
                    Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
                    static_assert(CUTE_STATIC_V(size(tdQrdQ_atomic)) == CUTE_STATIC_V(size(tdQgdQaccum_atomic)));
                    #pragma unroll
                    for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
                }

            } else {  // Slice_dQKV_Mma

                static_assert(!(Slice_dQKV_Mma && Mma_dKV_is_RS));
                static_assert(kHeadDim == kHeadDimV, "Slice_dQKV_Mma requires D == V");
                Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dV, sPt);
                Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                if constexpr (TwoComponentDV) {
                    Tensor tdVrPResidual = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dV, sPResidualt);
                    Tensor tdVrPResidual_cur = tdVrPResidual(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dV, tdVrPResidual_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                }

                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(AttentionBwdNamedBarrier::PdS) /*id*/);
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                if constexpr (TwoComponentDV) {
                    Tensor tdVrPResidual = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dV, sPResidualt);
                    Tensor tdVrPResidual_cur = tdVrPResidual(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dV, tdVrPResidual_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                }
                Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
                Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
                #pragma unroll
                for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

                Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dK, sdSt);
                Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dK, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO

                flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB, /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                #pragma unroll
                for (int i = size(tdQrdQ_atomic) / 2;  i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dK, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
            }

            warpgroup_wait<0>();
            pipeline_q.consumer_release(smem_pipe_read);   // release Q
            ++smem_pipe_read;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
        };

        // Separate causal-mask iterations.
        if constexpr ((Is_causal || Is_local) && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block, int segment_relation) {
                mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);
                if constexpr (Has_segment) {
                    mask.apply_segment(tSrS, m_block, n_block,
                                       segment_relation);
                }
            };
            static constexpr int kBlockM = get<0>(TileShape_MNK{});
            int const m_block_masking_max =
                ((n_block + 1) * kBlockN - 1 + seqlen_q - seqlen_k -
                 (Is_sliding_chunk ? 0 :
                                            params.window_size_right_for_head(
                                                bidh))) /
                    kBlockM +
                1;
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < std::min(m_block_max, m_block_masking_max); ++m_block) {
                bwd_step(m_block, mask_fn);
            }
        }

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        bool const use_chunk_range =
            Is_sliding_chunk &&
            (!Has_segment ||
             params.attention_chunk_divmod.divisor > 0);
        int m_block_max_before_local_mask = m_block_max;
        if constexpr (Is_local && SeparateMaskingIterations) {
            if (!use_chunk_range) {
                m_block_max_before_local_mask = std::min(
                    m_block_max_before_local_mask,
                    (n_block * kBlockN + seqlen_q - seqlen_k +
                     params.window_size_left) /
                        kBlockM);
            }
        }
        if constexpr (Is_local && SeparateMaskingIterations) {
            if (params.attention_chunk_divmod.divisor > 0) {
                int const m_idx_chunk_end =
                    flash::round_down(
                        params.attention_chunk_divmod,
                        n_block * kBlockN +
                            params.attention_chunk_divmod.divisor +
                            params.get_chunk_position_offset(
                                bidb, seqlen_q, seqlen_k)) -
                    params.get_chunk_position_offset(
                        bidb, seqlen_q, seqlen_k) +
                    params.attention_chunk_divmod.divisor + seqlen_q - seqlen_k;
                m_block_max_before_local_mask =
                    std::min(m_block_max_before_local_mask, m_idx_chunk_end / kBlockM);
            }
        }

        auto mask_fn = [&](auto& tSrS, int m_block, int segment_relation) {
            mask.template apply<true /*Seqlenk_mask*/, Is_causal && !SeparateMaskingIterations, Is_local && !SeparateMaskingIterations>(tSrS, m_block, n_block);
            if constexpr (Has_segment) {
                mask.apply_segment(tSrS, m_block, n_block,
                                   segment_relation);
            }
        };
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max_before_local_mask; ++m_block) {
            bwd_step(m_block, mask_fn);
        }

        if constexpr (Is_local && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block, int segment_relation) {
                mask.template apply<true /*Seqlenk_mask*/, false /*Causal_mask*/, Is_local>(tSrS, m_block, n_block);
                if constexpr (Has_segment) {
                    mask.apply_segment(tSrS, m_block, n_block,
                                       segment_relation);
                }
            };
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < m_block_max; ++m_block) {
                bwd_step(m_block, mask_fn);
            }
        }

        if constexpr (FinalizeAccumulators) {
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) {
                tdKrdK(i) *= params.softmax_scale;
            }
        }

        if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
        if constexpr (FinalizeAccumulators) {
            ++work_idx;
        }
        return true;
    }

    template <typename SharedStorage, typename FrgTensordK,
              typename FrgTensordV>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipeline pipeline_q,
        MainloopPipeline_dO pipeline_do,
        MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v,
        PipelineState& smem_pipe_read,
        PipelineState_dO& smem_pipe_read_do,
        PipelineStateKV& smem_pipe_read_kv,
        FrgTensordK& tdKrdK,
        FrgTensordV& tdVrdV,
        int thread_idx,
        int &work_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        cute::tuple<int32_t, int32_t> m_block_range,
        SharedStorage& shared_storage) {
        if constexpr (ReaderPairKVReuse) {
            int const kv_head = get<1>(block_coord);
            int const readers_per_kv_head =
                params.qhead_per_khead_divmod.divisor;
            if (get<1>(m_block_range) <= get<0>(m_block_range)) {
                return false;
            }
            clear(tdKrdK);
            clear(tdVrdV);
            cutlass::ConsumerToken barrier_token =
                static_cast<cutlass::BarrierStatus>(
                    shared_storage.pipelines.kv.barrier_KV.try_wait(
                        work_idx % 2));
            if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
                shared_storage.pipelines.kv.barrier_KV.wait(work_idx % 2);
            }
            // The runtime loop shares one K/V tile across mapped readers and
            // emits one copy of the reader MMA body in SASS.
            CUTLASS_PRAGMA_NO_UNROLL
            for (int reader = 0; reader < readers_per_kv_head; ++reader) {
                mma_for_q_head<false, false>(
                    params, pipeline_q, pipeline_do, smem_pipe_read,
                    smem_pipe_read_do, tdKrdK, tdVrdV, thread_idx,
                    work_idx, block_coord, m_block_range, shared_storage,
                    q_head_for_reader(
                        kv_head, reader, readers_per_kv_head));
            }
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) {
                tdKrdK(i) *= params.softmax_scale;
            }
            ++work_idx;
            return true;
        } else {
            int const scheduled_head = get<1>(block_coord);
            int const q_head = HeadPairParallel
                ? q_head_for_reader(
                      scheduled_head, head_pair_reader_index(),
                      params.qhead_per_khead_divmod.divisor)
                : scheduled_head;
            if constexpr (HeadPairClusterKVReuse) {
                if constexpr (Is_causal || Is_local || Varlen) {
                    if (get<1>(m_block_range) <= get<0>(m_block_range)) {
                        return false;
                    }
                }
                auto token_k =
                    pipeline_k.consumer_try_wait(smem_pipe_read_kv);
                auto token_v =
                    pipeline_v.consumer_try_wait(smem_pipe_read_kv);
                pipeline_k.consumer_wait(smem_pipe_read_kv, token_k);
                pipeline_v.consumer_wait(smem_pipe_read_kv, token_v);
            }
            bool const tile_valid = mma_for_q_head<true, true>(
                params, pipeline_q, pipeline_do, smem_pipe_read,
                smem_pipe_read_do, tdKrdK, tdVrdV, thread_idx, work_idx,
                block_coord, m_block_range, shared_storage,
                q_head);
            if constexpr (HeadPairClusterKVReuse) {
                if (tile_valid) {
                    pipeline_k.consumer_release(smem_pipe_read_kv);
                    pipeline_v.consumer_release(smem_pipe_read_kv);
                    ++smem_pipe_read_kv;
                }
            }
            return tile_valid;
        }
    }

};

} // namespace flash

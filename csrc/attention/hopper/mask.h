/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Modified by: Shicheng Wen (xattn adaptations).

#pragma once

#include <stdint.h>

#include <algorithm>

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "attention/hopper/semantics/segment.h"
#include "attention/cuda/detail/cute_compat.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma,
          bool SwapAB=false, bool HasSegment=false,
          bool DenseTileFastPath=false, bool IsSlidingChunk=false,
          bool DirectChunkTileFastPath=IsSlidingChunk,
          bool AllowReset=HasSegment,
          int BranchPairLogicalBlockM=0>
struct Mask
    : xattn::ops::attention::hopper::semantics::
          SegmentPartitionStorage<HasSegment> {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");
    static_assert(
        BranchPairLogicalBlockM == 0 ||
            kBlockM == 2 * BranchPairLogicalBlockM);
    static_assert(BranchPairLogicalBlockM == 0 || !PackGQA);
    static_assert(BranchPairLogicalBlockM == 0 || !SwapAB);
    using SegmentStorage = xattn::ops::attention::hopper::semantics::
        SegmentPartitionStorage<HasSegment>;
    using SegmentPartition = xattn::ops::attention::hopper::semantics::
        SegmentPartitionAdapter<kBlockM, kBlockN>;

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    int const right_boundary_offset;
    cutlass::FastDivmod const attention_chunk_divmod;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    int const chunk_position_offset;
    int const* const q_chunk_positions;
    cutlass::FastDivmod const reset_attention_chunk_divmod;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &attention_chunk_divmod,
         cutlass::FastDivmod const &qhead_per_khead_divmod,
         int64_t const* const q_segment_idx=nullptr,
         int64_t const* const k_segment_idx=nullptr,
         const int k_segment_len=0, const int bidb=0,
         const int chunk_position_offset=0,
         int const* const q_chunk_positions=nullptr,
         cutlass::FastDivmod const &reset_attention_chunk_divmod=
             cutlass::FastDivmod(1),
         const int right_boundary_offset=0)
        : SegmentStorage(q_segment_idx, k_segment_idx, k_segment_len, bidb)
        , thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , right_boundary_offset(right_boundary_offset)
        , attention_chunk_divmod(attention_chunk_divmod)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
        , chunk_position_offset(chunk_position_offset)
        , q_chunk_positions(q_chunk_positions)
        , reset_attention_chunk_divmod(reset_attention_chunk_divmod)
    {
    };

    CUTLASS_DEVICE
    bool use_direct_chunk() const {
        if constexpr (!IsSlidingChunk) { return false; }
        if constexpr (!AllowReset) { return true; }
        return attention_chunk_divmod.divisor > 0;
    }

    CUTLASS_DEVICE
    bool segment_tile_may_overlap(const int m_block, const int n_block) const {
        static_assert(HasSegment, "segment_tile_may_overlap requires HasSegment=true");
        if (this->q_segment_idx == nullptr || this->k_segment_idx == nullptr || this->k_segment_len <= 0) { return true; }
        return segment_tile_may_overlap_valid(m_block, n_block);
    }

    CUTLASS_DEVICE
    bool segment_tile_may_overlap_valid(const int m_block, const int n_block) const {
        static_assert(HasSegment, "segment_tile_may_overlap_valid requires HasSegment=true");
        return segment_tile_relation_valid(m_block, n_block) !=
            SegmentPartition::kNoOverlap;
    }

    CUTLASS_DEVICE
    int segment_tile_relation_valid(const int m_block, const int n_block) const {
        static_assert(HasSegment, "segment_tile_relation_valid requires HasSegment=true");
        return SegmentPartition::tile_relation(
            m_block, n_block, seqlen_q, seqlen_k,
            this->q_segment_idx, this->k_segment_idx,
            this->k_segment_len, this->bidb);
    }

    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        if (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }

        if constexpr (DenseTileFastPath && Local_mask) {
            static_assert(!PackGQA);
            int const m_tile_begin = m_block * kBlockM;
            int const m_tile_end = m_tile_begin + kBlockM;
            int const n_tile_begin = n_block * kBlockN;
            int const n_tile_end = n_tile_begin + kBlockN;
            if (m_tile_end <= seqlen_q &&
                (!Seqlenk_mask || n_tile_end <= seqlen_k)) {
                int const qk_offset = seqlen_k - seqlen_q;
                bool tile_is_valid =
                    n_tile_end - 1 <=
                    m_tile_begin + qk_offset + window_size_right +
                        right_boundary_offset;
                if constexpr (DirectChunkTileFastPath) {
                    int const chunk_begin =
                        flash::round_down(
                            attention_chunk_divmod,
                            m_tile_end - 1 + qk_offset +
                                chunk_position_offset) -
                        chunk_position_offset;
                    tile_is_valid &=
                        n_tile_begin >=
                        chunk_begin - attention_chunk_divmod.divisor;
                } else {
                    tile_is_valid &=
                        window_size_right + right_boundary_offset == 0 &&
                        n_tile_begin >=
                            m_tile_end - 1 + qk_offset - window_size_left;
                    if (attention_chunk_divmod.divisor > 0) {
                        int const chunk_begin =
                            flash::round_down(
                                attention_chunk_divmod,
                                m_tile_end - 1 + qk_offset +
                                    chunk_position_offset) -
                            chunk_position_offset;
                        tile_is_valid &=
                            n_tile_begin >=
                            chunk_begin - attention_chunk_divmod.divisor;
                    }
                }
                if (tile_is_valid) { return; }
            }
        }

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        // Compile-time column coordinates are normalized by the calling
        // thread's offset.
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        if constexpr (!Causal_mask && !Local_mask) {
            if constexpr (Seqlenk_mask) {  // Column-only mask.
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {  // mask based on both row and col
            if constexpr (!SwapAB) {
                // PackGQA divmod is distributed across threads in each row.
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                // Bounds are validated before use.
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                int const causal_row_offset =
                    1 + seqlen_k - n_block * kBlockN - seqlen_q -
                    thread_col_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_in_tile =
                            get<Row>(tScS_rowcol(m, _0{}));
                        int const branch = [&] {
                            if constexpr (BranchPairLogicalBlockM == 0) {
                                return 0;
                            } else {
                                return row_in_tile /
                                    BranchPairLogicalBlockM;
                            }
                        }();
                        int const row_idx = !PackGQA
                            ? (BranchPairLogicalBlockM == 0
                                  ? row_in_tile + m_block * kBlockM
                                  : row_in_tile -
                                        branch * BranchPairLogicalBlockM +
                                        m_block * BranchPairLogicalBlockM)
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const row_boundary_offset =
                            right_boundary_offset - branch;
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset + window_size_right +
                                  row_boundary_offset
                            : __viaddmin_s32(
                                  row_idx,
                                  causal_row_offset + window_size_right +
                                      row_boundary_offset,
                                  seqlenk_col_limit);
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const local_row_offset_left =
                        causal_row_offset - 1 - window_size_left;
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_in_tile =
                            get<Row>(tScS_rowcol(m, _0{}));
                        int const branch = [&] {
                            if constexpr (BranchPairLogicalBlockM == 0) {
                                return 0;
                            } else {
                                return row_in_tile /
                                    BranchPairLogicalBlockM;
                            }
                        }();
                        int const row_idx = !PackGQA
                            ? (BranchPairLogicalBlockM == 0
                                  ? row_in_tile + m_block * kBlockM
                                  : row_in_tile -
                                        branch * BranchPairLogicalBlockM +
                                        m_block * BranchPairLogicalBlockM)
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const local_row_offset_right =
                            causal_row_offset + window_size_right +
                            right_boundary_offset - branch;
                        int col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int col_limit_left;
                        if constexpr (IsSlidingChunk) {
                            col_limit_left = use_direct_chunk()
                                ? flash::round_down(
                                      attention_chunk_divmod,
                                      row_idx + seqlen_k - seqlen_q +
                                          chunk_position_offset) -
                                      chunk_position_offset -
                                      attention_chunk_divmod.divisor -
                                      n_block * kBlockN - thread_col_offset
                                : row_idx + local_row_offset_left;
                        } else {
                            col_limit_left =
                                row_idx + local_row_offset_left;
                            if (attention_chunk_divmod.divisor > 0) {
                                int const col_limit_left_chunk =
                                    flash::round_down(
                                        attention_chunk_divmod,
                                        row_idx + seqlen_k - seqlen_q +
                                            chunk_position_offset) -
                                    chunk_position_offset -
                                    attention_chunk_divmod.divisor -
                                    n_block * kBlockN - thread_col_offset;
                                col_limit_left =
                                    std::max(col_limit_left,
                                             col_limit_left_chunk);
                            }
                        }
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            bool const left_mask = IsSlidingChunk
                                ? col_idx < col_limit_left
                                : col_idx < col_limit_left &&
                                      col_idx >= col_limit_sink;
                            if (col_idx >= col_limit_right || left_mask) {
                                tSrS_rowcol(m, n) = -INFINITY;
                            }
                        }
                    }
                }
            } else {
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset =
                    seqlenk_col_limit - seqlen_q + m_block * kBlockM +
                    thread_row_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // col0 beyond the limit masks the full column.
                        int const row_limit_top =
                            col0 >= seqlenk_col_limit
                            ? kBlockM
                            : col0 - causal_row_offset - window_size_right -
                                  right_boundary_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    int const qk_offset = seqlen_k - seqlen_q;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // col0 beyond the limit masks the full column.
                        int const row_limit_top = col0 >= seqlenk_col_limit
                            ? kBlockM
                            : col0 - causal_row_offset -
                                  window_size_right - right_boundary_offset;
                        int row_limit_bot = col0 < col_limit_sink
                            ? kBlockM
                            : col0 - causal_row_offset + window_size_left;
                        if constexpr (IsSlidingChunk) {
                            if (use_direct_chunk()) {
                                int const col_idx =
                                    n_block * kBlockN + thread_col_offset +
                                    col0;
                                row_limit_bot =
                                    flash::round_up(
                                        attention_chunk_divmod,
                                        col_idx + 1 +
                                            chunk_position_offset) -
                                    chunk_position_offset +
                                    attention_chunk_divmod.divisor - 1 -
                                    qk_offset - m_block * kBlockM -
                                    thread_row_offset;
                            }
                        } else if (attention_chunk_divmod.divisor > 0 &&
                                   col0 >= col_limit_sink) {
                            int const col_idx =
                                n_block * kBlockN + thread_col_offset + col0;
                            int const row_limit_bot_chunk =
                                flash::round_up(
                                    attention_chunk_divmod,
                                    col_idx + 1 + chunk_position_offset) -
                                chunk_position_offset +
                                attention_chunk_divmod.divisor - 1 -
                                qk_offset - m_block * kBlockM -
                                thread_row_offset;
                            row_limit_bot =
                                std::min(row_limit_bot,
                                         row_limit_bot_chunk);
                        }
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            }
        }
    };

    template <bool Seqlenk_mask=false, typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply_local_right(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!SwapAB, "apply_local_right currently supports non-SwapAB layouts");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(t0ScS.layout()));

        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
        int mma_m_idx;
        if constexpr (PackGQA) {
            mma_m_idx = qhead_per_khead_divmod.divide(
                m_block * kBlockM + get<0>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
        }
        int const thread_col_offset = get<1>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
        int const local_row_offset_right = causal_row_offset +
            window_size_right + right_boundary_offset;
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = !PackGQA
                ? get<0>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                : __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
            int const col_limit_right = !Seqlenk_mask
                ? row_idx + local_row_offset_right
                : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                if (int(get<1>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) {
                    tSrS_rowcol(m, n) = -INFINITY;
                }
            }
        }
    }

    template <typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply_local_left(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!SwapAB, "apply_local_left currently supports non-SwapAB layouts");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(t0ScS.layout()));

        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
        int mma_m_idx;
        if constexpr (PackGQA) {
            mma_m_idx = qhead_per_khead_divmod.divide(
                m_block * kBlockM + get<0>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
        }
        int const thread_col_offset = get<1>(tScS_rowcol(_0{}, _0{}));
        int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
        int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
        int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = !PackGQA
                ? get<0>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                : __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
            int col_limit_left;
            if constexpr (IsSlidingChunk) {
                col_limit_left = use_direct_chunk()
                    ? flash::round_down(
                          attention_chunk_divmod,
                          row_idx + seqlen_k - seqlen_q +
                              chunk_position_offset) -
                          chunk_position_offset -
                          attention_chunk_divmod.divisor -
                          n_block * kBlockN - thread_col_offset
                    : row_idx + local_row_offset_left;
            } else {
                col_limit_left = row_idx + local_row_offset_left;
                if (attention_chunk_divmod.divisor > 0) {
                    int const col_limit_left_chunk =
                        flash::round_down(
                            attention_chunk_divmod,
                            row_idx + seqlen_k - seqlen_q +
                                chunk_position_offset) -
                        chunk_position_offset -
                        attention_chunk_divmod.divisor -
                        n_block * kBlockN - thread_col_offset;
                    col_limit_left =
                        std::max(col_limit_left, col_limit_left_chunk);
                }
            }
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const col_idx = int(get<1>(t0ScS_rowcol(_0{}, n)));
                bool const left_mask = IsSlidingChunk
                    ? col_idx < col_limit_left
                    : col_idx < col_limit_left &&
                          col_idx >= col_limit_sink;
                if (left_mask) {
                    tSrS_rowcol(m, n) = -INFINITY;
                }
            }
        }
    }

    template <typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply_segment(
            Tensor<Engine, Layout> &tSrS,
            const int m_block,
            const int n_block,
            const int segment_relation_in=-1) const {
        static_assert(HasSegment, "apply_segment requires HasSegment=true");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));

        int const segment_relation = segment_relation_in >= 0
            ? segment_relation_in
            : segment_tile_relation_valid(m_block, n_block);
        bool const apply_reset_chunk =
            reset_attention_chunk_divmod.divisor > 0 &&
            q_chunk_positions != nullptr;
        bool const check_segment =
            segment_relation != SegmentPartition::kFull;
        if (segment_relation == SegmentPartition::kNoOverlap) {
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    tSrS_rowcol(m, n) = -INFINITY;
                }
            }
            return;
        }
        if (segment_relation == SegmentPartition::kFull &&
            !apply_reset_chunk) {
            return;
        }

        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const k_segment_offset = this->k_segment_len == seqlen_k ? 0 : this->k_segment_len - seqlen_q;
        if constexpr (!SwapAB) {
            static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
            static constexpr int kRows = CUTE_STATIC_V(size<0>(tSrS_rowcol));
            static constexpr int kCols = CUTE_STATIC_V(size<1>(tSrS_rowcol));
            int mma_m_idx;
            if constexpr (PackGQA) {
                mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
            }
            #pragma unroll
            for (int m = 0; m < kRows; ++m) {
                int const q_pos = !PackGQA
                    ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                    : __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                bool const q_valid = q_pos < seqlen_q;
                int64_t const q_seg = q_valid && check_segment
                    ? this->q_segment_idx[this->bidb * seqlen_q + q_pos]
                    : int64_t(-1);
                int const q_chunk_pos = q_valid && apply_reset_chunk
                    ? q_chunk_positions[this->bidb * seqlen_q + q_pos]
                    : 0;
                int const k_chunk_left = apply_reset_chunk
                    ? q_pos + seqlen_k - seqlen_q - q_chunk_pos +
                          flash::round_down(
                              reset_attention_chunk_divmod, q_chunk_pos) -
                          reset_attention_chunk_divmod.divisor
                    : 0;
                #pragma unroll
                for (int n = 0; n < kCols; ++n) {
                    int const k_pos = n_block * kBlockN + thread_col_offset + int(get<Col>(t0ScS_rowcol(_0{}, n)));
                    int const k_segment_pos = k_segment_offset + k_pos;
                    bool const k_valid = k_pos < seqlen_k &&
                                         k_segment_pos >= 0 && k_segment_pos < this->k_segment_len;
                    int64_t const k_seg = k_valid && check_segment
                        ? this->k_segment_idx[this->bidb * this->k_segment_len + k_segment_pos]
                        : int64_t(-2);
                    if (!q_valid || !k_valid ||
                        (check_segment && q_seg != k_seg) ||
                        (apply_reset_chunk && k_pos < k_chunk_left)) {
                        tSrS_rowcol(m, n) = -INFINITY;
                    }
                }
            }
        } else {
            static constexpr int kRows = CUTE_STATIC_V(size<0>(tSrS_rowcol));
            static constexpr int kCols = CUTE_STATIC_V(size<1>(tSrS_rowcol));
            int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
            #pragma unroll
            for (int m = 0; m < kRows; ++m) {
                int const q_pos = thread_row_offset +
                                  int(get<Row>(t0ScS_rowcol(m, _0{}))) +
                                  m_block * kBlockM;
                bool const q_valid = q_pos < seqlen_q;
                int64_t const q_seg = q_valid && check_segment
                    ? this->q_segment_idx[this->bidb * seqlen_q + q_pos]
                    : int64_t(-1);
                int const q_chunk_pos = q_valid && apply_reset_chunk
                    ? q_chunk_positions[this->bidb * seqlen_q + q_pos]
                    : 0;
                int const k_chunk_left = apply_reset_chunk
                    ? q_pos + seqlen_k - seqlen_q - q_chunk_pos +
                          flash::round_down(
                              reset_attention_chunk_divmod, q_chunk_pos) -
                          reset_attention_chunk_divmod.divisor
                    : 0;
                #pragma unroll
                for (int n = 0; n < kCols; ++n) {
                    int const k_pos = n_block * kBlockN + thread_col_offset + int(get<Col>(t0ScS_rowcol(_0{}, n)));
                    int const k_segment_pos = k_segment_offset + k_pos;
                    bool const k_valid = k_pos < seqlen_k &&
                                         k_segment_pos >= 0 && k_segment_pos < this->k_segment_len;
                    int64_t const k_seg = k_valid && check_segment
                        ? this->k_segment_idx[this->bidb * this->k_segment_len + k_segment_pos]
                        : int64_t(-2);
                    if (!q_valid || !k_valid ||
                        (check_segment && q_seg != k_seg) ||
                        (apply_reset_chunk && k_pos < k_chunk_left)) {
                        tSrS_rowcol(m, n) = -INFINITY;
                    }
                }
            }
        }
    };

};

} // namespace flash

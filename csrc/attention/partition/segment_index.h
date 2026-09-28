#pragma once

#include <algorithm>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

namespace xattn {
namespace ops {
namespace attention {
namespace partition {

enum SegmentTileRelation : int {
    kSegmentTileNoOverlap = 0,
    kSegmentTileFull = 1,
    kSegmentTilePartial = 2
};

struct SegmentIndexView {
    int64_t const* q_segment_idx;
    int64_t const* k_segment_idx;
    int q_segment_len;
    int k_segment_len;
    int batch_count;
    int k_current_offset;

    CUTLASS_HOST_DEVICE
    constexpr SegmentIndexView(
        int64_t const* const q_segment_idx_ = nullptr,
        int64_t const* const k_segment_idx_ = nullptr,
        int const q_segment_len_ = 0,
        int const k_segment_len_ = 0,
        int const batch_count_ = 0,
        int const k_current_offset_ = 0)
        : q_segment_idx(q_segment_idx_)
        , k_segment_idx(k_segment_idx_)
        , q_segment_len(q_segment_len_)
        , k_segment_len(k_segment_len_)
        , batch_count(batch_count_)
        , k_current_offset(k_current_offset_) {}

    CUTLASS_HOST_DEVICE
    constexpr bool enabled() const {
        return q_segment_idx != nullptr && k_segment_idx != nullptr &&
            q_segment_len > 0 && k_segment_len > 0 && batch_count > 0;
    }

    CUTLASS_HOST_DEVICE
    constexpr int64_t q_segment(int const batch, int const q_pos) const {
        return q_segment_idx[batch * q_segment_len + q_pos];
    }

    CUTLASS_HOST_DEVICE
    constexpr int64_t k_segment(int const batch, int const k_pos) const {
        return k_segment_idx[
            batch * k_segment_len + k_current_offset + k_pos];
    }
};

CUTLASS_HOST_DEVICE
constexpr bool same_partition(
    int64_t const q_segment, int64_t const k_segment) {
    return q_segment == k_segment;
}

CUTLASS_HOST_DEVICE
constexpr bool partition_ranges_may_overlap(
    int64_t const q_segment_first, int64_t const q_segment_last,
    int64_t const k_segment_first, int64_t const k_segment_last) {
    return q_segment_first <= k_segment_last &&
        k_segment_first <= q_segment_last;
}

CUTLASS_DEVICE
int segment_lower_bound(
    int64_t const* const segment_idx,
    int const begin,
    int const end,
    int64_t const value) {
    int lo = begin;
    int hi = end;
    while (lo < hi) {
        int const mid = lo + ((hi - lo) >> 1);
        if (segment_idx[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

CUTLASS_DEVICE
int segment_upper_bound(
    int64_t const* const segment_idx,
    int const begin,
    int const end,
    int64_t const value) {
    int lo = begin;
    int hi = end;
    while (lo < hi) {
        int const mid = lo + ((hi - lo) >> 1);
        if (segment_idx[mid] <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <int kBlockM, int kBlockN>
CUTLASS_DEVICE
cute::tuple<int, int> segment_n_block_min_max(
    int n_block_min,
    int n_block_max,
    int const m_block,
    int const seqlen_q,
    int const seqlen_k,
    int64_t const* const q_segment_idx,
    int64_t const* const k_segment_idx,
    int const k_segment_len,
    int const bidb) {
    if (q_segment_idx == nullptr || k_segment_idx == nullptr || k_segment_len <= 0) {
        return {n_block_min, n_block_max};
    }

    int const q_start = m_block * kBlockM;
    int const q_end = q_start + kBlockM < seqlen_q ? q_start + kBlockM : seqlen_q;
    if (q_start >= q_end || n_block_min >= n_block_max) {
        return {n_block_min, n_block_min};
    }

    int const k_segment_offset = k_segment_len == seqlen_k ? 0 : k_segment_len - seqlen_q;
    int const k_pos_begin = n_block_min * kBlockN;
    int const k_pos_end_raw = n_block_max * kBlockN < seqlen_k ? n_block_max * kBlockN : seqlen_k;
    int const k_begin = k_segment_offset + k_pos_begin;
    int const k_end = k_segment_offset + k_pos_end_raw;
    if (k_begin < 0 || k_end > k_segment_len || k_begin >= k_end) {
        return {n_block_min, n_block_min};
    }

    int64_t q_seg_first = q_segment_idx[bidb * seqlen_q + q_start];
    int64_t q_seg_last = q_segment_idx[bidb * seqlen_q + q_end - 1];
    if (q_seg_last < q_seg_first) {
        int64_t const tmp = q_seg_first;
        q_seg_first = q_seg_last;
        q_seg_last = tmp;
    }

    int64_t const* const k_segment_base = k_segment_idx + bidb * k_segment_len;
    int const k_match_begin = segment_lower_bound(k_segment_base, k_begin, k_end, q_seg_first);
    int const k_match_end = segment_upper_bound(k_segment_base, k_match_begin, k_end, q_seg_last);
    if (k_match_end <= k_match_begin) {
        return {n_block_min, n_block_min};
    }

    int const k_pos_match_begin = k_match_begin - k_segment_offset;
    int const k_pos_match_end = k_match_end - k_segment_offset;
    int const new_n_block_min = std::max(n_block_min, k_pos_match_begin / kBlockN);
    int const new_n_block_max = std::min(n_block_max, (k_pos_match_end + kBlockN - 1) / kBlockN);
    return {new_n_block_min, new_n_block_max};
}

template <int kBlockM, int kBlockN>
CUTLASS_DEVICE
cute::tuple<int, int> segment_m_block_min_max(
    int m_block_min,
    int m_block_max,
    int const n_block,
    int const seqlen_q,
    int const seqlen_k,
    int64_t const* const q_segment_idx,
    int64_t const* const k_segment_idx,
    int const k_segment_len,
    int const bidb) {
    if (q_segment_idx == nullptr || k_segment_idx == nullptr || k_segment_len <= 0) {
        return {m_block_min, m_block_max};
    }

    int const q_pos_begin = m_block_min * kBlockM;
    int const q_pos_end_raw = m_block_max * kBlockM < seqlen_q ? m_block_max * kBlockM : seqlen_q;
    int const k_start = n_block * kBlockN;
    int const k_end = k_start + kBlockN < seqlen_k ? k_start + kBlockN : seqlen_k;
    if (q_pos_begin >= q_pos_end_raw || k_start >= k_end || m_block_min >= m_block_max) {
        return {m_block_min, m_block_min};
    }

    int const k_segment_offset = k_segment_len == seqlen_k ? 0 : k_segment_len - seqlen_q;
    int const k_segment_start = k_segment_offset + k_start;
    int const k_segment_end = k_segment_offset + k_end - 1;
    if (k_segment_start < 0 || k_segment_end >= k_segment_len) {
        return {m_block_min, m_block_min};
    }

    int64_t k_seg_first = k_segment_idx[bidb * k_segment_len + k_segment_start];
    int64_t k_seg_last = k_segment_idx[bidb * k_segment_len + k_segment_end];
    if (k_seg_last < k_seg_first) {
        int64_t const tmp = k_seg_first;
        k_seg_first = k_seg_last;
        k_seg_last = tmp;
    }

    int64_t const* const q_segment_base = q_segment_idx + bidb * seqlen_q;
    int const q_match_begin =
        segment_lower_bound(q_segment_base, q_pos_begin, q_pos_end_raw, k_seg_first);
    int const q_match_end =
        segment_upper_bound(q_segment_base, q_match_begin, q_pos_end_raw, k_seg_last);
    if (q_match_end <= q_match_begin &&
        (q_match_begin == q_pos_begin || q_match_begin == q_pos_end_raw ||
         q_match_begin % kBlockM == 0)) {
        return {m_block_min, m_block_min};
    }
    // Zero-contribution tiles spanning K-only segments still advance
    // the deterministic dQ semaphore within the forward bounding range.

    int const new_m_block_min = std::max(m_block_min, q_match_begin / kBlockM);
    int const new_m_block_max = std::min(m_block_max, (q_match_end + kBlockM - 1) / kBlockM);
    return {new_m_block_min, new_m_block_max};
}

template <int kBlockM, int kBlockN>
CUTLASS_DEVICE
int segment_tile_relation(
    int const m_block,
    int const n_block,
    int const seqlen_q,
    int const seqlen_k,
    int64_t const* const q_segment_idx,
    int64_t const* const k_segment_idx,
    int const k_segment_len,
    int const bidb) {
    // Segment ids are nondecreasing per sequence; Q/K absolute-position ids
    // are independent.
    int const q_start = m_block * kBlockM;
    int const q_end = q_start + kBlockM < seqlen_q ? q_start + kBlockM : seqlen_q;
    int const k_start = n_block * kBlockN;
    int const k_end = k_start + kBlockN < seqlen_k ? k_start + kBlockN : seqlen_k;
    if (q_start >= q_end || k_start >= k_end) {
        return kSegmentTileNoOverlap;
    }

    int const k_segment_offset = k_segment_len == seqlen_k ? 0 : k_segment_len - seqlen_q;
    int const k_segment_start = k_segment_offset + k_start;
    int const k_segment_end = k_segment_offset + k_end - 1;
    if (k_segment_start < 0 || k_segment_end >= k_segment_len) {
        return kSegmentTileNoOverlap;
    }

    int64_t const q_seg_first = q_segment_idx[bidb * seqlen_q + q_start];
    int64_t const q_seg_last = q_segment_idx[bidb * seqlen_q + q_end - 1];
    int64_t const k_seg_first = k_segment_idx[bidb * k_segment_len + k_segment_start];
    int64_t const k_seg_last = k_segment_idx[bidb * k_segment_len + k_segment_end];
    if (!partition_ranges_may_overlap(
            q_seg_first, q_seg_last, k_seg_first, k_seg_last)) {
        return kSegmentTileNoOverlap;
    }
    bool const is_full =
        q_seg_first == q_seg_last && q_seg_first == k_seg_first &&
        k_seg_first == k_seg_last;
    return is_full ? kSegmentTileFull : kSegmentTilePartial;
}

template <bool HasSegment>
struct SegmentMaskStorage {
    CUTLASS_DEVICE
    SegmentMaskStorage(
        int64_t const* const = nullptr,
        int64_t const* const = nullptr,
        const int = 0,
        const int = 0) {}
};

template <>
struct SegmentMaskStorage<true> {
    int64_t const* const q_segment_idx;
    int64_t const* const k_segment_idx;
    int const k_segment_len;
    int const bidb;

    CUTLASS_DEVICE
    SegmentMaskStorage(
        int64_t const* const q_segment_idx=nullptr,
        int64_t const* const k_segment_idx=nullptr,
        const int k_segment_len=0,
        const int bidb=0)
        : q_segment_idx(q_segment_idx)
        , k_segment_idx(k_segment_idx)
        , k_segment_len(k_segment_len)
        , bidb(bidb) {}
};

struct NoPartitionPolicy {
    static constexpr bool kHasPartition = false;
    static constexpr int kNoOverlap = kSegmentTileNoOverlap;
    static constexpr int kFull = kSegmentTileFull;
    static constexpr int kPartial = kSegmentTilePartial;

    CUTLASS_HOST_DEVICE
    static constexpr bool same_partition(
        int64_t const, int64_t const) {
        return true;
    }

    CUTLASS_DEVICE
    static cute::tuple<int, int> fwd_n_block_range(
        int const n_block_min, int const n_block_max) {
        return {n_block_min, n_block_max};
    }

    CUTLASS_DEVICE
    static cute::tuple<int, int> bwd_m_block_range(
        int const m_block_min, int const m_block_max) {
        return {m_block_min, m_block_max};
    }

    CUTLASS_DEVICE
    static constexpr int tile_relation() {
        return kFull;
    }
};

template <int kBlockM, int kBlockN>
struct SegmentPartitionPolicy {
    static constexpr bool kHasPartition = true;
    static constexpr int kNoOverlap = kSegmentTileNoOverlap;
    static constexpr int kFull = kSegmentTileFull;
    static constexpr int kPartial = kSegmentTilePartial;

    CUTLASS_HOST_DEVICE
    static constexpr bool same_partition(
        int64_t const q_segment, int64_t const k_segment) {
        return partition::same_partition(q_segment, k_segment);
    }

    CUTLASS_DEVICE
    static cute::tuple<int, int> fwd_n_block_range(
        int const n_block_min, int const n_block_max, int const m_block,
        int const seqlen_q, int const seqlen_k,
        int64_t const* const q_segment_idx,
        int64_t const* const k_segment_idx, int const k_segment_len,
        int const bidb) {
        return segment_n_block_min_max<kBlockM, kBlockN>(
            n_block_min, n_block_max, m_block, seqlen_q, seqlen_k,
            q_segment_idx, k_segment_idx, k_segment_len, bidb);
    }

    CUTLASS_DEVICE
    static cute::tuple<int, int> fwd_n_block_range(
        int const n_block_min, int const n_block_max, int const m_block,
        int const seqlen_q, int const seqlen_k,
        SegmentIndexView const& view, int const bidb) {
        return fwd_n_block_range(
            n_block_min, n_block_max, m_block, seqlen_q, seqlen_k,
            view.q_segment_idx, view.k_segment_idx, view.k_segment_len,
            bidb);
    }

    CUTLASS_DEVICE
    static cute::tuple<int, int> bwd_m_block_range(
        int const m_block_min, int const m_block_max, int const n_block,
        int const seqlen_q, int const seqlen_k,
        int64_t const* const q_segment_idx,
        int64_t const* const k_segment_idx, int const k_segment_len,
        int const bidb) {
        return segment_m_block_min_max<kBlockM, kBlockN>(
            m_block_min, m_block_max, n_block, seqlen_q, seqlen_k,
            q_segment_idx, k_segment_idx, k_segment_len, bidb);
    }

    CUTLASS_DEVICE
    static cute::tuple<int, int> bwd_m_block_range(
        int const m_block_min, int const m_block_max, int const n_block,
        int const seqlen_q, int const seqlen_k,
        SegmentIndexView const& view, int const bidb) {
        return bwd_m_block_range(
            m_block_min, m_block_max, n_block, seqlen_q, seqlen_k,
            view.q_segment_idx, view.k_segment_idx, view.k_segment_len,
            bidb);
    }

    CUTLASS_DEVICE
    static int tile_relation(
        int const m_block, int const n_block, int const seqlen_q,
        int const seqlen_k, int64_t const* const q_segment_idx,
        int64_t const* const k_segment_idx, int const k_segment_len,
        int const bidb) {
        return segment_tile_relation<kBlockM, kBlockN>(
            m_block, n_block, seqlen_q, seqlen_k, q_segment_idx,
            k_segment_idx, k_segment_len, bidb);
    }

    CUTLASS_DEVICE
    static int tile_relation(
        int const m_block, int const n_block, int const seqlen_q,
        int const seqlen_k, SegmentIndexView const& view,
        int const bidb) {
        return tile_relation(
            m_block, n_block, seqlen_q, seqlen_k, view.q_segment_idx,
            view.k_segment_idx, view.k_segment_len, bidb);
    }
};

}  // namespace partition
}  // namespace attention
}  // namespace ops
}  // namespace xattn

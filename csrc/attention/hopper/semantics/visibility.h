#pragma once

#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include "cute/tensor.hpp"
#include "attention/hopper/block.h"

namespace xattn {
namespace ops {
namespace attention {
namespace hopper {
namespace semantics {

template <
    class SeqlenInfo, int kBlockM, int kBlockN, bool IsCausal,
    bool IsLocal, bool PackGQA = false, bool Split = false,
    bool AllowReset = false>
struct SlidingChunkRangeAdapter {
  static_assert(!IsCausal && IsLocal);
  static_assert(!PackGQA && !Split);

  using ResetRange =
      flash::BlockMN<
          SeqlenInfo, kBlockM, kBlockN, IsCausal, IsLocal, PackGQA,
          Split>;

  CUTLASS_DEVICE
  static cute::tuple<int, int> fwd_n_block_range(
      SeqlenInfo const& seqlen_info, int m_block, int bidb,
      int split_idx, int num_splits, int window_size_left,
      int window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod,
      int chunk_position_offset = 0) {
    if constexpr (AllowReset) {
      if (attention_chunk_divmod.divisor == 0) {
        return ResetRange::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, num_splits,
            window_size_left, window_size_right, attention_chunk_divmod,
            qhead_per_khead_divmod, chunk_position_offset);
      }
    }
    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;
    int const qk_offset = seqlen_k - seqlen_q;
    int const q_tile_begin = m_block * kBlockM + qk_offset;
    int const q_tile_end = (m_block + 1) * kBlockM + qk_offset;
    int const chunk_left =
        flash::round_down(
            attention_chunk_divmod,
            q_tile_begin + chunk_position_offset) -
        chunk_position_offset - attention_chunk_divmod.divisor;
    int const n_block_min = std::max(0, chunk_left / kBlockN);
    int const n_block_max = std::min(
        cute::ceil_div(seqlen_k, kBlockN),
        std::max(0, cute::ceil_div(
            q_tile_end + window_size_right, kBlockN)));
    return {n_block_min, n_block_max};
  }

  CUTLASS_DEVICE
  static cute::tuple<int, int> new_k_block_range(
      SeqlenInfo const& seqlen_info, int m_block, int bidb,
      int split_idx, int num_splits, int window_size_left,
      int window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod,
      int chunk_position_offset = 0) {
    if constexpr (AllowReset) {
      if (attention_chunk_divmod.divisor == 0) {
        return ResetRange::get_n_block_k_new_min_max(
            seqlen_info, m_block, bidb, split_idx, num_splits,
            window_size_left, window_size_right, attention_chunk_divmod,
            qhead_per_khead_divmod, chunk_position_offset);
      }
    }
    auto [n_block_min, n_block_max] = fwd_n_block_range(
        seqlen_info, m_block, bidb, split_idx, num_splits,
        window_size_left, window_size_right, attention_chunk_divmod,
        qhead_per_khead_divmod, chunk_position_offset);
    int const idx_k_new_min =
        std::max(n_block_min * kBlockN - seqlen_info.seqlen_k_og, 0);
    int const idx_k_new_max = std::min(
        n_block_max * kBlockN - seqlen_info.seqlen_k_og,
        seqlen_info.seqlen_k_new);
    int const n_block_new_min = idx_k_new_min / kBlockN;
    int const n_block_new_max = idx_k_new_max > idx_k_new_min
        ? cute::ceil_div(idx_k_new_max, kBlockN)
        : n_block_new_min;
    return {n_block_new_min, n_block_new_max};
  }

  CUTLASS_DEVICE
  static cute::tuple<int, int> bwd_m_block_range(
      SeqlenInfo const& seqlen_info, int n_block, int bidb,
      int window_size_left, int window_size_right,
      int sink_token_length,
      cutlass::FastDivmod const& attention_chunk_divmod,
      int chunk_position_offset = 0) {
    if constexpr (AllowReset) {
      if (attention_chunk_divmod.divisor == 0) {
        return ResetRange::get_m_block_min_max(
            seqlen_info, n_block, bidb, window_size_left,
            window_size_right, sink_token_length,
            attention_chunk_divmod, chunk_position_offset);
      }
    }
    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;
    int const k_tile_begin = n_block * kBlockN;
    int const k_tile_end = (n_block + 1) * kBlockN;
    int const q_chunk_end =
        flash::round_up(
            attention_chunk_divmod,
            k_tile_end + chunk_position_offset) -
        chunk_position_offset + attention_chunk_divmod.divisor +
        seqlen_q - seqlen_k;
    int const m_block_min = std::max(
        0,
        (k_tile_begin + seqlen_q - seqlen_k - window_size_right) /
            kBlockM);
    int const m_block_max = std::min(
        cute::ceil_div(seqlen_q, kBlockM),
        cute::ceil_div(q_chunk_end, kBlockM));
    return {m_block_min, m_block_max};
  }

  CUTLASS_DEVICE
  static int right_boundary_n_block_min(
      SeqlenInfo const& seqlen_info, int m_block, int n_block_min,
      int window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod,
      int chunk_position_offset = 0) {
    if constexpr (AllowReset) {
      if (attention_chunk_divmod.divisor == 0) {
        return ResetRange::get_n_block_min_causal_local_mask(
            seqlen_info, m_block, n_block_min, window_size_right,
            attention_chunk_divmod, qhead_per_khead_divmod,
            chunk_position_offset);
      }
    }
    int const q_tile_begin =
        m_block * kBlockM + seqlen_info.seqlen_k -
        seqlen_info.seqlen_q;
    return std::max(
        n_block_min,
        (q_tile_begin + window_size_right) / kBlockN);
  }

  CUTLASS_DEVICE
  static int interior_n_block_min(
      SeqlenInfo const& seqlen_info, int m_block, int n_block_min,
      int window_size_left,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod,
      int chunk_position_offset = 0) {
    if constexpr (AllowReset) {
      if (attention_chunk_divmod.divisor == 0) {
        return ResetRange::get_n_block_min_before_local_mask(
            seqlen_info, m_block, n_block_min, window_size_left,
            attention_chunk_divmod, qhead_per_khead_divmod,
            chunk_position_offset);
      }
    }
    int const q_tile_last =
        std::min((m_block + 1) * kBlockM, seqlen_info.seqlen_q) - 1 +
        seqlen_info.seqlen_k - seqlen_info.seqlen_q;
    int const chunk_left =
        flash::round_down(
            attention_chunk_divmod,
            q_tile_last + chunk_position_offset) -
        chunk_position_offset - attention_chunk_divmod.divisor;
    return std::max(
        n_block_min, cute::ceil_div(chunk_left, kBlockN));
  }
};

template <
    class SeqlenInfo, int kBlockM, int kBlockN, bool IsCausal,
    bool IsLocal, bool PackGQA = false, bool Split = false>
struct StandardRangeAdapter {
  using Range =
      flash::BlockMN<
          SeqlenInfo, kBlockM, kBlockN, IsCausal, IsLocal, PackGQA,
          Split>;

  CUTLASS_DEVICE
  static cute::tuple<int, int> fwd_n_block_range(
      SeqlenInfo const& seqlen_info, int m_block, int bidb,
      int split_idx, int num_splits, int window_size_left,
      int window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod,
      int chunk_position_offset = 0) {
    return Range::get_n_block_min_max(
        seqlen_info, m_block, bidb, split_idx, num_splits,
        window_size_left, window_size_right, attention_chunk_divmod,
        qhead_per_khead_divmod, chunk_position_offset);
  }

  CUTLASS_DEVICE
  static cute::tuple<int, int> new_k_block_range(
      SeqlenInfo const& seqlen_info, int m_block, int bidb,
      int split_idx, int num_splits, int window_size_left,
      int window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod,
      int chunk_position_offset = 0) {
    return Range::get_n_block_k_new_min_max(
        seqlen_info, m_block, bidb, split_idx, num_splits,
        window_size_left, window_size_right, attention_chunk_divmod,
        qhead_per_khead_divmod, chunk_position_offset);
  }

  CUTLASS_DEVICE
  static cute::tuple<int, int> bwd_m_block_range(
      SeqlenInfo const& seqlen_info, int n_block, int bidb,
      int window_size_left, int window_size_right,
      int sink_token_length,
      cutlass::FastDivmod const& attention_chunk_divmod,
      int chunk_position_offset = 0) {
    return Range::get_m_block_min_max(
        seqlen_info, n_block, bidb, window_size_left,
        window_size_right, sink_token_length, attention_chunk_divmod,
        chunk_position_offset);
  }

  CUTLASS_DEVICE
  static int right_boundary_n_block_min(
      SeqlenInfo const& seqlen_info, int m_block, int n_block_min,
      int window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod,
      int chunk_position_offset = 0) {
    return Range::get_n_block_min_causal_local_mask(
        seqlen_info, m_block, n_block_min, window_size_right,
        attention_chunk_divmod, qhead_per_khead_divmod,
        chunk_position_offset);
  }

  CUTLASS_DEVICE
  static int interior_n_block_min(
      SeqlenInfo const& seqlen_info, int m_block, int n_block_min,
      int window_size_left,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod,
      int chunk_position_offset = 0) {
    return Range::get_n_block_min_before_local_mask(
        seqlen_info, m_block, n_block_min, window_size_left,
        attention_chunk_divmod, qhead_per_khead_divmod,
        chunk_position_offset);
  }
};

}  // namespace semantics
}  // namespace hopper
}  // namespace attention
}  // namespace ops
}  // namespace xattn

#pragma once

#include <torch/types.h>

#include <algorithm>
#include <cstdint>

namespace xattn {
namespace ops {

// Support of both SoftDelta reads: 0 causal, 1 sliding window (span = the
// inclusive left radius), 2 sliding chunk (span = the chunk size), as in the
// SoftDelta readers. The negative-query heads drop the aligned key in the kernels.
enum SoftDeltaVisibility : int64_t {
  kSoftDeltaCausal = 0,
  kSoftDeltaWindow = 1,
  kSoftDeltaChunk = 2,
};

// Chunk position of every query row when chunk positions restart in each segment
// (flash_sca's dense reset positions: matching Q / K runs aligned at their ends),
// int32 [B, Lq].
torch::Tensor SoftDeltaChunkPositions(const torch::Tensor& q_segment_idx,
                                           const torch::Tensor& k_segment_idx);

// Window / chunk fields of the forward or backward params (flash_swa's and
// flash_sca's dense settings, inclusive right boundary). q_chunk_positions is set
// for sliding chunks restarting in each segment.
template <typename Params>
void SetSoftDeltaVisibility(Params& params, int64_t visibility, int64_t span, int64_t k_length,
                                 const int* q_chunk_positions) {
  TORCH_CHECK(visibility >= kSoftDeltaCausal && visibility <= kSoftDeltaChunk,
              "softdelta visibility must be 0 (causal), 1 (sliding window) or 2 (sliding chunk)");
  TORCH_CHECK(visibility == kSoftDeltaCausal ||
                  (span >= (visibility == kSoftDeltaWindow ? 0 : 1) && span <= (int64_t(1) << 30)),
              "softdelta span is out of range");
  params.window_size_right = 0;
  if (visibility == kSoftDeltaCausal) {
    params.window_size_left = static_cast<int>(k_length - 1);
  } else if (visibility == kSoftDeltaWindow) {
    params.window_size_left = static_cast<int>(std::min<int64_t>(span, k_length - 1));
  } else {
    params.window_size_left = static_cast<int>(2 * span - 1);
    params.attention_chunk = q_chunk_positions != nullptr ? 0 : static_cast<int>(span);
    params.reset_attention_chunk = q_chunk_positions != nullptr ? static_cast<int>(span) : 0;
    params.q_chunk_positions = q_chunk_positions;
  }
}

}  // namespace ops
}  // namespace xattn

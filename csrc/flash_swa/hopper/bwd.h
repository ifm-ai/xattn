#pragma once

#include <cuda_runtime_api.h>

#include "attention/hopper/bwd_params.h"

namespace xattn {
namespace ops {

namespace flash_swa {

template <int kHeadDim, int kHeadDimV, int kDefaultBlockN>
struct FlashSWABwdSm90SegmentPolicy {
  static constexpr int kBlockM =
      kHeadDim <= 64 && kHeadDimV <= 128 ? 128 : 64;
  static constexpr int kBlockN =
      kHeadDim == 192 && kHeadDimV == 192 ? 96 : kDefaultBlockN;
};

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunFlashSWABwdSm90Variant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunFlashSWABwdSm90NonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunFlashSWABwdSm90VarlenVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunFlashSWABwdSm90VarlenNonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSWABwdSm90SegmentVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunCausalFlashAttnBwdSm90Variant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunCausalFlashAttnBwdSm90NonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream);

}  // namespace flash_swa
}  // namespace ops
}  // namespace xattn

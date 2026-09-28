#pragma once

#include <cuda_runtime_api.h>

#include "attention/hopper/bwd_params.h"

namespace xattn {
namespace ops {

template <int kHeadDim, int kHeadDimV, int kDefaultBlockN>
struct FlashSCABwdSm90SegmentPolicy {
  static constexpr int kDefaultBlockM =
      kHeadDim <= 64 && kHeadDimV <= 128 ? 128 : 64;
  static constexpr int kBlockM = kDefaultBlockM;
  static constexpr int kBlockN =
      kHeadDim == 192 && kHeadDimV == 192 ? 96 : kDefaultBlockN;
};

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunFlashSCABwdSm90Variant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM, int kStages, int kStagesDO, int kStagesDS,
          bool kPersistentScheduler>
void RunFlashSCABwdSm90DenseDet(
    AttentionBwdParams& params, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM, int kStages, int kStagesDO, int kStagesDS,
          bool kDeterministic, bool kDirectChunkRange>
void RunFlashSCABwdSm90DenseGQA(
    AttentionBwdParams& params, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunFlashSCABwdSm90NonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunFlashSCABwdSm90VarlenVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void RunFlashSCABwdSm90VarlenNonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSCABwdSm90SegmentVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream);

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSCABwdSm90SegmentDetVariant(
    AttentionBwdParams& params, cudaStream_t stream);

}  // namespace ops
}  // namespace xattn

#pragma once

#include "flash_sca/hopper/bwd.h"

#include "attention/hopper/bwd_template.cuh"

namespace xattn {
namespace ops {

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM, int kStages, int kStagesDO, int kStagesDS,
          bool kPersistentScheduler>
void RunFlashSCABwdSm90DenseDet(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunAttentionBwdSm90Kernel<
      kHeadDim, kHeadDimV, kBlockN, Element, true, false, false, kBlockM,
      false, false, attention::semantics::SlidingChunkVisibility,
      flash::AttentionBwdSm90, kStages, kStagesDO, kStagesDS,
      kPersistentScheduler>(params, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM, int kStages, int kStagesDO, int kStagesDS,
          bool kDeterministic, bool kDirectChunkRange>
void RunFlashSCABwdSm90DenseGQA(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunAttentionBwdSm90Kernel<
      kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic, false, false,
      kBlockM,
      true, false, attention::semantics::SlidingChunkVisibility,
      flash::AttentionBwdSm90, kStages, kStagesDO, kStagesDS,
      false, kDirectChunkRange, true>(params, stream);
}

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kDeterministic, bool kVarlen, bool kHasSegment,
          int kBlockMOverride = 0>
void RunFlashSCABwdSm90HeadDispatch(
    AttentionBwdParams& params, cudaStream_t stream) {
  TORCH_CHECK(params.h % params.h_k == 0,
              "FlashSCA SM90 BWD requires Hq divisible by Hkv");
  if constexpr (!kVarlen && !kHasSegment) {
    if (params.odd_head_window_right_delta != 0) {
      if (params.h / 2 == params.h_k) {
        RunAttentionBwdSm90Kernel<
            kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic,
            false, false, kBlockMOverride, false /*kGQA*/, false,
            attention::semantics::SlidingChunkVisibility,
            flash::AttentionBwdSm90, 0, 0, 0, false, true, true,
            true /*kReaderPairKVReuse*/>(params, stream);
      } else {
        RunAttentionBwdSm90Kernel<
            kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic,
            false, false, kBlockMOverride, true /*kGQA*/, false,
            attention::semantics::SlidingChunkVisibility,
            flash::AttentionBwdSm90, 0, 0, 0, false, true, true,
            true /*kReaderPairKVReuse*/>(params, stream);
      }
      return;
    }
  }
  if (params.h == params.h_k) {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic, kVarlen,
        kHasSegment, kBlockMOverride, false /*kGQA*/>(params, stream);
  } else {
    if (params.seqlen_k > params.seqlen_q) {
      RunAttentionBwdSm90Kernel<
          kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic, kVarlen,
          kHasSegment, kBlockMOverride, true /*kGQA*/,
          true /*kTwoComponentDV*/>(params, stream);
    } else {
      RunAttentionBwdSm90Kernel<
          kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic, kVarlen,
          kHasSegment, kBlockMOverride, true /*kGQA*/>(params, stream);
    }
  }
}

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kVarlen, bool kHasSegment, int kBlockMOverride = 0>
void RunFlashSCABwdSm90Deterministic(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  if (deterministic) {
    RunFlashSCABwdSm90HeadDispatch<
        kHeadDim, kHeadDimV, kBlockN, Element, true, kVarlen, kHasSegment,
        kBlockMOverride>(params, stream);
  } else {
    RunFlashSCABwdSm90HeadDispatch<
        kHeadDim, kHeadDimV, kBlockN, Element, false, kVarlen, kHasSegment,
        kBlockMOverride>(params, stream);
  }
}

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kVarlen, int kBlockMOverride = 0>
void RunFlashSCABwdSm90Segment(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  if (params.q_segment_idx != nullptr) {
    RunFlashSCABwdSm90Deterministic<kHeadDim, kHeadDimV, kBlockN, Element,
                                    kVarlen, true, kBlockMOverride>(
        params, deterministic, stream);
  } else {
    RunFlashSCABwdSm90Deterministic<kHeadDim, kHeadDimV, kBlockN, Element,
                                    kVarlen, false, kBlockMOverride>(
        params, deterministic, stream);
  }
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSCABwdSm90Variant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  RunFlashSCABwdSm90Segment<kHeadDim, kHeadDimV, kBlockN, Element,
                            false /*kVarlen*/, kBlockM>(
      params, deterministic, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSCABwdSm90VarlenVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  RunFlashSCABwdSm90Deterministic<kHeadDim, kHeadDimV, kBlockN, Element,
                                  true /*kVarlen*/, false /*kHasSegment*/,
                                  kBlockM>(
      params, deterministic, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSCABwdSm90SegmentVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  RunFlashSCABwdSm90Deterministic<kHeadDim, kHeadDimV, kBlockN, Element,
                                  false /*kVarlen*/, true /*kHasSegment*/,
                                  kBlockM>(
      params, deterministic, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSCABwdSm90SegmentDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSCABwdSm90HeadDispatch<
      kHeadDim, kHeadDimV, kBlockN, Element, true /*kDeterministic*/,
      false /*kVarlen*/, true /*kHasSegment*/, kBlockM>(params, stream);
}

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kVarlen, bool kHasSegment, int kBlockMOverride = 0>
void RunFlashSCABwdSm90NonDetSegment(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSCABwdSm90HeadDispatch<
      kHeadDim, kHeadDimV, kBlockN, Element, false, kVarlen, kHasSegment,
      kBlockMOverride>(params, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSCABwdSm90NonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  if (params.q_segment_idx != nullptr) {
    RunFlashSCABwdSm90NonDetSegment<kHeadDim, kHeadDimV, kBlockN, Element,
                                    false /*kVarlen*/, true /*kHasSegment*/,
                                    kBlockM>(
        params, stream);
  } else {
    RunFlashSCABwdSm90NonDetSegment<kHeadDim, kHeadDimV, kBlockN, Element,
                                    false /*kVarlen*/, false /*kHasSegment*/,
                                    kBlockM>(
        params, stream);
  }
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSCABwdSm90VarlenNonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSCABwdSm90NonDetSegment<kHeadDim, kHeadDimV, kBlockN, Element,
                                  true /*kVarlen*/, false /*kHasSegment*/,
                                  kBlockM>(
      params, stream);
}

}  // namespace ops
}  // namespace xattn

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_CHUNK_GLOBAL(               \
    Element, HeadDim, HeadDimV, BlockN)                                 \
  namespace xattn {                                                      \
  namespace ops {                                                       \
  template void RunFlashSCABwdSm90Variant<Element, HeadDim, HeadDimV,   \
                                          BlockN, 0>(                   \
      AttentionBwdParams&, bool, cudaStream_t);                          \
  template void RunFlashSCABwdSm90SegmentVariant<                       \
      Element, HeadDim, HeadDimV,                                       \
      FlashSCABwdSm90SegmentPolicy<HeadDim, HeadDimV, BlockN>::kBlockN, \
      FlashSCABwdSm90SegmentPolicy<HeadDim, HeadDimV, BlockN>::kBlockM>(\
      AttentionBwdParams&, bool, cudaStream_t);                          \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_CHUNK_RESET(                \
    Element, HeadDim, HeadDimV, BlockN)                                 \
  namespace xattn {                                                      \
  namespace ops {                                                       \
  template void RunFlashSCABwdSm90VarlenVariant<Element, HeadDim,       \
                                                HeadDimV, BlockN, 0>(   \
      AttentionBwdParams&, bool, cudaStream_t);                          \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE(Element, HeadDim, HeadDimV, \
                                            BlockN)                     \
  XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_CHUNK_GLOBAL(                     \
      Element, HeadDim, HeadDimV, BlockN)                               \
  XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_CHUNK_RESET(                      \
      Element, HeadDim, HeadDimV, BlockN)

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_DET(                 \
    Element, HeadDim, HeadDimV, BlockN, BlockM, Stages, StagesDO,       \
    StagesDS, Persistent)                                               \
  namespace xattn {                                                     \
  namespace ops {                                                      \
  template void RunFlashSCABwdSm90DenseDet<                            \
      Element, HeadDim, HeadDimV, BlockN, BlockM, Stages, StagesDO,    \
      StagesDS, Persistent>(                                            \
      AttentionBwdParams&, cudaStream_t);                               \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_DENSE_GQA(                 \
    Element, HeadDim, HeadDimV, BlockN, BlockM, Stages, StagesDO,       \
    StagesDS, Deterministic, DirectChunkRange)                          \
  namespace xattn {                                                     \
  namespace ops {                                                       \
  template void RunFlashSCABwdSm90DenseGQA<                             \
      Element, HeadDim, HeadDimV, BlockN, BlockM, Stages, StagesDO,     \
      StagesDS, Deterministic, DirectChunkRange>(                       \
      AttentionBwdParams&, cudaStream_t);                                \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_SEGMENT(                    \
    Element, HeadDim, HeadDimV, BlockN, BlockM)                         \
  namespace xattn {                                                      \
  namespace ops {                                                       \
  template void RunFlashSCABwdSm90SegmentVariant<                       \
      Element, HeadDim, HeadDimV, BlockN, BlockM>(                      \
      AttentionBwdParams&, bool, cudaStream_t);                          \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_SEGMENT_DET(                \
    Element, HeadDim, HeadDimV, BlockN, BlockM)                         \
  namespace xattn {                                                      \
  namespace ops {                                                       \
  template void RunFlashSCABwdSm90SegmentDetVariant<                    \
      Element, HeadDim, HeadDimV, BlockN, BlockM>(                      \
      AttentionBwdParams&, cudaStream_t);                                \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_NONDET_CHUNK_GLOBAL(        \
    Element, HeadDim, HeadDimV, BlockN)                                 \
  namespace xattn {                                                      \
  namespace ops {                                                       \
  template void RunFlashSCABwdSm90NonDetVariant<Element, HeadDim,       \
                                                HeadDimV, BlockN, 0>(   \
      AttentionBwdParams&, cudaStream_t);                                \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_NONDET_CHUNK_RESET(         \
    Element, HeadDim, HeadDimV, BlockN)                                 \
  namespace xattn {                                                      \
  namespace ops {                                                       \
  template void RunFlashSCABwdSm90VarlenNonDetVariant<                  \
      Element, HeadDim, HeadDimV, BlockN, 0>(                           \
      AttentionBwdParams&, cudaStream_t);                                \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_NONDET(Element, HeadDim,    \
                                                   HeadDimV, BlockN)    \
  XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_NONDET_CHUNK_GLOBAL(              \
      Element, HeadDim, HeadDimV, BlockN)                               \
  XATTN_FLASH_SCA_BWD_SM90_INSTANTIATE_NONDET_CHUNK_RESET(               \
      Element, HeadDim, HeadDimV, BlockN)

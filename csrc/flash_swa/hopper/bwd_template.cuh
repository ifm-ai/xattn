#pragma once

#include "flash_swa/hopper/bwd.h"

#include "attention/hopper/bwd_template.cuh"
#include "flash_swa/hopper/bwd_kernel_sm90.h"

namespace xattn {
namespace ops {
namespace flash_swa {

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kDeterministic, bool kVarlen, bool kHasSegment,
          int kBlockM = 0>
void RunFlashSWABwdSm90HeadDispatch(
    AttentionBwdParams& params, cudaStream_t stream) {
  TORCH_CHECK(params.h % params.h_k == 0,
              "FlashSWA SM90 BWD requires Hq divisible by Hkv");
  if (params.h == params.h_k) {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic, kVarlen,
        kHasSegment, kBlockM, false, false,
        attention::semantics::CausalSlidingWindowVisibility,
        flash::FlashSWABwdSm90>(
        params, stream);
  } else if (params.seqlen_k > params.seqlen_q) {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic, kVarlen,
        kHasSegment, kBlockM, true, true,
        attention::semantics::CausalSlidingWindowVisibility,
        flash::FlashSWABwdSm90>(
        params, stream);
  } else {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic, kVarlen,
        kHasSegment, kBlockM, true, false,
        attention::semantics::CausalSlidingWindowVisibility,
        flash::FlashSWABwdSm90>(
        params, stream);
  }
}

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kVarlen, bool kHasSegment, int kBlockM = 0>
void RunFlashSWABwdSm90Deterministic(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  if (deterministic) {
    RunFlashSWABwdSm90HeadDispatch<
        kHeadDim, kHeadDimV, kBlockN, Element, true, kVarlen,
        kHasSegment, kBlockM>(params, stream);
  } else {
    RunFlashSWABwdSm90HeadDispatch<
        kHeadDim, kHeadDimV, kBlockN, Element, false, kVarlen,
        kHasSegment, kBlockM>(params, stream);
  }
}

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kVarlen, int kBlockM = 0>
void RunFlashSWABwdSm90Segment(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  if (params.q_segment_idx != nullptr) {
    RunFlashSWABwdSm90Deterministic<
        kHeadDim, kHeadDimV, kBlockN, Element, kVarlen, true, kBlockM>(
        params, deterministic, stream);
  } else {
    RunFlashSWABwdSm90Deterministic<
        kHeadDim, kHeadDimV, kBlockN, Element, kVarlen, false, kBlockM>(
        params, deterministic, stream);
  }
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSWABwdSm90Variant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  RunFlashSWABwdSm90Segment<
      kHeadDim, kHeadDimV, kBlockN, Element, false, kBlockM>(
      params, deterministic, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSWABwdSm90VarlenVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  RunFlashSWABwdSm90Deterministic<
      kHeadDim, kHeadDimV, kBlockN, Element, true, false, kBlockM>(
      params, deterministic, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSWABwdSm90SegmentVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  RunFlashSWABwdSm90Deterministic<
      kHeadDim, kHeadDimV, kBlockN, Element, false, true, kBlockM>(
      params, deterministic, stream);
}

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kVarlen, bool kHasSegment, int kBlockM = 0>
void RunFlashSWABwdSm90NonDet(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSWABwdSm90HeadDispatch<
      kHeadDim, kHeadDimV, kBlockN, Element, false, kVarlen,
      kHasSegment, kBlockM>(params, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSWABwdSm90NonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  if (params.q_segment_idx != nullptr) {
    RunFlashSWABwdSm90NonDet<
        kHeadDim, kHeadDimV, kBlockN, Element, false, true, kBlockM>(
        params, stream);
  } else {
    RunFlashSWABwdSm90NonDet<
        kHeadDim, kHeadDimV, kBlockN, Element, false, false, kBlockM>(
        params, stream);
  }
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunFlashSWABwdSm90VarlenNonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSWABwdSm90NonDet<
      kHeadDim, kHeadDimV, kBlockN, Element, true, false, kBlockM>(
      params, stream);
}

template <int kHeadDim, int kHeadDimV, int kBlockN, typename Element,
          bool kDeterministic, int kBlockM = 0>
void RunCausalFlashAttnBwdSm90HeadDispatch(
    AttentionBwdParams& params, cudaStream_t stream) {
  TORCH_CHECK(params.h % params.h_k == 0,
              "FlashSWA SM90 full BWD requires Hq divisible by Hkv");
  static constexpr int kStages =
      kDeterministic && kHeadDim == 192 && kHeadDimV == 192 ? 2 : 0;
  if (params.h == params.h_k) {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic,
        false, false, kBlockM, false, false,
        attention::semantics::CausalFullVisibility,
        flash::FlashSWABwdSm90, kStages>(params, stream);
  } else if (params.seqlen_k > params.seqlen_q) {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic,
        false, false, kBlockM, true, true,
        attention::semantics::CausalFullVisibility,
        flash::FlashSWABwdSm90, kStages>(params, stream);
  } else {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, kDeterministic,
        false, false, kBlockM, true, false,
        attention::semantics::CausalFullVisibility,
        flash::FlashSWABwdSm90, kStages>(params, stream);
  }
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunCausalFlashAttnBwdSm90Variant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  if (deterministic) {
    RunCausalFlashAttnBwdSm90HeadDispatch<
        kHeadDim, kHeadDimV, kBlockN, Element, true, kBlockM>(
        params, stream);
  } else {
    RunCausalFlashAttnBwdSm90HeadDispatch<
        kHeadDim, kHeadDimV, kBlockN, Element, false, kBlockM>(
        params, stream);
  }
}

// Packed varlen rows with causal-full visibility (no window mask), equal Q / K
// lengths, nondeterministic: the dispatcher picks it when the window covers
// every sequence.
template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunCausalFlashAttnBwdSm90VarlenNonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  TORCH_CHECK(params.h % params.h_k == 0,
              "FlashSWA SM90 full varlen BWD requires Hq divisible by Hkv");
  if (params.h == params.h_k) {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, false, true, false, kBlockM,
        false, false, attention::semantics::CausalFullVisibility,
        flash::FlashSWABwdSm90>(params, stream);
  } else {
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, false, true, false, kBlockM,
        true, false, attention::semantics::CausalFullVisibility,
        flash::FlashSWABwdSm90>(params, stream);
  }
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM>
void RunCausalFlashAttnBwdSm90NonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunCausalFlashAttnBwdSm90HeadDispatch<
      kHeadDim, kHeadDimV, kBlockN, Element, false, kBlockM>(
      params, stream);
}

}  // namespace flash_swa
}  // namespace ops
}  // namespace xattn

#define XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_DENSE_SEGMENT(             \
    Element, HeadDim, HeadDimV, BlockN)                                \
  namespace xattn {                                                     \
  namespace ops {                                                       \
  namespace flash_swa {                                                 \
  template void RunFlashSWABwdSm90Variant<                             \
      Element, HeadDim, HeadDimV, BlockN, 0>(                          \
      AttentionBwdParams&, bool, cudaStream_t);                         \
  template void RunFlashSWABwdSm90SegmentVariant<                      \
      Element, HeadDim, HeadDimV,                                      \
      FlashSWABwdSm90SegmentPolicy<                                    \
          HeadDim, HeadDimV, BlockN>::kBlockN,                         \
      FlashSWABwdSm90SegmentPolicy<                                    \
          HeadDim, HeadDimV, BlockN>::kBlockM>(                        \
      AttentionBwdParams&, bool, cudaStream_t);                         \
  template void RunCausalFlashAttnBwdSm90Variant<                        \
      Element, HeadDim, HeadDimV, BlockN, 0>(                         \
      AttentionBwdParams&, bool, cudaStream_t);                        \
  }                                                                    \
  }                                                                    \
  }

#define XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_VARLEN(                   \
    Element, HeadDim, HeadDimV, BlockN)                               \
  namespace xattn {                                                    \
  namespace ops {                                                      \
  namespace flash_swa {                                                \
  template void RunFlashSWABwdSm90VarlenVariant<                      \
      Element, HeadDim, HeadDimV, BlockN, 0>(                         \
      AttentionBwdParams&, bool, cudaStream_t);                        \
  template void RunCausalFlashAttnBwdSm90VarlenNonDetVariant<          \
      Element, HeadDim, HeadDimV, BlockN, 0>(                         \
      AttentionBwdParams&, cudaStream_t);                              \
  }                                                                    \
  }                                                                    \
  }

#define XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE(                          \
    Element, HeadDim, HeadDimV, BlockN)                               \
  XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_DENSE_SEGMENT(                  \
      Element, HeadDim, HeadDimV, BlockN)                             \
  XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_VARLEN(                         \
      Element, HeadDim, HeadDimV, BlockN)

#define XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_NONDET_DENSE_SEGMENT(     \
    Element, HeadDim, HeadDimV, BlockN)                               \
  namespace xattn {                                                    \
  namespace ops {                                                      \
  namespace flash_swa {                                                \
  template void RunFlashSWABwdSm90NonDetVariant<                      \
      Element, HeadDim, HeadDimV, BlockN, 0>(                         \
      AttentionBwdParams&, cudaStream_t);                              \
  template void RunCausalFlashAttnBwdSm90NonDetVariant<                  \
      Element, HeadDim, HeadDimV, BlockN, 0>(                         \
      AttentionBwdParams&, cudaStream_t);                              \
  }                                                                    \
  }                                                                    \
  }

#define XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_NONDET_VARLEN(            \
    Element, HeadDim, HeadDimV, BlockN)                               \
  namespace xattn {                                                    \
  namespace ops {                                                      \
  namespace flash_swa {                                                \
  template void RunFlashSWABwdSm90VarlenNonDetVariant<                \
      Element, HeadDim, HeadDimV, BlockN, 0>(                         \
      AttentionBwdParams&, cudaStream_t);                              \
  template void RunCausalFlashAttnBwdSm90VarlenNonDetVariant<          \
      Element, HeadDim, HeadDimV, BlockN, 0>(                         \
      AttentionBwdParams&, cudaStream_t);                              \
  }                                                                    \
  }                                                                    \
  }

#define XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_NONDET(                   \
    Element, HeadDim, HeadDimV, BlockN)                               \
  XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_NONDET_DENSE_SEGMENT(           \
      Element, HeadDim, HeadDimV, BlockN)                             \
  XATTN_FLASH_SWA_BWD_SM90_INSTANTIATE_NONDET_VARLEN(                  \
      Element, HeadDim, HeadDimV, BlockN)

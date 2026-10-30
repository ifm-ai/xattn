#pragma once

#include "softdelta/bwd_launch.h"
#include "softdelta/buckets.h"

#include <type_traits>

#include "attention/hopper/bwd_template.cuh"
#include "attention/semantics/visibility.h"
#include "softdelta/bwd_kernel_sm90.h"

namespace xattn {
namespace ops {

template <>
struct AttentionBwdKernelUsesNegQReaders<flash::SoftDeltaBwdSm90>
    : std::true_type {};

// The causal_flash_attn non-deterministic GQA backward configuration per
// (D, V) bucket (its BN per bucket, default BM / stages; causal-full visibility
// for dense rows, the full-span window with segments) over logical heads:
// dK / dV of every reader accumulate in the grouped FP32 buffers. With
// kReaderGroups one work tile covers a key tile and KV head over all of its
// readers and writes dK / dV once (dK in FP32 when the negative-query gradient
// folds into it). Dense sliding windows and sliding chunks take flash_swa's /
// flash_sca's BN per bucket (chunks also flash_sca's direct chunk ranges and
// tiles); windows with segments run on the causal segment instance.
template <int kHeadDim, int kHeadDimV, typename Visibility, bool kHasSegment>
constexpr int SoftDeltaBwdBlockN() {
  using attention::semantics::VisibilityKind;
  if constexpr (Visibility::kKind == VisibilityKind::kSlidingChunk) {
    return kHeadDim == 256 || (kHeadDim == 160 && (kHeadDimV == 160 || kHeadDimV == 256)) ||
                   (kHeadDim == 192 && kHeadDimV == 256)
        ? 64
        : (kHeadDim == 192 && kHeadDimV == 192) ? 96
        : 128;
  } else if constexpr (Visibility::kKind == VisibilityKind::kCausalSlidingWindow && !kHasSegment) {
    return (kHeadDim == 160 && kHeadDimV == 256) || (kHeadDim == 192 && (kHeadDimV == 160 || kHeadDimV == 256)) ||
                   (kHeadDim == 256 && kHeadDimV == 256)
        ? 64
        : (kHeadDim == 192 && kHeadDimV == 192) ? 96
        : 128;
  } else {
    return kHeadDim == 256 ? (kHeadDimV == 256 ? 80 : 128)
        : (kHeadDim == 192 && kHeadDimV == 192) ? 96
        : (kHeadDim == 192 && kHeadDimV == 256) ? 64
        : (kHeadDim == 160 && (kHeadDimV == 160 || kHeadDimV == 256)) ? 64
        : 128;
  }
}

template <typename Element, int kHeadDim, int kHeadDimV, typename Visibility, bool kHasSegment,
          bool kReaderGroups>
void RunSoftDeltaBwdSm90(AttentionBwdParams& params, cudaStream_t stream) {
  if constexpr (!kHasSegment && Visibility::kKind == attention::semantics::VisibilityKind::kCausalFull &&
                SoftDeltaBwdDenseSquareDim(kHeadDim, kHeadDimV) != 0) {
    // Dense rows of this pair run on the square instance (bwd.cu BwdBucket).
    TORCH_CHECK(false, "softdelta fused BWD: dense D=", kHeadDim, ", V=", kHeadDimV,
                " runs on its square instance");
  } else {
    static constexpr int kBlockN = SoftDeltaBwdBlockN<kHeadDim, kHeadDimV, Visibility, kHasSegment>();
    static constexpr bool kDirectChunk = Visibility::kIsSlidingChunk;
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, false /*kDeterministic*/,
        false /*kVarlen*/, kHasSegment, 0 /*kBlockMOverride*/, !kReaderGroups /*kGQA*/,
        false /*kTwoComponentDV*/, Visibility, flash::SoftDeltaBwdSm90,
        0, 0, 0, false /*kPersistentScheduler*/, kDirectChunk /*kDirectChunkRange*/,
        kDirectChunk /*kDirectChunkTile*/, kReaderGroups /*kReaderPairKVReuse*/>(
        params, stream);
  }
}

}  // namespace ops
}  // namespace xattn

#define XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV, Visibility, \
                                                              HasSegment, Groups)                    \
  namespace xattn {                                                                                  \
  namespace ops {                                                                                    \
  template void RunSoftDeltaBwdSm90<Element, HeadDim, HeadDimV,                                 \
                                         attention::semantics::Visibility, HasSegment, Groups>(      \
      AttentionBwdParams&, cudaStream_t);                                                            \
  }                                                                                                  \
  }
// Causal: dense rows and segments (the segment instance also runs sliding windows with segments).
#define XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, Groups)       \
  XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                 \
                                                        CausalFullVisibility, false, Groups)        \
  XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                 \
                                                        CausalSlidingWindowVisibility, true, Groups)
// Sliding windows on dense rows, sliding chunks on dense rows and with segments.
#define XATTN_SOFTDELTA_BWD_LOCAL_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, Groups) \
  XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                 \
                                                        CausalSlidingWindowVisibility, false,       \
                                                        Groups)                                     \
  XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                 \
                                                        SlidingChunkVisibility, false, Groups)      \
  XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                 \
                                                        SlidingChunkVisibility, true, Groups)
#define XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE(Element, HeadDim, HeadDimV) \
  XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, false)
#define XATTN_SOFTDELTA_BWD_GROUPS_SM90_INSTANTIATE(Element, HeadDim, HeadDimV) \
  XATTN_SOFTDELTA_BWD_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, true)
#define XATTN_SOFTDELTA_BWD_LOCAL_SM90_INSTANTIATE(Element, HeadDim, HeadDimV) \
  XATTN_SOFTDELTA_BWD_LOCAL_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, false)
#define XATTN_SOFTDELTA_BWD_GROUPS_LOCAL_SM90_INSTANTIATE(Element, HeadDim, HeadDimV) \
  XATTN_SOFTDELTA_BWD_LOCAL_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, true)

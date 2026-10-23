#pragma once

#include "group_softdelta/bwd_launch.h"
#include "group_softdelta/buckets.h"

#include <type_traits>

#include "attention/hopper/bwd_template.cuh"
#include "attention/semantics/visibility.h"
#include "group_softdelta/bwd_kernel_sm90.h"

namespace xattn {
namespace ops {

template <>
struct AttentionBwdKernelUsesNegQReaders<flash::GroupSoftDeltaBwdSm90>
    : std::true_type {};

// The causal_flash_attn non-deterministic GQA backward configuration per
// (D, V) bucket (its BN per bucket, default BM / stages; causal-full visibility
// for dense rows, the full-span window with segments) over logical heads:
// dK / dV of every reader accumulate in the grouped FP32 buffers. With
// kReaderGroups one work tile covers a key tile and KV head over all of its
// readers and writes dK / dV once (dK in FP32 when the negative-query gradient
// folds into it).
template <typename Element, int kHeadDim, int kHeadDimV, bool kHasSegment, bool kReaderGroups>
void RunGroupSoftDeltaBwdSm90(AttentionBwdParams& params, cudaStream_t stream) {
  if constexpr (!kHasSegment && GroupSoftDeltaBwdDenseSquareDim(kHeadDim, kHeadDimV) != 0) {
    // Dense rows of this pair run on the square instance (bwd.cu BwdBucket).
    TORCH_CHECK(false, "group_softdelta fused BWD: dense D=", kHeadDim, ", V=", kHeadDimV,
                " runs on its square instance");
  } else {
    using Visibility = std::conditional_t<
        kHasSegment, attention::semantics::CausalSlidingWindowVisibility,
        attention::semantics::CausalFullVisibility>;
    static constexpr int kBlockN =
        kHeadDim == 256 ? (kHeadDimV == 256 ? 80 : 128)
        : (kHeadDim == 192 && kHeadDimV == 192) ? 96
        : (kHeadDim == 192 && kHeadDimV == 256) ? 64
        : (kHeadDim == 160 && (kHeadDimV == 160 || kHeadDimV == 256)) ? 64
        : 128;
    RunAttentionBwdSm90Kernel<
        kHeadDim, kHeadDimV, kBlockN, Element, false /*kDeterministic*/,
        false /*kVarlen*/, kHasSegment, 0 /*kBlockMOverride*/, !kReaderGroups /*kGQA*/,
        false /*kTwoComponentDV*/, Visibility, flash::GroupSoftDeltaBwdSm90,
        0, 0, 0, false /*kPersistentScheduler*/, false /*kDirectChunkRange*/,
        false /*kDirectChunkTile*/, kReaderGroups /*kReaderPairKVReuse*/>(
        params, stream);
  }
}

}  // namespace ops
}  // namespace xattn

#define XATTN_GROUP_SOFTDELTA_BWD_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, Groups) \
  namespace xattn {                                                                           \
  namespace ops {                                                                             \
  template void RunGroupSoftDeltaBwdSm90<Element, HeadDim, HeadDimV, false, Groups>(          \
      AttentionBwdParams&, cudaStream_t);                                                     \
  template void RunGroupSoftDeltaBwdSm90<Element, HeadDim, HeadDimV, true, Groups>(           \
      AttentionBwdParams&, cudaStream_t);                                                     \
  }                                                                                           \
  }
#define XATTN_GROUP_SOFTDELTA_BWD_SM90_INSTANTIATE(Element, HeadDim, HeadDimV) \
  XATTN_GROUP_SOFTDELTA_BWD_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, false)
#define XATTN_GROUP_SOFTDELTA_BWD_GROUPS_SM90_INSTANTIATE(Element, HeadDim, HeadDimV) \
  XATTN_GROUP_SOFTDELTA_BWD_SM90_INSTANTIATE_GROUPS(Element, HeadDim, HeadDimV, true)

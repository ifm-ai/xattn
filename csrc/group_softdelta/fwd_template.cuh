#pragma once

#include "group_softdelta/fwd_launch.h"

#include <tuple>
#include <type_traits>

#include "attention/hopper/fwd_template.cuh"
#include "attention/semantics/visibility.h"
#include "group_softdelta/fwd_epilogue_sm90.h"
#include "group_softdelta/fwd_kernel_sm90.h"

namespace xattn {
namespace ops {

template <>
struct AttentionFwdKernelUsesNegQReaders<flash::GroupSoftDeltaFwdSm90>
    : std::true_type {};

struct GroupSoftDeltaEpilogueArguments {
  template <typename CollectiveEpilogue>
  static typename CollectiveEpilogue::Arguments Make(AttentionFwdParams& params) {
    return {AttentionFwdEpilogueArguments::template Make<typename CollectiveEpilogue::BaseEpilogue>(params),
            params.h_k};
  }
};

// The causal_flash_attn forward configuration per (D, V) bucket
// (RunAttentionFwdSm90Kernel: AttentionTileSizeFwdSm90 tiles, its dense
// full-span tiles and the low-precision NoState epilogue for D = V in {64, 128};
// causal-full visibility for dense rows, the full-span window with segments)
// over logical heads, with the class-major readers epilogue. Sliding windows and
// chunks take flash_swa's / flash_sca's tiles (RunAttentionFwdSm90Kernel with
// their visibility, including the dense 128-row window tiles); dense chunks at
// D = V in {64, 128} also the NoState epilogue.
template <typename Element, int kHeadDim, int kHeadDimV, typename Visibility, bool kHasSegment>
void RunGroupSoftDeltaFwdSm90(AttentionFwdParams& params, cudaStream_t stream) {
  using attention::semantics::VisibilityKind;
  static constexpr bool kUseFastExp2 = true;  // ex2.approx.ftz with and without segments
  static constexpr std::tuple<int, int, bool, bool> kTile =
      AttentionTileSizeFwdSm90<Visibility, false, kHasSegment>(kHeadDim, kHeadDimV);
  static constexpr int kBlockM = std::get<0>(kTile), kBlockN = std::get<1>(kTile);
  static constexpr bool kMmaPVIsRS = std::get<2>(kTile), kIntraWGOverlap = std::get<3>(kTile);
  if constexpr (!kHasSegment && Visibility::kKind == VisibilityKind::kCausalFull) {
    static constexpr bool kFullTile =
        (kHeadDim == 32 && kHeadDimV == 64) || (kHeadDim == 64 && kHeadDimV == 32) ||
        (kHeadDim == 192 && kHeadDimV == 128) || (kHeadDim == 256 && kHeadDimV == 32);
    if constexpr (kFullTile) {
      static constexpr bool kLongFull = kHeadDim <= 64;
      if (params.d == kHeadDim && params.dv == kHeadDimV && params.seqlen_q == params.seqlen_k &&
          (!kLongFull || params.seqlen_q >= 16384)) {
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, Element, false, false, true,
            kLongFull ? 192 : 128, 128, kMmaPVIsRS, kIntraWGOverlap, kUseFastExp2, Visibility,
            flash::GroupSoftDeltaFwdSm90, 1, false, false, void,
            GroupSoftDeltaEpilogueFwd, GroupSoftDeltaEpilogueArguments>(params, stream);
        return;
      }
    }
    if constexpr (kHeadDim == kHeadDimV && (kHeadDim == 64 || kHeadDim == 128)) {
      if (params.o_state_ptr == nullptr && params.use_fast_reciprocal) {
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, Element, false, false, true,
            kBlockM, kBlockN, kMmaPVIsRS, kIntraWGOverlap, kUseFastExp2, Visibility,
            flash::GroupSoftDeltaFwdSm90, 1, false, false, void,
            GroupSoftDeltaEpilogueFwdNoState, GroupSoftDeltaEpilogueArguments>(params, stream);
        return;
      }
    }
  }
  if constexpr (!kHasSegment && Visibility::kKind == VisibilityKind::kSlidingChunk && kHeadDim == kHeadDimV &&
                (kHeadDim == 64 || kHeadDim == 128)) {
    // The causal NoState epilogue for dense chunks (dense windows keep the state epilogue).
    if (params.o_state_ptr == nullptr && params.use_fast_reciprocal) {
      RunAttentionFwdSm90KernelTile<
          kHeadDim, kHeadDimV, Element, Element, false, false, true,
          kBlockM, kBlockN, kMmaPVIsRS, kIntraWGOverlap, kUseFastExp2, Visibility,
          flash::GroupSoftDeltaFwdSm90, 1, false, false, void,
          GroupSoftDeltaEpilogueFwdNoState, GroupSoftDeltaEpilogueArguments>(params, stream);
      return;
    }
  }
  if constexpr (!kHasSegment && Visibility::kKind == VisibilityKind::kCausalSlidingWindow) {
    // RunAttentionFwdSm90Kernel's dense window tiles: 128 rows for windows up to 511.
    static constexpr bool kWindowTile =
        (kHeadDim == 32 && kHeadDimV <= 128) || (kHeadDim == 64 && kHeadDimV <= 96) ||
        (kHeadDim == 96 && kHeadDimV == 96);
    if constexpr (kWindowTile) {
      static constexpr bool kWideWindow =
          (kHeadDim == 32 && kHeadDimV >= 96) || (kHeadDim == 64 && kHeadDimV == 96);
      if (params.d == kHeadDim && params.dv == kHeadDimV && params.seqlen_q == params.seqlen_k &&
          (kWideWindow || (params.window_size_left >= 0 && params.window_size_left <= 511))) {
        RunAttentionFwdSm90KernelTile<
            kHeadDim, kHeadDimV, Element, Element, false, false, true,
            128, 128, kMmaPVIsRS, kIntraWGOverlap, kUseFastExp2, Visibility,
            flash::GroupSoftDeltaFwdSm90, 1, false, false, void,
            GroupSoftDeltaEpilogueFwd, GroupSoftDeltaEpilogueArguments>(params, stream);
        return;
      }
    }
  }
  RunAttentionFwdSm90KernelTile<
      kHeadDim, kHeadDimV, Element, Element, false, kHasSegment, true,
      kBlockM, kBlockN, kMmaPVIsRS, kIntraWGOverlap, kUseFastExp2, Visibility,
      flash::GroupSoftDeltaFwdSm90, 1, false, false, void,
      GroupSoftDeltaEpilogueFwd, GroupSoftDeltaEpilogueArguments>(params, stream);
}

}  // namespace ops
}  // namespace xattn

#define XATTN_GROUP_SOFTDELTA_FWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV, Visibility, \
                                                              HasSegment)                            \
  namespace xattn {                                                                                  \
  namespace ops {                                                                                    \
  template void RunGroupSoftDeltaFwdSm90<Element, HeadDim, HeadDimV,                                 \
                                         attention::semantics::Visibility, HasSegment>(              \
      AttentionFwdParams&, cudaStream_t);                                                            \
  }                                                                                                  \
  }
// Causal: dense rows and segments (the segment instance also runs sliding windows with segments).
#define XATTN_GROUP_SOFTDELTA_FWD_SM90_INSTANTIATE(Element, HeadDim, HeadDimV)                       \
  XATTN_GROUP_SOFTDELTA_FWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                  \
                                                        CausalFullVisibility, false)                 \
  XATTN_GROUP_SOFTDELTA_FWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                  \
                                                        CausalSlidingWindowVisibility, true)
// Sliding windows on dense rows, sliding chunks on dense rows and with segments.
#define XATTN_GROUP_SOFTDELTA_FWD_LOCAL_SM90_INSTANTIATE(Element, HeadDim, HeadDimV)                 \
  XATTN_GROUP_SOFTDELTA_FWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                  \
                                                        CausalSlidingWindowVisibility, false)        \
  XATTN_GROUP_SOFTDELTA_FWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                  \
                                                        SlidingChunkVisibility, false)               \
  XATTN_GROUP_SOFTDELTA_FWD_SM90_INSTANTIATE_VISIBILITY(Element, HeadDim, HeadDimV,                  \
                                                        SlidingChunkVisibility, true)

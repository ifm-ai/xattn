#include "softdelta/fwd_template.cuh"
#include "softdelta/fwd_branchless_tile_scheduler.h"

namespace xattn {
namespace ops {

template void RunFlashSoftDeltaFwdSm90VD<
    cutlass::bfloat16_t, cutlass::bfloat16_t, 32, 256,
    attention::semantics::CausalFullVisibility>(
    AttentionFwdParams&, cudaStream_t);

template void RunFlashSoftDeltaFwdSm90VD<
    cutlass::bfloat16_t, cutlass::bfloat16_t, 32, 256,
    attention::semantics::CausalSlidingWindowVisibility>(
    AttentionFwdParams&, cudaStream_t);

template <>
void RunFlashSoftDeltaFwdSm90VD<
    cutlass::bfloat16_t, cutlass::bfloat16_t, 32, 256,
    attention::semantics::SlidingChunkVisibility>(
    AttentionFwdParams& params, cudaStream_t stream) {
  using Visibility =
      attention::semantics::SlidingChunkVisibility;
  if (MaybeRunFlashSoftDeltaInterleavedReaderFwdSm90VD<
          cutlass::bfloat16_t, cutlass::bfloat16_t, 32, 256,
          Visibility>(params, stream)) {
    return;
  }
  static constexpr int kBlockN =
      FlashSoftDeltaSingleCTABlockNFwdSm90(
          32, 256, Visibility::kKind);
  static constexpr bool kIntraWGOverlap =
      FlashSoftDeltaSingleCTAIntraWGOverlapFwdSm90(
          32, 256, Visibility::kKind);
  static_assert(kBlockN == 96);
  static_assert(kIntraWGOverlap);
  RunAttentionFwdSm90KernelTile<
      32, 256, cutlass::bfloat16_t, cutlass::bfloat16_t, false,
      false, true, 128, kBlockN, true, kIntraWGOverlap, true,
      Visibility, flash::FlashSoftDeltaFwdSm90, 1, true, true,
      FlashSoftDeltaSingleCTABranchlessTileScheduler<
          256, 32, FlashSoftDeltaAllHeadTileGrouping>,
      FlashSoftDeltaSingleCTAEpilogueFwd,
      FlashSoftDeltaOutputEpilogueArguments,
      FlashSoftDeltaMainloopAdapterFwdSm90, 2, false, false,
      true>(params, stream);
}

}  // namespace ops
}  // namespace xattn

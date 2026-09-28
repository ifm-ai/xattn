#pragma once

#include "attention/hopper/bwd_params.h"

#include <ATen/cuda/Exceptions.h>
#include <cuda_runtime_api.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>
#include <cutlass/numeric_types.h>
#include <limits>
#include <torch/types.h>
#include <type_traits>

#include "cute/tensor.hpp"
#include "attention/semantics/visibility.h"
#include "attention/hopper/epilogue_bwd.hpp"
#include "attention/hopper/tile_scheduler.hpp"
#include "attention/hopper/bwd_kernel_sm90.h"
#include "attention/hopper/bwd_postprocess_kernel.h"
#include "attention/hopper/bwd_preprocess_kernel.h"
#include "attention/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "attention/hopper/semantics/segment.h"
#include "attention/hopper/semantics/visibility.h"

namespace xattn {
namespace ops {

using namespace cute;

template <int kBlockM, int kBlockN, typename Visibility>
__global__ void AttentionBuildBwdSegmentWorklistKernel(
    int seqlen_q, int seqlen_k, int num_head, int num_batch,
    int window_size_left, int window_size_right, int attention_chunk,
    int64_t const* q_segment_idx, int64_t const* k_segment_idx,
    int k_segment_len, flash::SegmentBwdWorkTile* work_ptr) {
  int const num_n_blocks = cute::ceil_div(seqlen_k, kBlockN);
  int const total_dense_tiles = num_n_blocks * num_head * num_batch;
  int const dense_tile_idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (dense_tile_idx >= total_dense_tiles) {
    return;
  }

  cutlass::FastDivmod attention_chunk_divmod(
      attention_chunk >= 1 ? attention_chunk : 1);
  attention_chunk_divmod.divisor = attention_chunk;

  using SeqlenInfo = flash::SeqlenInfoQK<false /*Varlen*/, kBlockM>;
  using AttentionRange = std::conditional_t<
      Visibility::kIsSlidingChunk,
      attention::hopper::semantics::SlidingChunkRangeAdapter<
          SeqlenInfo, kBlockM, kBlockN,
          false /*Is_causal*/, true /*Is_local*/,
          false, false, true /*AllowReset*/>,
      attention::hopper::semantics::StandardRangeAdapter<
          SeqlenInfo, kBlockM, kBlockN,
          Visibility::kUseCausalMask, Visibility::kUseLocalMask>>;
  using SegmentPartition =
      attention::hopper::semantics::SegmentPartitionAdapter<
          kBlockM, kBlockN>;
  int n_block = dense_tile_idx;
  int const bidb = n_block / (num_n_blocks * num_head);
  n_block -= bidb * num_n_blocks * num_head;
  int const bidh = n_block / num_n_blocks;
  n_block -= bidh * num_n_blocks;

  SeqlenInfo seqlen_info{
      bidb, seqlen_q, seqlen_k,
      nullptr, nullptr, nullptr, nullptr};
  auto [m_block_min, m_block_max] = AttentionRange::bwd_m_block_range(
      seqlen_info, n_block, bidb, window_size_left, window_size_right,
      0 /*sink_token_length*/, attention_chunk_divmod);
  auto [segment_m_block_min, segment_m_block_max] =
      SegmentPartition::bwd_m_block_range(
          m_block_min, m_block_max, n_block, seqlen_q, seqlen_k,
          q_segment_idx, k_segment_idx, k_segment_len, bidb);
  m_block_min = segment_m_block_min;
  m_block_max = segment_m_block_max;
  if (m_block_max <= m_block_min) {
    work_ptr[dense_tile_idx] =
        flash::SegmentBwdWorkTile{dense_tile_idx, n_block, bidh, bidb,
                                  m_block_min, m_block_min};
    return;
  }
  work_ptr[dense_tile_idx] =
      flash::SegmentBwdWorkTile{dense_tile_idx, n_block, bidh, bidb,
                                m_block_min, m_block_max};
}

template <int kBlockM, int kBlockN, typename Visibility>
__global__ void AttentionBuildBwdDeterministicSegmentRangesKernel(
    int seqlen_q, int seqlen_k, int num_batch,
    int window_size_left, int window_size_right, int attention_chunk,
    int64_t const* q_segment_idx, int64_t const* k_segment_idx,
    int k_segment_len, int* range_ptr) {
  int const num_m_blocks = cute::ceil_div(seqlen_q, kBlockM);
  int const range_idx =
      int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (range_idx >= num_m_blocks * num_batch) {
    return;
  }
  int const bidb = range_idx / num_m_blocks;
  int const m_block = range_idx - bidb * num_m_blocks;
  cutlass::FastDivmod attention_chunk_divmod(
      attention_chunk >= 1 ? attention_chunk : 1);
  attention_chunk_divmod.divisor = attention_chunk;
  cutlass::FastDivmod qhead_per_khead_divmod(1);
  using SeqlenInfo = flash::SeqlenInfoQK<false /*Varlen*/, kBlockM>;
  using AttentionRange = std::conditional_t<
      Visibility::kIsSlidingChunk,
      attention::hopper::semantics::SlidingChunkRangeAdapter<
          SeqlenInfo, kBlockM, kBlockN,
          false /*Is_causal*/, true /*Is_local*/,
          false, false, true /*AllowReset*/>,
      attention::hopper::semantics::StandardRangeAdapter<
          SeqlenInfo, kBlockM, kBlockN,
          Visibility::kUseCausalMask, Visibility::kUseLocalMask>>;
  using SegmentPartition =
      attention::hopper::semantics::SegmentPartitionAdapter<
          kBlockM, kBlockN>;
  SeqlenInfo seqlen_info{
      bidb, seqlen_q, seqlen_k,
      nullptr, nullptr, nullptr, nullptr};
  auto [n_block_min, n_block_max] = AttentionRange::fwd_n_block_range(
      seqlen_info, m_block, bidb, 0 /*split_idx*/, 1 /*num_splits*/,
      window_size_left, window_size_right, attention_chunk_divmod,
      qhead_per_khead_divmod);
  auto [segment_n_block_min, segment_n_block_max] =
      SegmentPartition::fwd_n_block_range(
          n_block_min, n_block_max, m_block, seqlen_q, seqlen_k,
          q_segment_idx, k_segment_idx, k_segment_len, bidb);
  range_ptr[2 * range_idx] = segment_n_block_min;
  range_ptr[2 * range_idx + 1] = segment_n_block_max;
}

template <typename Element>
__global__ void AttentionCastGroupedKVGradsKernel(
    int64_t dk_numel, const float* __restrict__ dk_input,
    Element* __restrict__ dk_output, int64_t dv_numel,
    const float* __restrict__ dv_input, Element* __restrict__ dv_output) {
  int64_t const idx =
      int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (idx < dk_numel) {
    dk_output[idx] = static_cast<Element>(dk_input[idx]);
  }
  if (idx < dv_numel) {
    dv_output[idx] = static_cast<Element>(dv_input[idx]);
  }
}

// Cast directly into the unpadded gradient tensor.
template <typename Element, int kHeadDim, int kHeadDimV>
__global__ void AttentionCastTrimGroupedKVGradsKernel(
    int64_t dk_numel, const float* __restrict__ dk_input,
    Element* __restrict__ dk_output, int dk_dim, int64_t dv_numel,
    const float* __restrict__ dv_input, Element* __restrict__ dv_output,
    int dv_dim) {
  int64_t const idx =
      int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  int const dk_col = idx % kHeadDim;
  if (idx < dk_numel && dk_col < dk_dim) {
    dk_output[(idx / kHeadDim) * dk_dim + dk_col] =
        static_cast<Element>(dk_input[idx]);
  }
  int const dv_col = idx % kHeadDimV;
  if (idx < dv_numel && dv_col < dv_dim) {
    dv_output[(idx / kHeadDimV) * dv_dim + dv_col] =
        static_cast<Element>(dv_input[idx]);
  }
}

template <typename Element>
__global__ void AttentionAccumulateSecondaryKVGradsKernel(
    int64_t dk_vectors, int64_t dv_vectors,
    int seqlen_k, int kv_heads,
    int dk_vectors_per_head, const Element* __restrict__ dk_secondary,
    Element* __restrict__ dk_output,
    int64_t dk_row_stride, int64_t dk_head_stride,
    int64_t dk_batch_stride,
    int dv_vectors_per_head, const Element* __restrict__ dv_secondary,
    Element* __restrict__ dv_output,
    int64_t dv_row_stride, int64_t dv_head_stride,
    int64_t dv_batch_stride) {
  constexpr int kElementsPerVector = 8;
  using Access = cutlass::AlignedArray<
      Element, kElementsPerVector, 16>;
  int64_t const idx =
      int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (idx < dk_vectors) {
    int const column_vector = int(idx % dk_vectors_per_head);
    int64_t const output_row = idx / dk_vectors_per_head;
    int const head = int(output_row % kv_heads);
    int64_t const batch_row = output_row / kv_heads;
    int const row = int(batch_row % seqlen_k);
    int const batch = int(batch_row / seqlen_k);
    int64_t const destination =
        int64_t(batch) * dk_batch_stride + int64_t(row) * dk_row_stride +
        int64_t(head) * dk_head_stride +
        int64_t(column_vector) * kElementsPerVector;
    Access const primary =
        *reinterpret_cast<Access const*>(dk_output + destination);
    Access const second =
        reinterpret_cast<Access const*>(dk_secondary)[idx];
    Access merged;
    #pragma unroll
    for (int lane = 0; lane < kElementsPerVector; ++lane) {
      merged[lane] = static_cast<Element>(
          static_cast<float>(primary[lane]) +
          static_cast<float>(second[lane]));
    }
    *reinterpret_cast<Access*>(dk_output + destination) = merged;
  }
  if (idx < dv_vectors) {
    int const column_vector = int(idx % dv_vectors_per_head);
    int64_t const output_row = idx / dv_vectors_per_head;
    int const head = int(output_row % kv_heads);
    int64_t const batch_row = output_row / kv_heads;
    int const row = int(batch_row % seqlen_k);
    int const batch = int(batch_row / seqlen_k);
    int64_t const destination =
        int64_t(batch) * dv_batch_stride + int64_t(row) * dv_row_stride +
        int64_t(head) * dv_head_stride +
        int64_t(column_vector) * kElementsPerVector;
    Access const primary =
        *reinterpret_cast<Access const*>(dv_output + destination);
    Access const second =
        reinterpret_cast<Access const*>(dv_secondary)[idx];
    Access merged;
    #pragma unroll
    for (int lane = 0; lane < kElementsPerVector; ++lane) {
      merged[lane] = static_cast<Element>(
          static_cast<float>(primary[lane]) +
          static_cast<float>(second[lane]));
    }
    *reinterpret_cast<Access*>(dv_output + destination) = merged;
  }
}

template <int kHeadDim, int kHeadDimV, int kRequestedBlockN, typename Element,
          bool kDeterministic, bool kVarlen, bool kHasSegment,
          int kBlockMOverride = 0, bool kGQA = false,
          bool kTwoComponentDV = false,
          typename Visibility =
              attention::semantics::SlidingChunkVisibility,
          template <class, class, class> class BwdKernel =
              flash::AttentionBwdSm90,
          int kStagesOverride = 0, int kStagesDOOverride = 0,
          int kStagesDSOverride = 0,
          bool kPersistentScheduler = false,
          bool kDirectChunkRange = Visibility::kIsSlidingChunk,
          bool kDirectChunkTile = kDirectChunkRange,
          bool kReaderPairKVReuse = false,
          bool kHeadPairParallel = false,
          bool kHeadPairClusterKVReuse = false,
          bool kHeadPairPrivateKVGrad = false>
void RunAttentionBwdSm90Kernel(
    AttentionBwdParams params, cudaStream_t stream) {
  static constexpr bool kWindowN64 =
      !kDeterministic && kRequestedBlockN == 128 &&
      Visibility::kKind == attention::semantics::VisibilityKind::kCausalSlidingWindow &&
      ((kHeadDim == 160 && kHeadDimV == 192) ||
       (kHeadDim == 192 && kHeadDimV == 160));
  static constexpr bool kFullStage2 =
      kDeterministic && kStagesOverride == 0 && kHeadDim == 192 &&
      (kHeadDimV == 32 || kHeadDimV == 64) &&
      Visibility::kKind == attention::semantics::VisibilityKind::kCausalFull;
  if constexpr (!kVarlen && !kHasSegment &&
                (!kReaderPairKVReuse || kWindowN64) &&
                !kHeadPairParallel && (kWindowN64 || kFullStage2)) {
    if (params.d == kHeadDim && params.dv == kHeadDimV &&
        params.seqlen_q == params.seqlen_k &&
        (!kReaderPairKVReuse ||
         (params.b == 1 && params.h == 16 && params.h_k == 1 &&
          params.seqlen_q == 8192 && params.window_size_left == 255))) {
      RunAttentionBwdSm90Kernel<
          kHeadDim, kHeadDimV, kWindowN64 ? 64 : kRequestedBlockN, Element,
          kDeterministic, kVarlen, kHasSegment, kBlockMOverride, kGQA,
          kTwoComponentDV, Visibility, BwdKernel,
          kFullStage2 ? 2 : kStagesOverride, kStagesDOOverride,
          kStagesDSOverride, kPersistentScheduler, kDirectChunkRange,
          kDirectChunkTile, kReaderPairKVReuse, kHeadPairParallel,
          kHeadPairClusterKVReuse, kHeadPairPrivateKVGrad>(params, stream);
      return;
    }
  }
  // Dense local tile width for wide dQ/dK accumulators.
  static constexpr int kBlockN =
      Visibility::kKind != attention::semantics::VisibilityKind::kCausalFull &&
      !kHasSegment && !kVarlen && !kDeterministic && kHeadDim == 256 &&
      (kHeadDimV == 160 || kHeadDimV == 192)
          ? 64 : kRequestedBlockN;
  static_assert(!(kVarlen && kHasSegment),
                "attention SM90 BWD cannot enable varlen and segment mask together");
  static_assert(!kReaderPairKVReuse || (!kVarlen && !kHasSegment),
                "reader-pair K/V reuse requires dense attention BWD");
  static_assert(!kHeadPairParallel ||
                    (!kVarlen && !kHasSegment && !kDeterministic && kGQA),
                "parallel head-pair execution requires dense, grouped-output, non-deterministic BWD");
  static_assert(!kHeadPairClusterKVReuse || kHeadPairParallel,
                "cluster K/V reuse requires parallel head-pair execution");
  static_assert(!kHeadPairPrivateKVGrad || kHeadPairParallel,
                "private K/V gradients require parallel head-pair execution");
  static_assert(!(kReaderPairKVReuse && kHeadPairParallel),
                "same-CTA and parallel head-pair execution are exclusive");
  if constexpr (kReaderPairKVReuse || kHeadPairParallel) {
    TORCH_CHECK(
        params.h % 2 == 0 && (params.h / 2) % params.h_k == 0,
        "paired-reader execution requires an even Q-head count and logical "
        "Q heads divisible by KV heads");
    TORCH_CHECK(
        params.odd_head_window_right_delta == -1 &&
            params.window_size_right == 0,
        "paired-reader execution requires inclusive/strict-past paired "
        "boundaries");
  }
  if constexpr (kHeadPairParallel) {
    TORCH_CHECK(
        params.h == 2 * params.h_k,
        "parallel head-pair execution requires two physical Q "
        "heads per KV head");
  }
  int const scheduled_q_heads =
      (kReaderPairKVReuse || kHeadPairParallel)
          ? params.h_k
          : params.h;
  static constexpr bool IsCausal = Visibility::kUseCausalMask;
  static constexpr bool IsLocal = Visibility::kUseLocalMask;
  static constexpr bool HasSoftcap = false;
  static constexpr bool Varlen = kVarlen;
  static constexpr int kMaxHeadDim = kHeadDim > kHeadDimV ? kHeadDim : kHeadDimV;
  static constexpr bool UseLargeBlockM =
      kHeadDim <= 64 && kHeadDimV <= 128 &&
      (!kVarlen || kDeterministic || kHeadDimV <= 64);
  static constexpr int kDefaultBlockM = UseLargeBlockM ? 128 : 64;
  static constexpr int kBlockM =
      kBlockMOverride > 0 ? kBlockMOverride : kDefaultBlockM;
  static_assert(kBlockM == 64 || kBlockM == 128,
                "attention SM90 BWD supports BM 64 or 128");
  static constexpr int DefaultStagesDO =
      kMaxHeadDim <= 128 && (kBlockM < 128 || kHeadDimV <= 64) ? 2 : 1;
  static constexpr int DefaultStagesDS = kMaxHeadDim <= 128 ? 2 : 1;
  static constexpr int StagesDO =
      kStagesDOOverride > 0 ? kStagesDOOverride : DefaultStagesDO;
  static constexpr int StagesDS =
      kStagesDSOverride > 0 ? kStagesDSOverride : DefaultStagesDS;
  static constexpr bool SdPSwapAB = kHeadDim <= 128 && kBlockN % 64 == 0;
  // dKV swap layout: D == V.
  static constexpr bool DKvSwapAB =
      kHeadDim == kHeadDimV && kMaxHeadDim >= 192 &&
      kHeadDim % 64 == 0 && kHeadDimV % 64 == 0;
  static constexpr bool DqSwapAB = !kDeterministic && kHeadDim == 256;
  // Three-WG layout: D=V=192.
  static constexpr int NumMmaWarpGroups =
      kHeadDim == 192 && kHeadDimV == 192 ? 3 : 2;
  static constexpr int AtomLayoutMSdP = 1;
  // Varlen mixed-D/V layouts use unsplit K/V slices when either dimension is
  // not warp-group aligned.
  static constexpr bool DKvCanSplitHeadDim =
      kMaxHeadDim > 128 &&
      kHeadDim % (NumMmaWarpGroups * 64) == 0 &&
      kHeadDimV % (NumMmaWarpGroups * 64) == 0;
  static constexpr int AtomLayoutNdKV =
      kMaxHeadDim <= 128
          ? 2
          : (kVarlen && !DKvCanSplitHeadDim ? NumMmaWarpGroups : 1);
  static constexpr int AtomLayoutMdQ =
      kHeadDim == 64 && kBlockM == 128 ? 2 : 1;
  static constexpr bool MmaDPIsRS = SdPSwapAB && kHeadDimV == 96;
  static constexpr bool MmaDKvIsRS =
      AtomLayoutMSdP == 1 && AtomLayoutNdKV == NumMmaWarpGroups &&
      SdPSwapAB && !DKvSwapAB;
  static constexpr size_t Stage2TensorStorageEstimate =
      size_t(kBlockN) * (kHeadDim + kHeadDimV) * sizeof(Element) +
      ((kHeadDim < 256 || kDeterministic)
           ? size_t(kBlockM) * kHeadDim * sizeof(float)
           : 0) +
      size_t(2) * kBlockM * kHeadDim * sizeof(Element) +
      size_t(StagesDO) * kBlockM * kHeadDimV * sizeof(Element) +
      size_t(4) * ((kBlockM + 63) / 64 * 64) * sizeof(float) +
      size_t(StagesDS) * kBlockM * kBlockN * sizeof(Element) *
          (1 + int(!MmaDKvIsRS) +
           int(!MmaDKvIsRS && kTwoComponentDV));
  static constexpr bool UseDenseStage2 =
      !kDeterministic && !kVarlen && !kHasSegment &&
      Stage2TensorStorageEstimate <= 224 * 1024;
  static constexpr bool UseDenseDetStage2 =
      kDeterministic && Visibility::kIsSlidingChunk && !kVarlen &&
      !kHasSegment && !kGQA &&
      (kHeadDim == 160 || kHeadDim == 192) &&
      Stage2TensorStorageEstimate <= 224 * 1024;
  static constexpr int DefaultStages =
      kMaxHeadDim <= 128 || UseDenseStage2 || UseDenseDetStage2 ? 2 : 1;
  static constexpr int Stages =
      kStagesOverride > 0 ? kStagesOverride : DefaultStages;
  static_assert(Stages > 0 && StagesDO > 0 && StagesDS > 0,
                "attention SM90 BWD pipeline stages must be positive");
  static_assert(kHeadDim % 8 == 0,
                "attention SM90 BWD requires D to be a multiple of 8");
  static_assert(kHeadDimV % 8 == 0,
                "attention SM90 BWD requires V to be a multiple of 8");
  static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0,
                "attention SM90 BWD SdP atom layout must divide WG count");
  static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0,
                "attention SM90 BWD dKV atom layout must divide WG count");
  static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0,
                "attention SM90 BWD dQ atom layout must divide WG count");
  static_assert(!MmaDPIsRS || SdPSwapAB,
                "attention SM90 BWD RS dP requires SdP swap");

  static constexpr int kSdPWarpGroupSplit =
      NumMmaWarpGroups / AtomLayoutMSdP;
  static constexpr int kDKvWarpGroupSplit =
      NumMmaWarpGroups / AtomLayoutNdKV;
  static constexpr int kDQWarpGroupSplit =
      NumMmaWarpGroups / AtomLayoutMdQ;
  static_assert(kBlockN % kSdPWarpGroupSplit == 0,
                "attention SM90 BWD BN must split evenly across SdP WGs");
  static_assert(kHeadDim % kDKvWarpGroupSplit == 0,
                "attention SM90 BWD D must split evenly across dK WGs");
  static_assert(kHeadDimV % kDKvWarpGroupSplit == 0,
                "attention SM90 BWD V must split evenly across dV WGs");
  static_assert(kHeadDim % kDQWarpGroupSplit == 0,
                "attention SM90 BWD D must split evenly across dQ WGs");
  static_assert(kBlockM % AtomLayoutMSdP == 0,
                "attention SM90 BWD BM must split evenly across SdP atoms");
  static_assert(kBlockN % AtomLayoutNdKV == 0,
                "attention SM90 BWD BN must split evenly across dKV atoms");
  static_assert(kBlockM % AtomLayoutMdQ == 0,
                "attention SM90 BWD BM must split evenly across dQ atoms");

  static constexpr int kSdPGmmaM = SdPSwapAB ? kBlockN : kBlockM;
  static constexpr int kSdPGmmaN =
      SdPSwapAB ? kBlockM / AtomLayoutMSdP
                : kBlockN / kSdPWarpGroupSplit;
  static constexpr int kDKGmmaM = DKvSwapAB ? kHeadDim : kBlockN;
  static constexpr int kDVGmmaM = DKvSwapAB ? kHeadDimV : kBlockN;
  static constexpr int kDKvGmmaN =
      DKvSwapAB ? kBlockN / AtomLayoutNdKV
                : kHeadDim / kDKvWarpGroupSplit;
  static constexpr int kDVGmmaN =
      DKvSwapAB ? kBlockN / AtomLayoutNdKV
                : kHeadDimV / kDKvWarpGroupSplit;
  static constexpr int kDQGmmaM = DqSwapAB ? kHeadDim : kBlockM;
  static constexpr int kDQGmmaN =
      DqSwapAB ? kBlockM / AtomLayoutMdQ
               : kHeadDim / kDQWarpGroupSplit;
  static_assert(kSdPGmmaM % 64 == 0,
                "attention SM90 BWD SdP/dP GMMA M must be a multiple of 64");
  static_assert(kDKGmmaM % 64 == 0,
                "attention SM90 BWD dK GMMA M must be a multiple of 64");
  static_assert(kDVGmmaM % 64 == 0,
                "attention SM90 BWD dV GMMA M must be a multiple of 64");
  static_assert(kDQGmmaM % 64 == 0,
                "attention SM90 BWD dQ GMMA M must be a multiple of 64");
  static_assert(kSdPGmmaN % 8 == 0,
                "attention SM90 BWD SdP/dP GMMA N must be a multiple of 8");
  static_assert(kDKvGmmaN % 8 == 0,
                "attention SM90 BWD dK GMMA N must be a multiple of 8");
  static_assert(kDVGmmaN % 8 == 0,
                "attention SM90 BWD dV GMMA N must be a multiple of 8");
  static_assert(kDQGmmaN % 8 == 0,
                "attention SM90 BWD dQ GMMA N must be a multiple of 8");
  static_assert((kBlockM * kHeadDim) % NumMmaWarpGroups == 0,
                "attention SM90 BWD dQ accumulator must split across WGs");
  static_assert((kBlockM * kBlockN) %
                        (NumMmaWarpGroups *
                         cutlass::NumThreadsPerWarpGroup) ==
                    0,
                "attention SM90 BWD PdS store must split across WG threads");

  using ElementAccum = float;
  using ArchTag = cutlass::arch::Sm90;
  using TileShapeMK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
  using TileShapeMV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>>;
  const int seqlen_q_rounded =
      kVarlen ? params.seqlen_q_padded : cute::round_up(params.seqlen_q, kBlockM);
  const int logical_batch = kVarlen ? params.num_sequences : params.b;
  // Varlen workspace batch: 1; launch and scheduling batch: num_sequences.
  const int workspace_batch = kVarlen ? 1 : logical_batch;
  static constexpr bool SplitDQ =
      kDeterministic && Visibility::kIsSlidingChunk &&
      !kVarlen && !kHasSegment;
  const int64_t dq_slot_stride =
      int64_t(workspace_batch) * params.h * seqlen_q_rounded * kHeadDim;
  int dq_slots = 1;
  if constexpr (SplitDQ) {
    const int64_t chunk = params.attention_chunk;
    if (chunk > 0) {
      // Collision-free dQ slots cover BM + 2*C - 1 visible keys
      // and up to BN - 1 alignment positions.
      int64_t slots = cute::ceil_div(
          2 * chunk + kBlockM + kBlockN - 2, int64_t(kBlockN));
      if (chunk % kBlockM == 0 && chunk % kBlockN == 0 &&
          (params.seqlen_k - params.seqlen_q) % chunk == 0) {
        slots = 2 * chunk / kBlockN;
      }
      // Ordered reduction above the partial-workspace limit.
      constexpr int64_t kMaxPartialBytes = int64_t(16) << 30;
      if (slots > 1 && slots <= 16 &&
          dq_slot_stride <= kMaxPartialBytes / sizeof(float) / slots) {
        dq_slots = int(slots);
      }
    }
  }
  torch::Tensor dpsum = torch::empty(
      {workspace_batch, params.h, seqlen_q_rounded},
      torch::TensorOptions()
          .device(torch::kCUDA)
          .dtype(at::kFloat)
          .memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor lse_log2 = torch::empty_like(dpsum);
  const auto dq_options = torch::TensorOptions()
      .device(torch::kCUDA).dtype(at::kFloat)
      .memory_format(at::MemoryFormat::Contiguous);
  torch::Tensor dq_accum = dq_slots > 1
      ? torch::zeros({dq_slots, dq_slot_stride}, dq_options)
      : torch::empty({dq_slot_stride}, dq_options);
  const int num_m_block = cute::ceil_div(
      kVarlen ? params.max_seqlen_q : params.seqlen_q, kBlockM);
  dim3 auxiliary_grid(num_m_block, params.h, logical_batch);
  if constexpr (kVarlen) {
    if (logical_batch > 65535) {
      int64_t const auxiliary_grid_blocks =
          int64_t(num_m_block) * params.h * logical_batch;
      TORCH_CHECK(
          auxiliary_grid_blocks <= std::numeric_limits<int>::max(),
          "attention SM90 BWD auxiliary grid exceeds the CUDA x-dimension limit");
      auxiliary_grid = dim3(uint32_t(auxiliary_grid_blocks));
    }
  }
  torch::Tensor dq_semaphore;
  params.dpsum_ptr = dpsum.data_ptr<float>();
  params.lse_log2_ptr = lse_log2.data_ptr<float>();
  params.dq_accum_ptr = dq_accum.data_ptr<float>();
  if constexpr (kDeterministic) {
    dq_semaphore = torch::empty(
        {num_m_block, logical_batch, params.h},
        torch::TensorOptions()
            .device(torch::kCUDA)
            .dtype(at::kInt)
            .memory_format(at::MemoryFormat::Contiguous));
    params.dq_semaphore = dq_semaphore.data_ptr<int>();
    // Preprocess initializes each live query tile's semaphore on this stream
    // before the main kernel. Padded varlen tiles have no semaphore consumers.
  } else {
    params.dq_semaphore = nullptr;
  }
  params.tile_count_semaphore = nullptr;

  auto launch_preprocess = [&](auto output_state_tag) {
    static constexpr bool OutputStateFp32 =
        decltype(output_state_tag)::value;
    using PreprocessKernel = flash::AttentionBwdPreprocess<
        TileShapeMV, kHeadDim, Element, ElementAccum, ArchTag,
        true /*Clear_dQaccum*/, Varlen, OutputStateFp32>;
    typename PreprocessKernel::Arguments preprocess_args{
        OutputStateFp32 ? nullptr
                        : static_cast<Element const*>(params.y_ptr),
        {params.seqlen_q, params.dv, params.h, params.b},
        {params.y_row_stride, _1{}, params.y_head_stride,
         params.y_batch_stride},
        static_cast<Element const*>(params.dy_ptr),
        {params.dy_row_stride, _1{}, params.dy_head_stride,
         params.dy_batch_stride},
        static_cast<float*>(params.dpsum_ptr),
        {seqlen_q_rounded, params.h, workspace_batch},
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},
        static_cast<float const*>(params.lse_ptr),
        {_1{}, params.seqlen_q, params.h * params.seqlen_q},
        static_cast<float*>(params.lse_log2_ptr),
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * kHeadDim, params.h, workspace_batch},
        {_1{}, seqlen_q_rounded * kHeadDim,
         params.h * seqlen_q_rounded * kHeadDim},
        logical_batch,
        num_m_block,
        params.dq_semaphore,
        kVarlen ? params.cu_seqlens_q : nullptr,
        nullptr,
        OutputStateFp32 ? static_cast<float const*>(params.y_ptr)
                        : nullptr};
    typename PreprocessKernel::Params preprocess_params =
        PreprocessKernel::to_underlying_arguments(preprocess_args);
    return cutlass::kernel_launch<PreprocessKernel>(
        auxiliary_grid,
        PreprocessKernel::MaxThreadsPerBlock,
        PreprocessKernel::SharedStorageSize, stream, preprocess_params,
        false);
  };
  cutlass::Status preprocess_status = params.y_is_fp32
      ? launch_preprocess(std::true_type{})
      : launch_preprocess(std::false_type{});
  TORCH_CHECK(preprocess_status == cutlass::Status::kSuccess,
              "attention SM90 BWD preprocess launch failed: ",
              cutlass::cutlassGetStatusString(preprocess_status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using TileShapeMNK =
      cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using ClusterShape = cute::Shape<
      Int<kHeadPairClusterKVReuse ? 2 : 1>, _1, _1>;
  using CollectiveMainloop = flash::CollectiveMainloopBwdSm90<
      Stages, StagesDO, StagesDS, ClusterShape, TileShapeMNK, kHeadDimV,
      Element, ElementAccum, ArchTag, IsCausal, IsLocal, HasSoftcap, Varlen,
      kDeterministic, kHasSegment, kDirectChunkRange,
      SdPSwapAB, DKvSwapAB,
      DqSwapAB,
      NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
      MmaDPIsRS, kTwoComponentDV, kReaderPairKVReuse, kDirectChunkTile,
      kHeadPairParallel, kHeadPairClusterKVReuse,
      kHeadPairPrivateKVGrad, SplitDQ>;
  using CollectiveEpilogueDirect = flash::CollectiveEpilogueBwd<
      TileShapeMNK, kHeadDimV, Element, ArchTag,
      CollectiveMainloop::NumMmaThreads, Varlen, DKvSwapAB,
      NumMmaWarpGroups / AtomLayoutNdKV>;
  using CollectiveEpilogueSplitDestinations =
      flash::CollectiveEpilogueBwdSplitBatchDestinations<
          TileShapeMNK, kHeadDimV, Element, ArchTag,
          CollectiveMainloop::NumMmaThreads, Varlen, DKvSwapAB,
          NumMmaWarpGroups / AtomLayoutNdKV>;
  using CollectiveEpilogueMHA = std::conditional_t<
      kHeadPairPrivateKVGrad, CollectiveEpilogueSplitDestinations,
      CollectiveEpilogueDirect>;
  using CollectiveEpilogueGQA = flash::CollectiveEpilogueBwdGQA<
      TileShapeMNK, kHeadDimV, ElementAccum, ArchTag,
      CollectiveMainloop::NumMmaThreads, Varlen, kDeterministic, DKvSwapAB>;
  using CollectiveEpilogue = std::conditional_t<
      kGQA && !kReaderPairKVReuse && !kHeadPairPrivateKVGrad,
      CollectiveEpilogueGQA, CollectiveEpilogueMHA>;
  using DenseScheduler =
      flash::SingleTileScheduler<Varlen, false /*Split*/, false /*PackGQA*/,
                                 kBlockN>;
  using DenseLPTScheduler =
      flash::SingleTileBwdLPTScheduler<
          false, kBlockN, false /*SPT*/,
          kHeadPairParallel ? 2 : 1,
          !kHeadPairParallel ||
              kHeadPairClusterKVReuse /*ContiguousWorkGroup*/>;
  using CausalDeterministicScheduler =
      flash::SingleTileBwdLPTScheduler<false, kBlockN, true /*SPT*/>;
  static constexpr int VarlenSchedulerProducerThreads =
      cutlass::NumThreadsPerWarp *
      (CollectiveMainloop::dQacc_use_TMA ? 2 : 1);
  static constexpr bool UseDensePersistentScheduler =
      kPersistentScheduler && kDeterministic && !kVarlen &&
      !kHasSegment && !kGQA;
  using DeterministicDenseScheduler =
      std::conditional_t<
          Visibility::kKind ==
              attention::semantics::VisibilityKind::kCausalFull,
          CausalDeterministicScheduler,
          flash::DeterministicDenseBwdTileScheduler<IsCausal>>;
  using DeterministicDensePersistentScheduler =
      flash::DeterministicDensePersistentBwdTileScheduler<
          CollectiveMainloop::NumMmaThreads,
          VarlenSchedulerProducerThreads>;
  static constexpr bool UseVarlenPersistentScheduler =
      kVarlen && kDeterministic;
  using VarlenPersistentScheduler =
      flash::VarlenSequencePersistentTileScheduler<
          kBlockN, CollectiveMainloop::NumMmaThreads,
          VarlenSchedulerProducerThreads, true /*WaitForWorkCompletion*/>;
  using SegmentScheduler = flash::StaticSegmentBwdTileScheduler;
  using DeterministicSegmentScheduler =
      flash::DeterministicSegmentBwdTileScheduler;
  using Scheduler =
      std::conditional_t<
          kHasSegment,
          std::conditional_t<kDeterministic,
                             DeterministicSegmentScheduler,
                             SegmentScheduler>,
                         std::conditional_t<
                             UseVarlenPersistentScheduler,
                             VarlenPersistentScheduler,
                             std::conditional_t<
                                 kDeterministic && !kVarlen && !kHasSegment,
                                 std::conditional_t<
                                     UseDensePersistentScheduler,
                                     DeterministicDensePersistentScheduler,
                                     DeterministicDenseScheduler>,
                                 std::conditional_t<kVarlen, DenseScheduler,
                                                    DenseLPTScheduler>>>>;
  using AttnKernel = flash::enable_sm90<BwdKernel<
      CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

  torch::Tensor deterministic_segment_ranges;
  int const* deterministic_segment_ranges_ptr = nullptr;
  if constexpr (kHasSegment && kDeterministic) {
    deterministic_segment_ranges = torch::empty(
        {params.b, num_m_block, 2},
        torch::TensorOptions()
            .device(torch::kCUDA)
            .dtype(at::kInt)
            .memory_format(at::MemoryFormat::Contiguous));
    deterministic_segment_ranges_ptr =
        deterministic_segment_ranges.data_ptr<int>();
    int constexpr kRangeThreads = 256;
    int const range_count = params.b * num_m_block;
    int const range_blocks =
        cutlass::ceil_div(range_count, kRangeThreads);
    AttentionBuildBwdDeterministicSegmentRangesKernel<
        kBlockM, kBlockN, Visibility>
        <<<range_blocks, kRangeThreads, 0, stream>>>(
            params.seqlen_q, params.seqlen_k, params.b,
            params.window_size_left, params.window_size_right,
            params.attention_chunk, params.q_segment_idx,
            params.k_segment_idx, params.k_segment_len,
            deterministic_segment_ranges.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  typename CollectiveMainloop::Arguments mainloop_args{
      static_cast<Element const*>(params.q_ptr),
      {params.seqlen_q, params.d, params.h, params.b},
      {params.q_row_stride, _1{}, params.q_head_stride,
       params.q_batch_stride},
      static_cast<Element const*>(params.k_ptr),
      {params.seqlen_k, params.d, params.h_k, params.b},
      {params.k_row_stride, _1{}, params.k_head_stride,
       params.k_batch_stride},
      static_cast<Element const*>(params.v_ptr),
      {params.seqlen_k, params.dv, params.h_k, params.b},
      {params.v_row_stride, _1{}, params.v_head_stride,
       params.v_batch_stride},
      static_cast<Element const*>(params.dy_ptr),
      {params.seqlen_q, params.dv, params.h, params.b},
      {params.dy_row_stride, _1{}, params.dy_head_stride,
       params.dy_batch_stride},
      static_cast<ElementAccum*>(params.dq_accum_ptr),
      {seqlen_q_rounded * kHeadDim, params.h, workspace_batch},
      {_1{}, seqlen_q_rounded * kHeadDim,
       params.h * seqlen_q_rounded * kHeadDim},
      static_cast<float const*>(params.lse_log2_ptr),
      {seqlen_q_rounded, params.h, workspace_batch},
      {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},
      static_cast<float const*>(params.dpsum_ptr),
      {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},
      params.scale_softmax,
      params.window_size_left,
      params.window_size_right,
      params.attention_chunk,
      0.0f,
      logical_batch,
      params.dq_semaphore,
      kVarlen ? params.cu_seqlens_q : nullptr,
      kVarlen ? params.cu_seqlens_k : nullptr,
      nullptr,
      kVarlen ? params.seqused_k : nullptr,
      kHasSegment ? params.q_segment_idx : nullptr,
      kHasSegment ? params.k_segment_idx : nullptr,
      kHasSegment ? params.k_segment_len : 0,
      kVarlen ? params.q_position_offsets : nullptr,
      kHasSegment ? params.q_chunk_positions : nullptr,
      kHasSegment ? params.reset_attention_chunk : 0,
      deterministic_segment_ranges_ptr,
      params.odd_head_window_right_delta,
      dq_slots, dq_slot_stride};

  int num_blocks_n = cutlass::ceil_div(
      kVarlen ? params.max_seqlen_k : params.seqlen_k, kBlockN);
  torch::Tensor secondary_dkv;
  torch::Tensor secondary_dk;
  torch::Tensor secondary_dv;
  torch::Tensor dkv_accum;
  torch::Tensor dk_accum;
  torch::Tensor dv_accum;
  torch::Tensor dk_semaphore;
  torch::Tensor dv_semaphore;
  if constexpr (kHeadPairPrivateKVGrad) {
    at::ScalarType constexpr pair_dtype =
        std::is_same_v<Element, cutlass::bfloat16_t>
            ? at::kBFloat16
            : at::kHalf;
    auto pair_options = torch::TensorOptions()
                            .device(torch::kCUDA)
                            .dtype(pair_dtype)
                            .memory_format(at::MemoryFormat::Contiguous);
    int64_t const dk_numel = int64_t(params.b) * params.seqlen_k *
                             params.h_k * kHeadDim;
    int64_t const dv_numel = int64_t(params.b) * params.seqlen_k *
                             params.h_k * kHeadDimV;
    secondary_dkv = torch::empty({dk_numel + dv_numel}, pair_options);
    secondary_dk = secondary_dkv.narrow(0, 0, dk_numel).view(
        {params.b, params.seqlen_k, params.h_k, kHeadDim});
    secondary_dv = secondary_dkv.narrow(0, dk_numel, dv_numel).view(
        {params.b, params.seqlen_k, params.h_k, kHeadDimV});
  } else if constexpr (kGQA && !kReaderPairKVReuse) {
    auto accum_options = torch::TensorOptions()
                             .device(torch::kCUDA)
                             .dtype(at::kFloat)
                             .memory_format(at::MemoryFormat::Contiguous);
    int64_t const dk_numel = int64_t(params.b) * params.seqlen_k *
                             params.h_k * kHeadDim;
    int64_t const dv_numel = int64_t(params.b) * params.seqlen_k *
                             params.h_k * kHeadDimV;
    dkv_accum = torch::zeros({dk_numel + dv_numel}, accum_options);
    dk_accum = dkv_accum.narrow(0, 0, dk_numel).view(
        {params.b, params.seqlen_k, params.h_k, kHeadDim});
    dv_accum = dkv_accum.narrow(0, dk_numel, dv_numel).view(
        {params.b, params.seqlen_k, params.h_k, kHeadDimV});
    if constexpr (kDeterministic) {
      auto semaphore_options = torch::TensorOptions()
                                   .device(torch::kCUDA)
                                   .dtype(at::kInt)
                                   .memory_format(
                                       at::MemoryFormat::Contiguous);
      int64_t const semaphore_count =
          int64_t(num_blocks_n) * logical_batch * params.h_k;
      dk_semaphore = torch::zeros({semaphore_count}, semaphore_options);
      dv_semaphore = torch::zeros({semaphore_count}, semaphore_options);
    }
  }

  typename CollectiveEpilogue::Arguments epilogue_args = [&] {
    if constexpr (kHeadPairPrivateKVGrad) {
      using BaseArguments = typename CollectiveEpilogue::BaseArguments;
      int64_t const dk_head_stride = kHeadDim;
      int64_t const dv_head_stride = kHeadDimV;
      int64_t const dk_row_stride = params.h_k * dk_head_stride;
      int64_t const dv_row_stride = params.h_k * dv_head_stride;
      int64_t const dk_batch_stride =
          int64_t(params.seqlen_k) * dk_row_stride;
      int64_t const dv_batch_stride =
          int64_t(params.seqlen_k) * dv_row_stride;
      return typename CollectiveEpilogue::Arguments{
          BaseArguments{
              static_cast<Element*>(params.dk_ptr),
              {params.seqlen_k, kHeadDim, params.h_k, params.b},
              {params.dk_row_stride, _1{}, params.dk_head_stride,
               params.dk_batch_stride},
              static_cast<Element*>(params.dv_ptr),
              {params.seqlen_k, kHeadDimV, params.h_k, params.b},
              {params.dv_row_stride, _1{}, params.dv_head_stride,
               params.dv_batch_stride},
              params.b,
              params.h_k,
              nullptr,
              nullptr,
              nullptr,
              nullptr},
          BaseArguments{
              reinterpret_cast<Element*>(secondary_dk.data_ptr()),
              {params.seqlen_k, kHeadDim, params.h_k, params.b},
              {dk_row_stride, _1{}, dk_head_stride, dk_batch_stride},
              reinterpret_cast<Element*>(secondary_dv.data_ptr()),
              {params.seqlen_k, kHeadDimV, params.h_k, params.b},
              {dv_row_stride, _1{}, dv_head_stride, dv_batch_stride},
              params.b,
              params.h_k,
              nullptr,
              nullptr,
              nullptr,
              nullptr}};
    } else if constexpr (kGQA && !kReaderPairKVReuse) {
      int64_t const dk_head_stride = kHeadDim;
      int64_t const dv_head_stride = kHeadDimV;
      int64_t const dk_row_stride = params.h_k * dk_head_stride;
      int64_t const dv_row_stride = params.h_k * dv_head_stride;
      int64_t const dk_batch_stride =
          int64_t(params.seqlen_k) * params.h_k * kHeadDim;
      int64_t const dv_batch_stride =
          int64_t(params.seqlen_k) * params.h_k * kHeadDimV;
      return typename CollectiveEpilogue::Arguments{
          dk_accum.data_ptr<float>(),
          {params.seqlen_k, kHeadDim, params.h_k, params.b},
          {dk_row_stride, _1{}, dk_head_stride,
           dk_batch_stride},
          dv_accum.data_ptr<float>(),
          {params.seqlen_k, kHeadDimV, params.h_k, params.b},
          {dv_row_stride, _1{}, dv_head_stride,
           dv_batch_stride},
          logical_batch,
          scheduled_q_heads,
          kDeterministic ? dk_semaphore.data_ptr<int>() : nullptr,
          kDeterministic ? dv_semaphore.data_ptr<int>() : nullptr,
          kVarlen ? params.cu_seqlens_k : nullptr,
          kVarlen ? params.seqused_k : nullptr};
    } else {
      int const output_heads = kReaderPairKVReuse ? params.h_k : params.h;
      return typename CollectiveEpilogue::Arguments{
          static_cast<Element*>(params.dk_ptr),
          {params.seqlen_k, params.d, output_heads, params.b},
          {params.dk_row_stride, _1{}, params.dk_head_stride,
           params.dk_batch_stride},
          static_cast<Element*>(params.dv_ptr),
          {params.seqlen_k, params.dv, output_heads, params.b},
          {params.dv_row_stride, _1{}, params.dv_head_stride,
           params.dv_batch_stride},
          params.b,
          scheduled_q_heads,
          nullptr,
          nullptr,
          kVarlen ? params.cu_seqlens_k : nullptr,
          kVarlen ? params.seqused_k : nullptr};
    }
  }();

  torch::Tensor segment_work;
  flash::SegmentBwdWorkTile* segment_work_ptr = nullptr;
  if constexpr (kHasSegment) {
    static_assert(!kVarlen,
                  "Segment worklist is only used by dense segment BWD");
    static_assert(sizeof(flash::SegmentBwdWorkTile) % sizeof(int) == 0,
                  "SegmentBwdWorkTile must be int-addressable");
    int const max_segment_work = num_blocks_n * params.h * params.b;
    segment_work = torch::empty(
        {max_segment_work,
         static_cast<int64_t>(sizeof(flash::SegmentBwdWorkTile) /
                              sizeof(int))},
        torch::TensorOptions()
            .device(torch::kCUDA)
            .dtype(at::kInt)
            .memory_format(at::MemoryFormat::Contiguous));
    segment_work_ptr = reinterpret_cast<flash::SegmentBwdWorkTile*>(
        segment_work.data_ptr<int>());
    int constexpr kWorklistThreads = 256;
    int const worklist_blocks =
        cutlass::ceil_div(max_segment_work, kWorklistThreads);
    AttentionBuildBwdSegmentWorklistKernel<
        kBlockM, kBlockN, Visibility>
        <<<worklist_blocks, kWorklistThreads, 0, stream>>>(
            params.seqlen_q, params.seqlen_k, params.h, params.b,
            params.window_size_left, params.window_size_right,
            params.attention_chunk, params.q_segment_idx,
            params.k_segment_idx, params.k_segment_len, segment_work_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  torch::Tensor tile_count_semaphore;
  int const varlen_scheduler_mode =
      kVarlen ? flash::attention_varlen_scheduler_mode_from_env() : 0;
  if constexpr (UseVarlenPersistentScheduler) {
    bool const needs_tile_counter =
        VarlenPersistentScheduler::needs_tile_counter(
            params.h, params.num_sequences, params.max_seqlen_k,
            params.seqlen_k, params.num_sm, varlen_scheduler_mode);
    if (needs_tile_counter) {
      tile_count_semaphore = torch::empty(
          {1}, torch::TensorOptions()
                   .device(torch::kCUDA)
                   .dtype(at::kInt)
                   .memory_format(at::MemoryFormat::Contiguous));
      params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
      C10_CUDA_CHECK(cudaMemsetAsync(
          params.tile_count_semaphore, 0, sizeof(int), stream));
    } else {
      params.tile_count_semaphore = nullptr;
    }
  } else if constexpr (UseDensePersistentScheduler) {
    tile_count_semaphore = torch::empty(
        {1}, torch::TensorOptions()
                 .device(torch::kCUDA)
                 .dtype(at::kInt)
                 .memory_format(at::MemoryFormat::Contiguous));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
    C10_CUDA_CHECK(cudaMemsetAsync(
        params.tile_count_semaphore, 0, sizeof(int), stream));
  } else {
    params.tile_count_semaphore = nullptr;
  }
  typename flash::TileSchedulerArguments scheduler_args{
      num_blocks_n,
      scheduled_q_heads,
      logical_batch,
      1,
      1,
      kVarlen ? params.max_seqlen_k : params.seqlen_k,
      kVarlen ? params.max_seqlen_q : params.seqlen_q,
      params.d,
      params.dv,
      sizeof(Element),
      params.tile_count_semaphore,
      kVarlen ? params.cu_seqlens_k : nullptr,
      kVarlen ? params.seqused_k : nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      segment_work_ptr,
      params.seqlen_k,
      varlen_scheduler_mode};

  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int max_dynamic_smem = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  typename AttnKernel::Params kernel_params =
      AttnKernel::to_underlying_arguments(
          {mainloop_args, epilogue_args, {device, params.num_sm},
           scheduler_args});
  if constexpr (kVarlen && !kDeterministic) {
    if (logical_batch > 65535) {
      int64_t const mainloop_grid_blocks =
          int64_t(num_blocks_n) * scheduled_q_heads * logical_batch;
      TORCH_CHECK(
          mainloop_grid_blocks <= std::numeric_limits<int>::max(),
          "attention SM90 BWD mainloop grid exceeds the CUDA x-dimension limit");
    }
  }
  dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
  dim3 block_dims = AttnKernel::get_block_shape();
  int smem_size = AttnKernel::SharedStorageSize;
  auto kernel = cutlass::device_kernel<AttnKernel>;
  if (smem_size >= 48 * 1024) {
    TORCH_CHECK(
        smem_size <= max_dynamic_smem,
        "attention SM90 BWD mainloop requires too much dynamic shared memory: ",
        smem_size, " bytes, device opt-in limit is ", max_dynamic_smem,
        " bytes");
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
  }
  cutlass::Status status;
  if constexpr (kHeadPairClusterKVReuse) {
    void const* kernel_ptr =
        const_cast<void const*>(reinterpret_cast<void*>(kernel));
    status = cutlass::launch_kernel_on_cluster(
        {grid_dims, block_dims, dim3(2, 1, 1), smem_size, stream},
        kernel_ptr, kernel_params);
  } else {
    status = cutlass::kernel_launch<AttnKernel>(
        grid_dims, block_dims, smem_size, stream, kernel_params, false);
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "attention SM90 BWD mainloop launch failed: ",
              cutlass::cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if constexpr (kHeadPairPrivateKVGrad) {
    constexpr int kElementsPerVector = 8;
    static_assert(kHeadDim % kElementsPerVector == 0 &&
                      kHeadDimV % kElementsPerVector == 0,
                  "secondary K/V accumulation requires 128-bit head vectors");
    constexpr int kMergeThreads = 256;
    int64_t const dk_numel = int64_t(params.b) * params.seqlen_k *
                             params.h_k * kHeadDim;
    int64_t const dv_numel = int64_t(params.b) * params.seqlen_k *
                             params.h_k * kHeadDimV;
    int64_t const dk_vectors = dk_numel / kElementsPerVector;
    int64_t const dv_vectors = dv_numel / kElementsPerVector;
    int64_t const merge_vectors =
        dk_vectors > dv_vectors ? dk_vectors : dv_vectors;
    AttentionAccumulateSecondaryKVGradsKernel<Element>
        <<<cutlass::ceil_div(merge_vectors, int64_t(kMergeThreads)),
           kMergeThreads, 0, stream>>>(
            dk_vectors, dv_vectors, params.seqlen_k, params.h_k,
            kHeadDim / kElementsPerVector,
            reinterpret_cast<Element const*>(secondary_dk.data_ptr()),
            static_cast<Element*>(params.dk_ptr), params.dk_row_stride,
            params.dk_head_stride, params.dk_batch_stride,
            kHeadDimV / kElementsPerVector,
            reinterpret_cast<Element const*>(secondary_dv.data_ptr()),
            static_cast<Element*>(params.dv_ptr), params.dv_row_stride,
            params.dv_head_stride, params.dv_batch_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if constexpr (kGQA && !kReaderPairKVReuse) {
    constexpr int kCastThreads = 256;
    int64_t const dk_numel = dk_accum.numel();
    int64_t const dv_numel = dv_accum.numel();
    int64_t const dkv_numel = dk_numel > dv_numel ? dk_numel : dv_numel;
    auto launch_cast = [&] {
      if constexpr (
          Visibility::kKind ==
              attention::semantics::VisibilityKind::kCausalFull &&
          !kVarlen && !kHasSegment && kHeadDim == kHeadDimV &&
          (kHeadDim == 64 || kHeadDim == 128 || kHeadDim == 192 ||
           kHeadDim == 256)) {
        if ((params.d != kHeadDim || params.dv != kHeadDimV) &&
            params.d <= kHeadDim && params.dv <= kHeadDimV &&
            params.dk_head_stride == params.d &&
            params.dv_head_stride == params.dv) {
          AttentionCastTrimGroupedKVGradsKernel<Element, kHeadDim, kHeadDimV>
              <<<cutlass::ceil_div(dkv_numel, int64_t(kCastThreads)),
                 kCastThreads, 0, stream>>>(
                  dk_numel, dk_accum.data_ptr<float>(),
                  static_cast<Element*>(params.dk_ptr), params.d,
                  dv_numel, dv_accum.data_ptr<float>(),
                  static_cast<Element*>(params.dv_ptr), params.dv);
          return;
        }
      }
      AttentionCastGroupedKVGradsKernel<Element>
          <<<cutlass::ceil_div(dkv_numel, int64_t(kCastThreads)), kCastThreads,
             0, stream>>>(dk_numel, dk_accum.data_ptr<float>(),
                          static_cast<Element*>(params.dk_ptr), dv_numel,
                          dv_accum.data_ptr<float>(),
                          static_cast<Element*>(params.dv_ptr));
    };
    launch_cast();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  using PostprocessKernel = flash::AttentionBwdPostprocessDQ<
      TileShapeMK, Element, ElementAccum, ArchTag,
      AttnKernel::CollectiveMainloop::NumMmaThreads,
      typename AttnKernel::CollectiveMainloop::TiledMmadQ,
      AttnKernel::CollectiveMainloop::dQ_swapAB, SplitDQ>;
  typename PostprocessKernel::Arguments postprocess_args{
      static_cast<ElementAccum const*>(params.dq_accum_ptr),
      {seqlen_q_rounded * kHeadDim, params.h, workspace_batch},
      {_1{}, seqlen_q_rounded * kHeadDim,
       params.h * seqlen_q_rounded * kHeadDim},
      static_cast<Element*>(params.dq_ptr),
      {params.seqlen_q, params.d, params.h, params.b},
      {params.dq_row_stride, _1{}, params.dq_head_stride,
       params.dq_batch_stride},
      params.scale_softmax,
      logical_batch,
      num_m_block,
      kVarlen ? params.cu_seqlens_q : nullptr,
      nullptr, dq_slots, dq_slot_stride};
  typename PostprocessKernel::Params postprocess_params =
      PostprocessKernel::to_underlying_arguments(postprocess_args);
  int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
  auto postprocess_kernel = cutlass::device_kernel<PostprocessKernel>;
  if (smem_size_postprocess >= 48 * 1024) {
    TORCH_CHECK(
        smem_size_postprocess <= max_dynamic_smem,
        "attention SM90 BWD postprocess requires too much dynamic shared memory: ",
        smem_size_postprocess, " bytes, device opt-in limit is ",
        max_dynamic_smem, " bytes");
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        postprocess_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_postprocess));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        postprocess_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
  }
  postprocess_kernel<<<auxiliary_grid,
                       PostprocessKernel::MaxThreadsPerBlock,
                       smem_size_postprocess, stream>>>(postprocess_params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


}  // namespace ops
}  // namespace xattn

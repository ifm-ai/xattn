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
#include <cutlass/numeric_conversion.h>
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
static __global__ void HPInit(const float* lse, const float* delta, float* log2lse, float* dpsum, int N, int R, int H) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=H*R)return;
 int h=i/R, r=i%R; log2lse[i]=r<N?lse[h*N+r]*float(M_LOG2E):INFINITY; dpsum[i]=r<N?delta[h*N+r]:0.f;
}
// Fixed-point dQ bounds: max_j |V_j| (row L2 norm) and max |K|. Eight lanes
// share one key row, so each row is read as contiguous 128-byte segments; the
// fixed shuffle tree keeps the bounds (and the dQ grids) run-to-run identical.
template <typename Element, int kMaxChunksK, int kMaxChunksV>
__global__ void __launch_bounds__(256) FixedPointBoundsKernel(
    Element const* k, Element const* v, int64_t kb, int64_t kr, int64_t kh,
    int64_t vb, int64_t vr, int64_t vh, int seqlen_k, int64_t rows, int hk,
    int d, int dv, float* bounds) {
  constexpr int kLanes = 8, kRowsPerWarp = 32 / kLanes;
  // Non-negative floats compare as ints (atomicMax below). Varlen passes all
  // packed key rows with seqlen_k = 0.
  int64_t const total = rows * hk;
  int const lane = threadIdx.x % 32, sub = lane % kLanes;
  int64_t const warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  int64_t const num_warps = int64_t(gridDim.x) * blockDim.x / 32;
  int const ck = d / 8, cv = dv / 8;
  float vmax = 0.f, kmax = 0.f;
  for (int64_t base = warp * kRowsPerWarp; base < total; base += num_warps * kRowsPerWarp) {
    int64_t const w = base + lane / kLanes;
    float sq = 0.f;
    if (w < total) {
      int64_t const row = w / hk; int const head = int(w % hk);
      int64_t const b = seqlen_k > 0 ? row / seqlen_k : 0, j = seqlen_k > 0 ? row % seqlen_k : row;
      uint4 const* kp = reinterpret_cast<uint4 const*>(k + b * kb + j * kr + head * kh);
      uint4 const* vp = reinterpret_cast<uint4 const*>(v + b * vb + j * vr + head * vh);
      uint4 xv[(kMaxChunksV + kLanes - 1) / kLanes], xk[(kMaxChunksK + kLanes - 1) / kLanes];
      #pragma unroll
      for (int i = 0; i < (kMaxChunksV + kLanes - 1) / kLanes; ++i) {
        int const c = sub + i * kLanes;
        xv[i] = c < cv ? vp[c] : make_uint4(0, 0, 0, 0);
      }
      #pragma unroll
      for (int i = 0; i < (kMaxChunksK + kLanes - 1) / kLanes; ++i) {
        int const c = sub + i * kLanes;
        xk[i] = c < ck ? kp[c] : make_uint4(0, 0, 0, 0);
      }
      #pragma unroll
      for (int i = 0; i < (kMaxChunksV + kLanes - 1) / kLanes; ++i) {
        Element const* e = reinterpret_cast<Element const*>(&xv[i]);
        #pragma unroll
        for (int t = 0; t < 8; ++t) { float f = float(e[t]); sq += f * f; }
      }
      #pragma unroll
      for (int i = 0; i < (kMaxChunksK + kLanes - 1) / kLanes; ++i) {
        Element const* e = reinterpret_cast<Element const*>(&xk[i]);
        #pragma unroll
        for (int t = 0; t < 8; ++t) kmax = fmaxf(kmax, fabsf(float(e[t])));
      }
    }
    #pragma unroll
    for (int o = 1; o < kLanes; o *= 2) { sq += __shfl_xor_sync(0xffffffffu, sq, o); }
    vmax = fmaxf(vmax, sqrtf(sq));
  }
  for (int o = 16; o > 0; o /= 2) {
    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
    kmax = fmaxf(kmax, __shfl_xor_sync(0xffffffffu, kmax, o));
  }
  __shared__ float part[2][8];
  int const warp_in_block = threadIdx.x / 32;
  if (lane == 0) { part[0][warp_in_block] = vmax; part[1][warp_in_block] = kmax; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < int(blockDim.x / 32); ++i) { vmax = fmaxf(vmax, part[0][i]); kmax = fmaxf(kmax, part[1][i]); }
    atomicMax(reinterpret_cast<int*>(bounds), __float_as_int(vmax));
    atomicMax(reinterpret_cast<int*>(bounds) + 1, __float_as_int(kmax));
  }
}
// Padded per-row staging planes: raw max, anchored dP, dO scale. Padding
// rows hold (0, 0, 1).
static __global__ void HPInitRows(const float* mass, float* rows, int N, int R, int H) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=H*R)return;
 int h=i/R, r=i%R; int64_t count=int64_t(H)*N, src=int64_t(h)*N+r, plane=int64_t(H)*R;
 rows[i]=r<N?mass[src]:0.f; rows[plane+i]=r<N?mass[count+src]:0.f; rows[2*plane+i]=r<N?mass[2*count+src]:1.f;
 rows[3*plane+i]=r<N?mass[3*count+src]:0.f;
}


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

// Segment backward pre-pass: key interval of each (batch, query) and query
// interval of each (batch, key).
static __global__ void AttentionSegmentBwdIntervalKernel(
    int num_batch, int seqlen_q, int seqlen_k,
    int64_t const* q_segment_idx, int64_t const* k_segment_idx,
    int k_segment_len, int2* query_intervals, int2* key_intervals) {
  int const i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (i < num_batch * seqlen_q) {
    int const bidb = i / seqlen_q;
    query_intervals[i] = attention::partition::segment_row_interval(
        i - bidb * seqlen_q, seqlen_q, q_segment_idx, k_segment_idx,
        k_segment_len, bidb);
  }
  if (i < num_batch * seqlen_k) {
    int const bidb = i / seqlen_k;
    key_intervals[i] = attention::partition::segment_key_interval(
        i - bidb * seqlen_k, seqlen_q, seqlen_k, q_segment_idx,
        k_segment_idx, k_segment_len, bidb);
  }
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

// Grouped dK / dV cast, 8 values per thread: output rows of d / 8 vectors
// (contiguous) from accumulator rows of accum_row_vectors (the instance's head
// dimension, d rounded up). Thread (row, column): row = idx >> row_shift,
// column = idx & (2^row_shift - 1), past a row's vectors idle (no division).
template <typename Element>
__global__ void AttentionCastGroupedKVGradsVectorKernel(
    int64_t rows, int row_shift, const float* __restrict__ dk_input,
    Element* __restrict__ dk_output, int dk_row_vectors, int dk_accum_row_vectors,
    const float* __restrict__ dv_input, Element* __restrict__ dv_output,
    int dv_row_vectors, int dv_accum_row_vectors) {
  using Input = cutlass::AlignedArray<float, 8>;
  using Output = cutlass::AlignedArray<Element, 8>;
  cutlass::NumericArrayConverter<Element, float, 8> convert;
  int64_t const idx =
      int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  int64_t const row = idx >> row_shift;
  int const column = int(idx & ((int64_t(1) << row_shift) - 1));
  if (row >= rows) {
    return;
  }
  Output out;
  if (column < dk_row_vectors) {
    static_cast<cutlass::Array<Element, 8>&>(out) = convert(
        reinterpret_cast<Input const*>(dk_input)[row * dk_accum_row_vectors + column]);
    reinterpret_cast<Output*>(dk_output)[row * dk_row_vectors + column] = out;
  }
  if (column < dv_row_vectors) {
    static_cast<cutlass::Array<Element, 8>&>(out) = convert(
        reinterpret_cast<Input const*>(dv_input)[row * dv_accum_row_vectors + column]);
    reinterpret_cast<Output*>(dv_output)[row * dv_row_vectors + column] = out;
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

// Backward kernels that read negative-query heads (group_softdelta) specialize this.
template <template <class, class, class> class BwdKernel>
struct AttentionBwdKernelUsesNegQReaders : std::false_type {};

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
          bool kHeadPairPrivateKVGrad = false, int kPrecisionLevel = 0>
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
          kHeadPairClusterKVReuse, kHeadPairPrivateKVGrad, kPrecisionLevel>(params, stream);
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
  static constexpr bool kNegQReaders =
      AttentionBwdKernelUsesNegQReaders<BwdKernel>::value;
  static_assert(!kReaderPairKVReuse || (!kVarlen && (!kHasSegment || kNegQReaders)),
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
  // Negative-query readers run per logical head (kGQA, FP32 dK / dV accumulators)
  // or as reader groups (kReaderPairKVReuse without kGQA: one work tile per
  // key tile and KV head over its readers, dK / dV written once).
  static_assert(!kNegQReaders ||
                    (!kDeterministic && !kVarlen && kPrecisionLevel < 2 &&
                     !kHeadPairParallel && kGQA != kReaderPairKVReuse),
                "negative-query reader BWD: non-deterministic dense/segment GQA only");
  static constexpr bool kNegQGroups = kNegQReaders && kReaderPairKVReuse;
  if constexpr ((kReaderPairKVReuse || kHeadPairParallel) && !kNegQReaders) {
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
  static_assert(kPrecisionLevel >= 0 && kPrecisionLevel <= 4);
  static_assert(kPrecisionLevel < 2 || (!kVarlen && !kReaderPairKVReuse && !kHeadPairParallel),
                "advanced precision needs its dedicated dense reader dispatch");
  static_assert(kPrecisionLevel != 4 || kTwoComponentDV,
                "protected precision requires dual P operands");
  using HPOut = std::conditional_t<(kPrecisionLevel >= 2), float, Element>;
  static constexpr bool IsCausal = Visibility::kUseCausalMask;
  static constexpr bool IsLocal = Visibility::kUseLocalMask;
  static constexpr bool HasSoftcap = false;
  static constexpr bool Varlen = kVarlen;
  static constexpr int kMaxHeadDim = kHeadDim > kHeadDimV ? kHeadDim : kHeadDimV;
  // 128-row tiles for narrow Q/K; with V > 64 only dense rows keep them:
  // segment and varlen rows spill registers there.
  static constexpr bool UseLargeBlockM =
      kHeadDim <= 64 && kHeadDimV <= 128 &&
      (kHeadDimV <= 64 || (!kHasSegment && !kVarlen));
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
  // Non-transposed scores with BM=64 cannot place two 64-row MMA atoms
  // along M: that produces a 128-row copy tile over a 64-row allocation.
  // Preserve legacy layouts; advanced BN64 instances split the N dimension.
  static constexpr int AtomLayoutMSdP =
      kPrecisionLevel >= 2 && kBlockN == 64 && SdPSwapAB ? 2 : 1;
  // Varlen mixed-D/V layouts use unsplit K/V slices when either dimension is
  // not warp-group aligned.
  static constexpr bool DKvCanSplitHeadDim =
      kMaxHeadDim > 128 &&
      kHeadDim % (NumMmaWarpGroups * 64) == 0 &&
      kHeadDimV % (NumMmaWarpGroups * 64) == 0;
  static constexpr int AtomLayoutNdKV =
      kPrecisionLevel >= 2 && kBlockN == 64 ? 1 : kMaxHeadDim <= 128
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
  // D/V 128/256 (BN 128) estimates 225 KB; with the MHA epilogue the real
  // shared storage still fits the 227 KB opt-in limit, with the GQA one not.
  static constexpr size_t DenseStage2Limit =
      kHeadDim == 128 && kHeadDimV == 256 && !kGQA ? 225 * 1024 : 224 * 1024;
  static constexpr bool UseDenseStage2 =
      !kDeterministic && !kVarlen && !kHasSegment &&
      Stage2TensorStorageEstimate <= DenseStage2Limit;
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
  // Negative-query readers: the caller prepares dPsum / LSE_log2 / the zeroed dQaccum
  // over logical heads with rows padded to params.seqlen_q_padded.
  const int seqlen_q_rounded = kVarlen || kNegQReaders
      ? params.seqlen_q_padded
      : cute::round_up(params.seqlen_q, kBlockM);
  if constexpr (kNegQReaders) {
    TORCH_CHECK(seqlen_q_rounded % kBlockM == 0 && seqlen_q_rounded >= params.seqlen_q &&
                    params.dpsum_ptr != nullptr && params.lse_log2_ptr != nullptr &&
                    params.dq_accum_ptr != nullptr,
                "negative-query reader BWD needs caller-prepared dPsum, LSE_log2 and dQaccum");
  }
  const int logical_batch = kVarlen ? params.num_sequences : params.b;
  // Varlen workspace batch: 1; launch and scheduling batch: num_sequences.
  const int workspace_batch = kVarlen ? 1 : logical_batch;
  // Low-precision deterministic dQ uses order-free fixed-point sums for
  // D <= 128 and D = V = 192; other dimensions keep ordered or slot-split
  // FP32 accumulation.
  static constexpr bool kFixedDQ =
      kDeterministic && kPrecisionLevel < 2 &&
      (kHeadDim <= 128 || (kHeadDim == 192 && kHeadDimV == 192));
  static constexpr bool SplitDQ =
      kDeterministic && !kFixedDQ && Visibility::kIsSlidingChunk &&
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
      {kNegQReaders ? 0 : workspace_batch, params.h, seqlen_q_rounded},
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
      : torch::empty({kNegQReaders ? 0 : dq_slot_stride}, dq_options);
  const int num_m_block = cute::ceil_div(
      kVarlen ? params.max_seqlen_q : params.seqlen_q, kBlockM);
  // Any batch: the preprocess and postprocess kernels stride over batches
  // past the grid z limit.
  TORCH_CHECK(params.h <= 65535, "attention SM90 BWD supports at most 65535 query heads");
  dim3 const auxiliary_grid(num_m_block, params.h,
                            logical_batch < 65535 ? logical_batch : 65535);
  torch::Tensor dq_semaphore;
  torch::Tensor hp_rows;
  torch::Tensor fixed_bounds, dq_exponent;
  if constexpr (kFixedDQ) {
    fixed_bounds = torch::zeros({2}, dq_options);
    dq_exponent = torch::empty({workspace_batch, params.h, seqlen_q_rounded},
                               dq_options.dtype(at::kInt));
    // Varlen scans every packed key row (params.seqlen_k = k.size(0));
    // row-aligned SCA varlen metadata has a different number of key runs
    // than queries.
    const int64_t key_rows = kVarlen ? int64_t(params.seqlen_k)
                                     : int64_t(params.b) * params.seqlen_k;
    const int64_t warps = key_rows * params.h_k;
    if (warps > 0) {
      FixedPointBoundsKernel<Element, kHeadDim / 8, kHeadDimV / 8><<<int(std::min<int64_t>((warps + 31) / 32, int64_t(params.num_sm) * 8)), 256, 0, stream>>>(
          static_cast<Element const*>(params.k_ptr), static_cast<Element const*>(params.v_ptr),
          params.k_batch_stride, params.k_row_stride, params.k_head_stride,
          params.v_batch_stride, params.v_row_stride, params.v_head_stride,
          kVarlen ? 0 : params.seqlen_k, key_rows, params.h_k, params.d, params.dv,
          fixed_bounds.data_ptr<float>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  if constexpr (!kNegQReaders) {
    params.dpsum_ptr = dpsum.data_ptr<float>();
    params.lse_log2_ptr = lse_log2.data_ptr<float>();
    params.dq_accum_ptr = dq_accum.data_ptr<float>();
  }
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

  if constexpr (kPrecisionLevel >= 2) {
  // Advanced preprocessing does not run the normal semaphore initializer.
  // Keep LP initialization in its existing fused preprocess kernel.
  if constexpr (kDeterministic) {
    C10_CUDA_CHECK(cudaMemsetAsync(params.dq_semaphore, 0,
                                  dq_semaphore.numel() * sizeof(int), stream));
  }
  HPInit<<<(workspace_batch*params.h*seqlen_q_rounded+255)/256,256,0,stream>>>(static_cast<float const*>(params.lse_ptr),params.hp_delta,static_cast<float*>(params.lse_log2_ptr),static_cast<float*>(params.dpsum_ptr),params.seqlen_q,seqlen_q_rounded,workspace_batch*params.h);
  hp_rows = torch::empty({4, workspace_batch, params.h, seqlen_q_rounded}, dpsum.options());
  HPInitRows<<<(workspace_batch*params.h*seqlen_q_rounded+255)/256,256,0,stream>>>(params.hp_mass,hp_rows.data_ptr<float>(),params.seqlen_q,seqlen_q_rounded,workspace_batch*params.h);
  C10_CUDA_CHECK(cudaMemsetAsync(params.dq_accum_ptr,0,dq_accum.numel()*sizeof(float),stream));
  } else {
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
        {_1{}, seqlen_q_rounded, int64_t(params.h) * seqlen_q_rounded},
        static_cast<float const*>(params.lse_ptr),
        {_1{}, params.seqlen_q, int64_t(params.h) * params.seqlen_q},
        static_cast<float*>(params.lse_log2_ptr),
        {_1{}, seqlen_q_rounded, int64_t(params.h) * seqlen_q_rounded},
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {int64_t(seqlen_q_rounded) * kHeadDim, params.h, workspace_batch},
        {_1{}, int64_t(seqlen_q_rounded) * kHeadDim,
         int64_t(params.h) * seqlen_q_rounded * kHeadDim},
        logical_batch,
        num_m_block,
        params.dq_semaphore,
        kVarlen ? params.cu_seqlens_q : nullptr,
        nullptr,
        OutputStateFp32 ? static_cast<float const*>(params.y_ptr)
                        : nullptr,
        kFixedDQ ? fixed_bounds.data_ptr<float>() : nullptr,
        kFixedDQ ? dq_exponent.data_ptr<int>() : nullptr};
    typename PreprocessKernel::Params preprocess_params =
        PreprocessKernel::to_underlying_arguments(preprocess_args);
    return cutlass::kernel_launch<PreprocessKernel>(
        auxiliary_grid,
        PreprocessKernel::MaxThreadsPerBlock,
        PreprocessKernel::SharedStorageSize, stream, preprocess_params,
        false);
  };
  cutlass::Status preprocess_status = cutlass::Status::kSuccess;
  if constexpr (!kNegQReaders) {
    preprocess_status = params.y_is_fp32
        ? launch_preprocess(std::true_type{})
        : launch_preprocess(std::false_type{});
  }
  TORCH_CHECK(preprocess_status == cutlass::Status::kSuccess,
              "attention SM90 BWD preprocess launch failed: ",
              cutlass::cutlassGetStatusString(preprocess_status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  using TileShapeMNK =
      cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using ClusterShape = cute::Shape<
      Int<kHeadPairClusterKVReuse ? 2 : 1>, _1, _1>;
  // Fixed-point dQ stages its row exponents in shared memory when the whole
  // kernel storage still fits; the staged variant also sizes the epilogue and
  // scheduler below (both independent of the staging).
  using CollectiveMainloopStaged = flash::CollectiveMainloopBwdSm90<
      Stages, StagesDO, StagesDS, ClusterShape, TileShapeMNK, kHeadDimV,
      Element, ElementAccum, ArchTag, IsCausal, IsLocal, HasSoftcap, Varlen,
      kDeterministic, kHasSegment, kDirectChunkRange,
      SdPSwapAB, DKvSwapAB,
      DqSwapAB,
      NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
      MmaDPIsRS, kTwoComponentDV, kReaderPairKVReuse, kDirectChunkTile,
      kHeadPairParallel, kHeadPairClusterKVReuse,
      kHeadPairPrivateKVGrad, SplitDQ, kPrecisionLevel, true, kNegQReaders>;
  using CollectiveMainloopUnstaged = flash::CollectiveMainloopBwdSm90<
      Stages, StagesDO, StagesDS, ClusterShape, TileShapeMNK, kHeadDimV,
      Element, ElementAccum, ArchTag, IsCausal, IsLocal, HasSoftcap, Varlen,
      kDeterministic, kHasSegment, kDirectChunkRange,
      SdPSwapAB, DKvSwapAB,
      DqSwapAB,
      NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
      MmaDPIsRS, kTwoComponentDV, kReaderPairKVReuse, kDirectChunkTile,
      kHeadPairParallel, kHeadPairClusterKVReuse,
      kHeadPairPrivateKVGrad, SplitDQ, kPrecisionLevel, false, kNegQReaders>;
  using CollectiveEpilogueDirect = flash::CollectiveEpilogueBwd<
      TileShapeMNK, kHeadDimV, HPOut, ArchTag,
      CollectiveMainloopStaged::NumMmaThreads, Varlen, DKvSwapAB,
      NumMmaWarpGroups / AtomLayoutNdKV>;
  using CollectiveEpilogueSplitDestinations =
      flash::CollectiveEpilogueBwdSplitBatchDestinations<
          TileShapeMNK, kHeadDimV, Element, ArchTag,
          CollectiveMainloopStaged::NumMmaThreads, Varlen, DKvSwapAB,
          NumMmaWarpGroups / AtomLayoutNdKV>;
  using CollectiveEpilogueMHA = std::conditional_t<
      kHeadPairPrivateKVGrad, CollectiveEpilogueSplitDestinations,
      CollectiveEpilogueDirect>;
  using CollectiveEpilogueGQA = flash::CollectiveEpilogueBwdGQA<
      TileShapeMNK, kHeadDimV, ElementAccum, ArchTag,
      CollectiveMainloopStaged::NumMmaThreads, Varlen, kDeterministic, DKvSwapAB>;
  using CollectiveEpilogueNegQGroup = flash::CollectiveEpilogueBwdNegQGroup<
      TileShapeMNK, kHeadDimV, Element, ArchTag,
      CollectiveMainloopStaged::NumMmaThreads, DKvSwapAB,
      NumMmaWarpGroups / AtomLayoutNdKV>;
  using CollectiveEpilogue = std::conditional_t<
      kNegQGroups, CollectiveEpilogueNegQGroup,
      std::conditional_t<
          kGQA && !kReaderPairKVReuse && !kHeadPairPrivateKVGrad,
          CollectiveEpilogueGQA, CollectiveEpilogueMHA>>;
  using DenseScheduler =
      flash::SingleTileScheduler<Varlen, false /*Split*/, false /*PackGQA*/,
                                 kBlockN>;
  using DenseLPTScheduler =
      flash::SingleTileBwdLPTScheduler<
          false, kBlockN, false /*SPT*/,
          kHeadPairParallel ? 2 : 1,
          !kHeadPairParallel ||
              kHeadPairClusterKVReuse /*ContiguousWorkGroup*/,
          kNegQReaders /*ResidualFirst*/>;
  using CausalDeterministicScheduler =
      flash::SingleTileBwdLPTScheduler<false, kBlockN, true /*SPT*/>;
  static constexpr int VarlenSchedulerProducerThreads =
      cutlass::NumThreadsPerWarp *
      (CollectiveMainloopStaged::dQacc_use_TMA ? 2 : 1);
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
          CollectiveMainloopStaged::NumMmaThreads,
          VarlenSchedulerProducerThreads>;
  // Fixed-point dQ needs no ordering; grouped heads keep ordered dK/dV
  // semaphores and therefore the deterministic varlen/segment schedulers.
  static constexpr bool UseDeterministicSchedule =
      kDeterministic && !(kFixedDQ && (!kGQA || (!kVarlen && !kHasSegment)));
  static constexpr bool UseVarlenPersistentScheduler =
      kVarlen && UseDeterministicSchedule;
  using VarlenPersistentScheduler =
      flash::VarlenSequencePersistentTileScheduler<
          kBlockN, CollectiveMainloopStaged::NumMmaThreads,
          VarlenSchedulerProducerThreads, true /*WaitForWorkCompletion*/>;
  using SegmentScheduler = flash::StaticSegmentBwdTileScheduler;
  using DeterministicSegmentScheduler =
      flash::DeterministicSegmentBwdTileScheduler;
  using Scheduler =
      std::conditional_t<
          kHasSegment,
          std::conditional_t<UseDeterministicSchedule,
                             DeterministicSegmentScheduler,
                             SegmentScheduler>,
                         std::conditional_t<
                             UseVarlenPersistentScheduler,
                             VarlenPersistentScheduler,
                             std::conditional_t<
                                 UseDeterministicSchedule && !kVarlen && !kHasSegment,
                                 std::conditional_t<
                                     UseDensePersistentScheduler,
                                     DeterministicDensePersistentScheduler,
                                     DeterministicDenseScheduler>,
                                 std::conditional_t<kVarlen, DenseScheduler,
                                                    DenseLPTScheduler>>>>;
  // SM90 opt-in shared memory per block: 227 KB.
  static constexpr bool kStageDqExpFits =
      flash::enable_sm90<BwdKernel<CollectiveMainloopStaged, CollectiveEpilogue,
                                   Scheduler>>::SharedStorageSize <= 232448;
  using CollectiveMainloop = std::conditional_t<
      !kFixedDQ || kStageDqExpFits, CollectiveMainloopStaged,
      CollectiveMainloopUnstaged>;
  using AttnKernel = flash::enable_sm90<BwdKernel<
      CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

  torch::Tensor segment_intervals;
  int2 const* segment_query_intervals = nullptr;
  int2 const* segment_key_intervals = nullptr;
  if constexpr (CollectiveMainloop::SegmentIntervals) {
    TORCH_CHECK(params.q_segment_idx != nullptr && params.k_segment_idx != nullptr &&
                    params.k_segment_len > 0,
                "attention SM90 BWD segment masks need q/k segment ids");
    int64_t const q_count = int64_t(params.b) * params.seqlen_q;
    int64_t const k_count = int64_t(params.b) * params.seqlen_k;
    segment_intervals = torch::empty(
        {q_count + k_count, 2},
        torch::TensorOptions().device(torch::kCUDA).dtype(at::kInt));
    int2* base = reinterpret_cast<int2*>(segment_intervals.data_ptr<int>());
    int64_t const count = std::max(q_count, k_count);
    if (count > 0) {
      AttentionSegmentBwdIntervalKernel<<<int(cutlass::ceil_div(count, int64_t(256))), 256, 0, stream>>>(
          params.b, params.seqlen_q, params.seqlen_k, params.q_segment_idx,
          params.k_segment_idx, params.k_segment_len, base, base + q_count);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    segment_query_intervals = base;
    segment_key_intervals = base + q_count;
  }
  torch::Tensor deterministic_segment_ranges;
  int const* deterministic_segment_ranges_ptr = nullptr;
  if constexpr (kHasSegment && kDeterministic && !kFixedDQ) {
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
      {int64_t(seqlen_q_rounded) * kHeadDim, params.h, workspace_batch},
      {_1{}, int64_t(seqlen_q_rounded) * kHeadDim,
       int64_t(params.h) * seqlen_q_rounded * kHeadDim},
      static_cast<float const*>(params.lse_log2_ptr),
      {seqlen_q_rounded, params.h, workspace_batch},
      {_1{}, seqlen_q_rounded, int64_t(params.h) * seqlen_q_rounded},
      static_cast<float const*>(params.dpsum_ptr),
      {_1{}, seqlen_q_rounded, int64_t(params.h) * seqlen_q_rounded},
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
      dq_slots, dq_slot_stride, params.hp_rho, params.hp_mass,
      hp_rows.defined() ? hp_rows.data_ptr<float>() : nullptr,
      hp_rows.defined() ? hp_rows.data_ptr<float>() + hp_rows.numel() / 4 : nullptr,
      hp_rows.defined() ? hp_rows.data_ptr<float>() + 2 * (hp_rows.numel() / 4) : nullptr,
      hp_rows.defined() ? hp_rows.data_ptr<float>() + 3 * (hp_rows.numel() / 4) : nullptr,
      params.hp_rho_exponent,
      kFixedDQ ? dq_exponent.data_ptr<int>() : nullptr,
      segment_query_intervals, segment_key_intervals,
      kNegQReaders ? static_cast<Element const*>(params.neg_q_ptr) : nullptr,
      {params.neg_q_row_stride, _1{}, params.neg_q_head_stride, params.neg_q_batch_stride},
      kNegQReaders ? static_cast<Element const*>(params.neg_dy_ptr) : nullptr,
      {params.neg_dy_row_stride, _1{}, params.neg_dy_head_stride,
       params.neg_dy_batch_stride}};

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
  } else if constexpr (kNegQGroups) {
    // Folded negative-query gradient: dK rows in FP32 for the dQ postprocess.
    if (params.fold_neg_q) {
      int64_t const dk_numel = int64_t(params.b) * params.seqlen_k * params.h_k * kHeadDim;
      dk_accum = (params.dkv_accum_ptr != nullptr
          ? torch::from_blob(params.dkv_accum_ptr, {dk_numel},
                             torch::TensorOptions().device(torch::kCUDA).dtype(at::kFloat))
          : torch::empty({dk_numel}, torch::TensorOptions().device(torch::kCUDA).dtype(at::kFloat)))
          .view({params.b, params.seqlen_k, params.h_k, kHeadDim});
    }
  } else if constexpr (kGQA && !kReaderPairKVReuse) {
    auto accum_options = torch::TensorOptions()
                             .device(torch::kCUDA)
                             .dtype(at::kFloat)
                             .memory_format(at::MemoryFormat::Contiguous);
    int64_t const dk_numel = int64_t(params.b) * params.seqlen_k *
                             params.h_k * kHeadDim;
    int64_t const dv_numel = int64_t(params.b) * params.seqlen_k *
                             params.h_k * kHeadDimV;
    dkv_accum = params.dkv_accum_ptr != nullptr
        ? torch::from_blob(params.dkv_accum_ptr, {dk_numel + dv_numel}, accum_options)
        : torch::zeros({dk_numel + dv_numel}, accum_options);
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
              static_cast<HPOut*>(params.dk_ptr),
              {params.seqlen_k, kHeadDim, params.h_k, params.b},
              {params.dk_row_stride, _1{}, params.dk_head_stride,
               params.dk_batch_stride},
              static_cast<HPOut*>(params.dv_ptr),
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
    } else if constexpr (kNegQGroups) {
      return typename CollectiveEpilogue::Arguments{
          {static_cast<HPOut*>(params.dk_ptr),
           {params.seqlen_k, params.d, params.h_k, params.b},
           {params.dk_row_stride, _1{}, params.dk_head_stride,
            params.dk_batch_stride},
           static_cast<HPOut*>(params.dv_ptr),
           {params.seqlen_k, params.dv, params.h_k, params.b},
           {params.dv_row_stride, _1{}, params.dv_head_stride,
            params.dv_batch_stride},
           params.b,
           scheduled_q_heads,
           nullptr,
           nullptr,
           nullptr,
           nullptr},
          params.fold_neg_q ? dk_accum.data_ptr<float>() : nullptr,
          {params.seqlen_k, kHeadDim, params.h_k, params.b},
          {int64_t(params.h_k) * kHeadDim, _1{}, int64_t(kHeadDim),
           int64_t(params.seqlen_k) * params.h_k * kHeadDim}};
    } else {
      int const output_heads = kReaderPairKVReuse ? params.h_k : params.h;
      return typename CollectiveEpilogue::Arguments{
          static_cast<HPOut*>(params.dk_ptr),
          {params.seqlen_k, params.d, output_heads, params.b},
          {params.dk_row_stride, _1{}, params.dk_head_stride,
           params.dk_batch_stride},
          static_cast<HPOut*>(params.dv_ptr),
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
    int const max_segment_work = num_blocks_n * scheduled_q_heads * params.b;
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
            params.seqlen_q, params.seqlen_k, scheduled_q_heads, params.b,
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
            static_cast<HPOut*>(params.dk_ptr), params.dk_row_stride,
            params.dk_head_stride, params.dk_batch_stride,
            kHeadDimV / kElementsPerVector,
            reinterpret_cast<Element const*>(secondary_dv.data_ptr()),
            static_cast<HPOut*>(params.dv_ptr), params.dv_row_stride,
            params.dv_head_stride, params.dv_batch_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if constexpr (kGQA && !kReaderPairKVReuse && !kNegQReaders) {
    constexpr int kCastThreads = 256;
    int64_t const dk_numel = dk_accum.numel();
    int64_t const dv_numel = dv_accum.numel();
    int64_t const dkv_numel = dk_numel > dv_numel ? dk_numel : dv_numel;
    auto launch_cast = [&] {
      // Dense [rows, d] / [rows, dv] outputs: 8-value vectors (one 128-bit
      // or wider load and store per thread) instead of one value per thread.
      if constexpr (kHeadDim % 8 == 0 && kHeadDimV % 8 == 0) {
        using CastOutput = cutlass::AlignedArray<HPOut, 8>;
        int64_t const rows = dk_numel / kHeadDim;
        auto aligned = [](void const* ptr) {
          return reinterpret_cast<uintptr_t>(ptr) % alignof(CastOutput) == 0;
        };
        int const dk_row_vectors = params.d / 8;
        int const dv_row_vectors = params.dv / 8;
        int row_shift = 0;
        while ((1 << row_shift) < (dk_row_vectors > dv_row_vectors ? dk_row_vectors : dv_row_vectors)) {
          ++row_shift;
        }
        int64_t const cast_blocks = cutlass::ceil_div(rows << row_shift, int64_t(kCastThreads));
        if (params.d % 8 == 0 && params.dv % 8 == 0 && params.d <= kHeadDim &&
            params.dv <= kHeadDimV && dv_numel / kHeadDimV == rows &&
            params.dk_head_stride == params.d && params.dv_head_stride == params.dv &&
            aligned(params.dk_ptr) && aligned(params.dv_ptr) &&
            cast_blocks <= std::numeric_limits<int>::max()) {
          AttentionCastGroupedKVGradsVectorKernel<HPOut>
              <<<uint32_t(cast_blocks), kCastThreads, 0, stream>>>(
                  rows, row_shift, dk_accum.data_ptr<float>(),
                  static_cast<HPOut*>(params.dk_ptr), dk_row_vectors, kHeadDim / 8,
                  dv_accum.data_ptr<float>(), static_cast<HPOut*>(params.dv_ptr),
                  dv_row_vectors, kHeadDimV / 8);
          return;
        }
      }
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
          AttentionCastTrimGroupedKVGradsKernel<HPOut, kHeadDim, kHeadDimV>
              <<<cutlass::ceil_div(dkv_numel, int64_t(kCastThreads)),
                 kCastThreads, 0, stream>>>(
                  dk_numel, dk_accum.data_ptr<float>(),
                  static_cast<HPOut*>(params.dk_ptr), params.d,
                  dv_numel, dv_accum.data_ptr<float>(),
                  static_cast<HPOut*>(params.dv_ptr), params.dv);
          return;
        }
      }
      AttentionCastGroupedKVGradsKernel<HPOut>
          <<<cutlass::ceil_div(dkv_numel, int64_t(kCastThreads)), kCastThreads,
             0, stream>>>(dk_numel, dk_accum.data_ptr<float>(),
                          static_cast<HPOut*>(params.dk_ptr), dv_numel,
                          dv_accum.data_ptr<float>(),
                          static_cast<HPOut*>(params.dv_ptr));
    };
    launch_cast();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  using PostprocessKernel = flash::AttentionBwdPostprocessDQ<
      TileShapeMK, HPOut, ElementAccum, ArchTag,
      AttnKernel::CollectiveMainloop::NumMmaThreads,
      typename AttnKernel::CollectiveMainloop::TiledMmadQ,
      AttnKernel::CollectiveMainloop::dQ_swapAB, SplitDQ, kNegQReaders>;
  typename PostprocessKernel::Arguments postprocess_args{
      static_cast<ElementAccum const*>(params.dq_accum_ptr),
      {int64_t(seqlen_q_rounded) * kHeadDim, params.h, workspace_batch},
      {_1{}, int64_t(seqlen_q_rounded) * kHeadDim,
       int64_t(params.h) * seqlen_q_rounded * kHeadDim},
      static_cast<HPOut*>(params.dq_ptr),
      {params.seqlen_q, params.d, params.h, params.b},
      {params.dq_row_stride, _1{}, params.dq_head_stride,
       params.dq_batch_stride},
      params.scale_softmax,
      logical_batch,
      num_m_block,
      kVarlen ? params.cu_seqlens_q : nullptr,
      nullptr, dq_slots, dq_slot_stride,
      kFixedDQ ? dq_exponent.data_ptr<int>() : nullptr};
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
  if constexpr (kNegQReaders) {
    // One launch over logical heads: dq and dneg_q; folded, the negative-query heads
    // write dk = round(dK accumulator + negative-query gradient) and the cast below
    // covers dV only.
    int const readers = params.h / params.h_k;
    int64_t const dq_accum_head_stride = int64_t(seqlen_q_rounded) * kHeadDim;
    typename PostprocessKernel::Arguments reader_args{
        static_cast<ElementAccum const*>(params.dq_accum_ptr),
        {int64_t(seqlen_q_rounded) * kHeadDim, params.h, workspace_batch},
        {_1{}, dq_accum_head_stride, params.h * dq_accum_head_stride},
        static_cast<HPOut*>(params.dq_ptr),
        {params.seqlen_q, params.d, params.h - params.h_k, params.b},
        {params.dq_row_stride, _1{}, params.dq_head_stride,
         params.dq_batch_stride},
        params.scale_softmax,
        logical_batch,
        num_m_block,
        nullptr,
        nullptr, dq_slots, dq_slot_stride,
        nullptr,
        {static_cast<HPOut*>(params.fold_neg_q ? params.dk_ptr : params.dneg_q_ptr),
         params.fold_neg_q
             ? typename PostprocessKernel::StridedQ{params.dk_row_stride, _1{},
                                                    params.dk_head_stride,
                                                    params.dk_batch_stride}
             : typename PostprocessKernel::StridedQ{params.dneg_q_row_stride, _1{},
                                                    params.dneg_q_head_stride,
                                                    params.dneg_q_batch_stride},
         params.fold_neg_q ? dk_accum.data_ptr<float>() : nullptr,
         {int64_t(params.h_k) * kHeadDim, _1{}, int64_t(kHeadDim),
          int64_t(params.seqlen_k) * params.h_k * kHeadDim},
         readers}};
    postprocess_kernel<<<auxiliary_grid,
                         PostprocessKernel::MaxThreadsPerBlock,
                         smem_size_postprocess, stream>>>(
        PostprocessKernel::to_underlying_arguments(reader_args));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if constexpr (kNegQGroups) {
      return;  // dV and an unfolded dK were written by the epilogue
    }
    static_assert(kHeadDim % 8 == 0 && kHeadDimV % 8 == 0,
                  "negative-query reader dK / dV cast uses 8-value vectors");
    TORCH_CHECK(params.d % 8 == 0 && params.dv % 8 == 0 &&
                    params.dk_head_stride == params.d && params.dv_head_stride == params.dv,
                "negative-query reader BWD needs D, V multiples of 8 and contiguous dK / dV");
    // Accumulator rows hold the instance's head dimensions; the outputs the actual ones.
    int64_t const kv_rows = int64_t(params.b) * params.seqlen_k * params.h_k;
    int const dk_row_vectors = params.fold_neg_q ? 0 : params.d / 8;
    int const dv_row_vectors = params.dv / 8;
    int row_shift = 0;
    while ((1 << row_shift) < (dk_row_vectors > dv_row_vectors ? dk_row_vectors : dv_row_vectors)) {
      ++row_shift;
    }
    int64_t const cast_threads = kv_rows << row_shift;
    TORCH_CHECK(cutlass::ceil_div(cast_threads, int64_t(256)) <= std::numeric_limits<int>::max(),
                "negative-query reader dK / dV cast grid exceeds the CUDA x-dimension limit");
    AttentionCastGroupedKVGradsVectorKernel<HPOut>
        <<<uint32_t(cutlass::ceil_div(cast_threads, int64_t(256))), 256, 0, stream>>>(
            kv_rows, row_shift, dk_accum.data_ptr<float>(), static_cast<HPOut*>(params.dk_ptr),
            dk_row_vectors, kHeadDim / 8, dv_accum.data_ptr<float>(),
            static_cast<HPOut*>(params.dv_ptr), dv_row_vectors, kHeadDimV / 8);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  postprocess_kernel<<<auxiliary_grid,
                       PostprocessKernel::MaxThreadsPerBlock,
                       smem_size_postprocess, stream>>>(postprocess_params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


}  // namespace ops
}  // namespace xattn

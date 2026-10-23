// Author: Shicheng Wen

#include "group_softdelta/preprocess.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/fast_math.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace xattn {
namespace ops {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ float Sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

// Eight elements kept in their storage type until use (16-bit: one uint4).
template <typename T>
struct Packed8 {
  static constexpr int kWords = sizeof(T) * 8 / sizeof(uint4);
  uint4 words[kWords];

  __device__ __forceinline__ void load(const T* pointer) {
#pragma unroll
    for (int i = 0; i < kWords; ++i) words[i] = reinterpret_cast<const uint4*>(pointer)[i];
  }
  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int i = 0; i < kWords; ++i) words[i] = make_uint4(0, 0, 0, 0);
  }
  __device__ __forceinline__ float operator[](int i) const {
    return static_cast<float>(reinterpret_cast<const T*>(words)[i]);
  }
};

// One item per (batch, row, KV head g): kLanes x kSlices lanes. Lane l owns
// columns [8 (l % kLanes), +8) of V (kLanes = V / 8 rounded up to a power of
// two; lanes past V are idle) and slice l / kLanes the slice's share of the
// group's R - 1 query reads, in chunks of kChunk (registers stay at the
// four-read level for any group size). Forward state: the negative read neg
// ([B, L, Hkv, V] rows of neg_read_row_stride) and the query reads a (or the gated
// output o = a - s neg with gated_output; [B, L, H, V] rows of read_row_stride), with
// their LSEs [B, Hkv, L] / [B, H, L] (batch strides given). Logical heads
// (dPsum, LSE_log2) are g R + r with r = 0 the negative read. One streaming pass:
//   neg_dy       = -sum_r s_r dO_r                     (rounded once)
//   dg_r         = -s_r (1 - s_r) <dO_r, neg>          (per value group)
//   dPsum head r = <dO_r, a_r>,  head 0 = <neg_dy, neg> (0 on padded rows)
//   LSE_log2     = LSE log2(e), 0 for -inf             (+inf on padded rows)
//   dQaccum      = 0 (the item's share: D floats of each reader's head)
//   dK / dV accumulator rows (b, t, kv) = 0 (rows t < L)
// Loads are issued before their uses; item decoding is 32-bit.
constexpr int kChunk = 4;

template <int kWidth>
__device__ __forceinline__ float ButterflySum(float value) {
#pragma unroll
  for (int offset = 1; offset < kWidth; offset <<= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

// Sum over the value group's lanes (group_lanes contiguous lanes from
// group_base; any count): every lane gets its group's sum.
__device__ __forceinline__ float SegmentSum(float value, int group_base, int group_lanes) {
  float sum = 0.0f;
  for (int j = 0; j < group_lanes; ++j) {
    sum += __shfl_sync(0xffffffff, value, group_base + j);
  }
  return sum;
}

// kWholeGroup: one value group spanning the row (the gate gradient's group sum
// is the row sum); otherwise value groups of group_lanes lanes. kLooped: more
// than kChunk reads per slice, processed in chunks (only for the widest slice
// layout); otherwise one straight-line round, every load issued before any use.
template <typename T, typename StateT, int kLanes, int kSlices, bool kWholeGroup, bool kLooped>
__global__ void __launch_bounds__(kWarpSize * kWarpsPerBlock) GroupSoftDeltaBwdPreprocessKernel(
    const T* __restrict__ dy, const T* __restrict__ gate, const StateT* __restrict__ neg_read_state,
    int64_t neg_read_row_stride, const StateT* __restrict__ read_state, int64_t read_row_stride,
    const float* __restrict__ neg_read_lse, int64_t neg_read_lse_batch_stride, const float* __restrict__ read_lse,
    int64_t read_lse_batch_stride, T* __restrict__ neg_dy, T* __restrict__ gate_grad,
    float* __restrict__ dpsum, float* __restrict__ lse_log2, float4* __restrict__ dq_accum,
    int dq_row_vectors, float4* __restrict__ dk_accum, int dk_row_vectors, float4* __restrict__ dv_accum,
    int dv_row_vectors, int items, cutlass::FastDivmod kv_divmod, cutlass::FastDivmod padded_divmod,
    int length, int rep, int value_dim, int groups, int group_lanes, bool gated_output) {
  static_assert(kLanes * kSlices <= kWarpSize);
  constexpr int kItemLanes = kLanes * kSlices;
  const int lane = threadIdx.x % kWarpSize;
  const int item_lane = lane % kItemLanes;
  const int sub = item_lane % kLanes;
  const int slice = item_lane / kLanes;
  const int item = int((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / kItemLanes);
  // Subgroups past the end still join the shuffles of their warp.
  const bool live = item < items;
  int kv, t;
  const int bt = kv_divmod.divmod(kv, live ? item : 0);
  const int b = padded_divmod.divmod(t, bt);
  const int kv_heads = kv_divmod.divisor;
  const int padded_length = padded_divmod.divisor;
  const int readers = rep + 1;
  const int logical_heads = kv_heads * readers;
  const int q_heads = kv_heads * rep;
  const bool active_row = live && t < length;
  const int col = sub * 8;
  const bool active = active_row && col < value_dim;
  const int group = kWholeGroup ? 0 : (col < value_dim ? col / (group_lanes * 8) : 0);
  const int64_t row = int64_t(b) * length + (active_row ? t : 0);
  const int64_t stat_base = (int64_t(b) * logical_heads + kv * readers) * padded_length + t;
  // First and unconditional, from an in-bounds address (otherwise the compiler
  // sinks it behind the other loads' uses); reads past the item's lanes (groups
  // above kItemLanes - 1 query heads) load in a loop at the end.
  const int lse_reader = item_lane < readers ? item_lane : 0;
  const float* lse_row = lse_reader == 0
      ? neg_read_lse + b * neg_read_lse_batch_stride + int64_t(kv) * length
      : read_lse + b * read_lse_batch_stride + int64_t(kv * rep + lse_reader - 1) * length;
  const float lse_value = lse_row[active_row ? t : 0];
  // This slice's reads [first, first + count); per_slice is the same for every slice.
  const int per_slice = (rep + kSlices - 1) / kSlices;
  const int first = slice * per_slice;
  const int count = max(0, min(per_slice, rep - first));
  // Loads are unconditional from in-bounds addresses (column, row and read
  // clamped) and masked at use: guarded loads become branches that the
  // compiler interleaves with their uses, serializing the memory round trips.
  const int load_first = first < rep ? first : 0;
  const int load_col = col < value_dim ? col : 0;
  const int64_t head_row = row * q_heads + kv * rep + first;
  const int64_t load_head_row = row * q_heads + kv * rep + load_first;
  const T* grad_base = dy + load_head_row * value_dim + load_col;
  const T* gate_base = gate + load_head_row * groups + group;
  const StateT* read_base = read_state + row * read_row_stride + (kv * rep + load_first) * value_dim + load_col;
  const int load_count = max(1, min(per_slice, rep - load_first));
  // Packed until use: a conversion here would wait for neg before the other loads issue.
  Packed8<StateT> neg;
  neg.load(neg_read_state + row * neg_read_row_stride + kv * value_dim + load_col);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const int rounds = kLooped ? (per_slice + kChunk - 1) / kChunk : 1;
  for (int round = 0; round < rounds; ++round) {
    const int base = round * kChunk;
    Packed8<T> grads[kChunk];
    Packed8<StateT> reads[kChunk];
    T gate_logits[kChunk];
#pragma unroll
    for (int c = 0; c < kChunk; ++c) {
      const int r = min(base + c, load_count - 1);
      grads[c].load(grad_base + r * value_dim);
      reads[c].load(read_base + r * value_dim);
      gate_logits[c] = gate_base[r * groups];
    }
#pragma unroll
    for (int c = 0; c < kChunk; ++c) {
      const int r = base + c;
      if (r >= per_slice) {  // uniform across the warp's lanes and slices
        break;
      }
      const bool valid = r < count && active;
      const float s = Sigmoid(static_cast<float>(gate_logits[c]));
      float part_read = 0.0f, part_neg = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float grad = valid ? grads[c][i] : 0.0f;
        const float neg_value = active ? neg[i] : 0.0f;
        part_read += grad * reads[c][i];
        part_neg += grad * neg_value;
        acc[i] += s * grad;
      }
      const float dps = ButterflySum<kLanes>(gated_output ? part_read + s * part_neg : part_read);
      float group_sum;
      if constexpr (kWholeGroup) {
        group_sum = ButterflySum<kLanes>(part_neg);
      } else {
        group_sum = SegmentSum(part_neg, lane - sub + group * group_lanes, group_lanes);
      }
      if (r < count && active && (kWholeGroup ? sub == 0 : sub % group_lanes == 0)) {
        gate_grad[(head_row + r) * groups + group] = static_cast<T>(-s * (1.0f - s) * group_sum);
      }
      if (r < count && live && sub == 0) {
        dpsum[stat_base + int64_t(first + r + 1) * padded_length] = active_row ? dps : 0.0f;
      }
    }
  }
  // neg_dy sums every slice's reads.
#pragma unroll
  for (int offset = kLanes; offset < kItemLanes; offset <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i] += __shfl_xor_sync(0xffffffff, acc[i], offset);
    }
  }
  float part_neg_dps = 0.0f;
  if (active) {
    union { uint4 packed; T scalars[8]; } out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out.scalars[i] = static_cast<T>(-acc[i]);
      part_neg_dps += static_cast<float>(out.scalars[i]) * neg[i];
    }
    if (slice == 0) {
      *reinterpret_cast<uint4*>(neg_dy + (row * kv_heads + kv) * value_dim + col) = out.packed;
    }
  }
  const float dps_neg = ButterflySum<kLanes>(part_neg_dps);
  if (live && item_lane == 0) {
    dpsum[stat_base] = active_row ? dps_neg : 0.0f;
  }
  if (live && item_lane < readers) {
    lse_log2[stat_base + int64_t(item_lane) * padded_length] =
        !active_row ? INFINITY : (lse_value == -INFINITY ? 0.0f : lse_value * float(M_LOG2E));
  }
  for (int r = item_lane + kItemLanes; live && r < readers; r += kItemLanes) {
    const float value = active_row ? read_lse[b * read_lse_batch_stride + int64_t(kv * rep + r - 1) * length + t] : 0.0f;
    lse_log2[stat_base + int64_t(r) * padded_length] =
        !active_row ? INFINITY : (value == -INFINITY ? 0.0f : value * float(M_LOG2E));
  }
  // The backward accumulates dQ into zeroed rows. Items tile each logical head's
  // padded_length x D floats in D-float chunks (whatever layout the backward
  // uses inside the head), so together they clear the whole buffer.
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (live) {
    float4* const zero_row = dq_accum + stat_base * dq_row_vectors;
    const int64_t head_vectors = int64_t(padded_length) * dq_row_vectors;
    for (int r = 0; r < readers; ++r) {
      for (int c = item_lane; c < dq_row_vectors; c += kItemLanes) {
        zero_row[r * head_vectors + c] = zero;
      }
    }
  }
  if (active_row && dk_accum != nullptr) {
    const int64_t kv_row = row * kv_heads + kv;
    for (int c = item_lane; c < dk_row_vectors; c += kItemLanes) { dk_accum[kv_row * dk_row_vectors + c] = zero; }
    for (int c = item_lane; c < dv_row_vectors; c += kItemLanes) { dv_accum[kv_row * dv_row_vectors + c] = zero; }
  }
}

template <typename T, typename StateT, int kLanes, int kSlices, bool kWholeGroup, bool kLooped>
void LaunchShape(const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& neg_read_state,
                 const torch::Tensor& read_state, const torch::Tensor& neg_read_lse, const torch::Tensor& read_lse,
                 torch::Tensor& neg_dy, torch::Tensor& gate_grad, torch::Tensor& dpsum,
                 torch::Tensor& lse_log2, torch::Tensor& dq_accum, const DkvAccum& dkv, int padded_length,
                 bool gated_output, cudaStream_t stream) {
  const int batch = int(dy.size(0)), length = int(dy.size(1));
  const int kv_heads = int(neg_dy.size(2)), rep = int(dy.size(2)) / kv_heads;
  const int value_dim = int(dy.size(3)), groups = int(gate.size(3));
  const int64_t items = int64_t(batch) * padded_length * kv_heads;
  constexpr int kThreads = kWarpSize * kWarpsPerBlock;
  const int64_t blocks = (items * kLanes * kSlices + kThreads - 1) / kThreads;
  TORCH_CHECK(items <= std::numeric_limits<int>::max() && blocks <= std::numeric_limits<int>::max(),
              "group_softdelta BWD preprocess: batch x length x KV heads exceeds the 32-bit item index");
  GroupSoftDeltaBwdPreprocessKernel<T, StateT, kLanes, kSlices, kWholeGroup, kLooped>
      <<<int(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const T*>(dy.data_ptr()), reinterpret_cast<const T*>(gate.data_ptr()),
      reinterpret_cast<const StateT*>(neg_read_state.data_ptr()), neg_read_state.stride(1),
      reinterpret_cast<const StateT*>(read_state.data_ptr()), read_state.stride(1), neg_read_lse.data_ptr<float>(),
      neg_read_lse.stride(0), read_lse.data_ptr<float>(), read_lse.stride(0), reinterpret_cast<T*>(neg_dy.data_ptr()),
      reinterpret_cast<T*>(gate_grad.data_ptr()), dpsum.data_ptr<float>(), lse_log2.data_ptr<float>(),
      reinterpret_cast<float4*>(dq_accum.data_ptr<float>()), int(dq_accum.size(2) / padded_length / 4),
      reinterpret_cast<float4*>(dkv.dk), dkv.dk_row / 4, reinterpret_cast<float4*>(dkv.dv), dkv.dv_row / 4, int(items),
      cutlass::FastDivmod(kv_heads), cutlass::FastDivmod(padded_length), length, rep, value_dim, groups,
      value_dim / groups / 8, gated_output);
}

template <typename T, typename StateT, int kLanes, int kSlices, bool kLooped>
void LaunchGroups(const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& neg_read_state,
                  const torch::Tensor& read_state, const torch::Tensor& neg_read_lse, const torch::Tensor& read_lse,
                  torch::Tensor& neg_dy, torch::Tensor& gate_grad, torch::Tensor& dpsum,
                  torch::Tensor& lse_log2, torch::Tensor& dq_accum, const DkvAccum& dkv, int padded_length,
                  bool gated_output, cudaStream_t stream) {
  if (gate.size(3) == 1) {
    LaunchShape<T, StateT, kLanes, kSlices, true, kLooped>(dy, gate, neg_read_state, read_state, neg_read_lse, read_lse, neg_dy,
                                                           gate_grad, dpsum, lse_log2, dq_accum, dkv,
                                                           padded_length, gated_output, stream);
  } else {
    LaunchShape<T, StateT, kLanes, kSlices, false, kLooped>(dy, gate, neg_read_state, read_state, neg_read_lse, read_lse, neg_dy,
                                                            gate_grad, dpsum, lse_log2, dq_accum, dkv,
                                                            padded_length, gated_output, stream);
  }
}

// Slices: enough for one round of at most kChunk reads each, within a warp;
// the widest layout loops over rounds for larger groups.
template <typename T, typename StateT, int kLanes>
void LaunchSlices(const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& neg_read_state,
                  const torch::Tensor& read_state, const torch::Tensor& neg_read_lse, const torch::Tensor& read_lse,
                  torch::Tensor& neg_dy, torch::Tensor& gate_grad, torch::Tensor& dpsum,
                  torch::Tensor& lse_log2, torch::Tensor& dq_accum, const DkvAccum& dkv, int padded_length,
                  bool gated_output, cudaStream_t stream) {
  constexpr int kMaxSlices = kWarpSize / kLanes < 8 ? kWarpSize / kLanes : 8;
  const int rep = int(dy.size(2) / neg_dy.size(2));
  int slices = 1;
  while (slices * kChunk < rep && slices < kMaxSlices) { slices *= 2; }
  const bool looped = slices * kChunk < rep;
#define XATTN_GROUP_SOFTDELTA_PRE_SLICES(N)                                                              \
  if (slices == N) {                                                                                     \
    if constexpr (N <= kMaxSlices) {                                                                     \
      if (!looped) {                                                                                     \
        LaunchGroups<T, StateT, kLanes, N, false>(dy, gate, neg_read_state, read_state, neg_read_lse, read_lse, neg_dy, \
                                                  gate_grad, dpsum, lse_log2, dq_accum, dkv,             \
                                                  padded_length, gated_output, stream);                  \
        return;                                                                                          \
      }                                                                                                  \
      if constexpr (N == kMaxSlices) {                                                                   \
        LaunchGroups<T, StateT, kLanes, N, true>(dy, gate, neg_read_state, read_state, neg_read_lse, read_lse, neg_dy, \
                                                 gate_grad, dpsum, lse_log2, dq_accum, dkv,              \
                                                 padded_length, gated_output, stream);                   \
        return;                                                                                          \
      }                                                                                                  \
    }                                                                                                    \
  }
  XATTN_GROUP_SOFTDELTA_PRE_SLICES(1)
  XATTN_GROUP_SOFTDELTA_PRE_SLICES(2)
  XATTN_GROUP_SOFTDELTA_PRE_SLICES(4)
  XATTN_GROUP_SOFTDELTA_PRE_SLICES(8)
#undef XATTN_GROUP_SOFTDELTA_PRE_SLICES
  TORCH_CHECK(false, "group_softdelta BWD preprocess: no slice layout for ", rep, " query heads per KV head");
}

template <typename T, typename StateT>
void Launch(const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& neg_read_state,
            const torch::Tensor& read_state, const torch::Tensor& neg_read_lse, const torch::Tensor& read_lse,
            torch::Tensor& neg_dy, torch::Tensor& gate_grad, torch::Tensor& dpsum,
            torch::Tensor& lse_log2, torch::Tensor& dq_accum, const DkvAccum& dkv, int padded_length,
            bool gated_output, cudaStream_t stream) {
  const int column_groups = int(dy.size(3) / 8);
#define XATTN_GROUP_SOFTDELTA_PRE_LANES(N)                                                               \
  if (column_groups <= N) {                                                                              \
    LaunchSlices<T, StateT, N>(dy, gate, neg_read_state, read_state, neg_read_lse, read_lse, neg_dy, gate_grad, dpsum, \
                               lse_log2, dq_accum, dkv, padded_length, gated_output, stream);            \
    return;                                                                                              \
  }
  XATTN_GROUP_SOFTDELTA_PRE_LANES(1)
  XATTN_GROUP_SOFTDELTA_PRE_LANES(2)
  XATTN_GROUP_SOFTDELTA_PRE_LANES(4)
  XATTN_GROUP_SOFTDELTA_PRE_LANES(8)
  XATTN_GROUP_SOFTDELTA_PRE_LANES(16)
  XATTN_GROUP_SOFTDELTA_PRE_LANES(32)
#undef XATTN_GROUP_SOFTDELTA_PRE_LANES
  TORCH_CHECK(false, "group_softdelta BWD preprocess supports V <= 256, got V=", dy.size(3));
}

}  // namespace

void GroupSoftDeltaBwdPreprocess(
    const torch::Tensor& dy, const torch::Tensor& gate, const torch::Tensor& neg_read_state,
    const torch::Tensor& read_state, const torch::Tensor& neg_read_lse, const torch::Tensor& read_lse,
    torch::Tensor& neg_dy, torch::Tensor& gate_grad, torch::Tensor& dpsum, torch::Tensor& lse_log2,
    torch::Tensor& dq_accum, int padded_length, bool gated_output, const DkvAccum& dkv) {
  const auto B = dy.size(0), L = dy.size(1), H = dy.size(2), V = dy.size(3);
  const auto C = gate.size(3), Hkv = neg_dy.size(2);
  const auto logical_heads = H + Hkv;
  TORCH_CHECK(dy.is_contiguous() && gate.is_contiguous() && neg_dy.is_contiguous() && gate_grad.is_contiguous() &&
                  dpsum.is_contiguous() && lse_log2.is_contiguous(),
              "group_softdelta BWD preprocess tensors must be contiguous");
  // Forward state: dense rows of heads (row-major over batch and row), dense LSE rows.
  for (const torch::Tensor* t : {&neg_read_state, &read_state}) {
    TORCH_CHECK(t->dim() == 4 && t->stride(3) == 1 && t->stride(2) == V && t->stride(0) == L * t->stride(1) &&
                    t->scalar_type() == neg_read_state.scalar_type(),
                "group_softdelta BWD preprocess needs row-major forward state");
  }
  for (const torch::Tensor* t : {&neg_read_lse, &read_lse}) {
    TORCH_CHECK(t->dim() == 3 && t->scalar_type() == at::kFloat && t->stride(2) == 1 && t->stride(1) == L,
                "group_softdelta BWD preprocess needs FP32 [B, heads, L] LSE rows");
  }
  TORCH_CHECK(dkv.dk == nullptr || (dkv.dk_row % 4 == 0 && dkv.dv_row % 4 == 0),
              "group_softdelta BWD preprocess: dK / dV accumulator rows must be multiples of 4 floats");
  TORCH_CHECK(V % 8 == 0 && V <= 256 && V % C == 0 && (V / C) % 8 == 0,
              "group_softdelta BWD preprocess needs V <= 256 (a multiple of 8) and value groups of a multiple of 8");
  TORCH_CHECK(gate.dim() == 4 && gate.sizes() == torch::IntArrayRef({B, L, H, C}) &&
                  neg_read_state.sizes() == torch::IntArrayRef({B, L, Hkv, V}) &&
                  read_state.sizes() == torch::IntArrayRef({B, L, H, V}) &&
                  neg_read_lse.sizes() == torch::IntArrayRef({B, Hkv, L}) && read_lse.sizes() == torch::IntArrayRef({B, H, L}) &&
                  neg_dy.sizes() == torch::IntArrayRef({B, L, Hkv, V}) && gate_grad.sizes() == gate.sizes() &&
                  dpsum.sizes() == torch::IntArrayRef({B, logical_heads, padded_length}) &&
                  lse_log2.sizes() == dpsum.sizes() && padded_length >= L && dq_accum.is_contiguous() &&
                  dq_accum.dim() == 3 && dq_accum.size(0) == B && dq_accum.size(1) == logical_heads &&
                  dq_accum.size(2) % (4 * padded_length) == 0 && dq_accum.scalar_type() == at::kFloat,
              "group_softdelta BWD preprocess shapes are inconsistent");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool fp32_state = neg_read_state.scalar_type() == at::kFloat;
#define XATTN_GROUP_SOFTDELTA_PRE_LAUNCH(T, StateT)                                                           \
  Launch<T, StateT>(dy, gate, neg_read_state, read_state, neg_read_lse, read_lse, neg_dy, gate_grad, dpsum, lse_log2, dq_accum, \
                    dkv, padded_length, gated_output, stream)
  if (dy.scalar_type() == at::kHalf) {
    if (fp32_state) XATTN_GROUP_SOFTDELTA_PRE_LAUNCH(half, float);
    else XATTN_GROUP_SOFTDELTA_PRE_LAUNCH(half, half);
  } else {
    if (fp32_state) XATTN_GROUP_SOFTDELTA_PRE_LAUNCH(nv_bfloat16, float);
    else XATTN_GROUP_SOFTDELTA_PRE_LAUNCH(nv_bfloat16, nv_bfloat16);
  }
#undef XATTN_GROUP_SOFTDELTA_PRE_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ops
}  // namespace xattn

// Author: Shicheng Wen

#include "group_softdelta/gate.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/fast_math.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace xattn {
namespace ops {
namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;

__device__ __forceinline__ float Sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

// Loads/stores kVec consecutive elements as FP32; kVec = 8 uses 16-byte vectors
// for 16-bit types and two for FP32.
template <typename T, int kVec>
__device__ __forceinline__ void LoadVec(const T* pointer, float (&values)[kVec]) {
  if constexpr (kVec == 8 && sizeof(T) == 2) {
    union { uint4 packed; T scalars[8]; } vector;
    vector.packed = *reinterpret_cast<const uint4*>(pointer);
#pragma unroll
    for (int i = 0; i < 8; ++i) values[i] = static_cast<float>(vector.scalars[i]);
  } else if constexpr (kVec == 8 && sizeof(T) == 4) {
    const float4 low = reinterpret_cast<const float4*>(pointer)[0];
    const float4 high = reinterpret_cast<const float4*>(pointer)[1];
    values[0] = low.x; values[1] = low.y; values[2] = low.z; values[3] = low.w;
    values[4] = high.x; values[5] = high.y; values[6] = high.z; values[7] = high.w;
  } else {
#pragma unroll
    for (int i = 0; i < kVec; ++i) values[i] = static_cast<float>(pointer[i]);
  }
}

template <typename T, int kVec>
__device__ __forceinline__ void StoreVec(T* pointer, const float (&values)[kVec]) {
  if constexpr (kVec == 8 && sizeof(T) == 2) {
    union { uint4 packed; T scalars[8]; } vector;
#pragma unroll
    for (int i = 0; i < 8; ++i) vector.scalars[i] = static_cast<T>(values[i]);
    *reinterpret_cast<uint4*>(pointer) = vector.packed;
  } else {
#pragma unroll
    for (int i = 0; i < kVec; ++i) pointer[i] = static_cast<T>(values[i]);
  }
}

// Row strides (elements) of the reads: read row (bt, kv, r) starts at
// bt * read_bt + kv * read_kv + r * read_r, negative read row (bt, kv) at
// bt * neg_bt + kv * neg_kv; value group c adds c * Dg. Contiguous reads
// and the fused forward's class-major [B, L, Hkv + H, C, Dg] readers both fit.
struct ReadStrides {
  int64_t read_bt, read_kv, read_r, neg_bt, neg_kv;
};

// One kSubwarp-lane group per gate row (b, t, h, c). Lane l owns the kVec-wide
// chunks l, l + kSubwarp, ... of the Dg columns (kChunks of them at most).
template <typename T, typename StateT, int kVec, int kSubwarp, int kChunks>
__global__ void GroupSoftDeltaGateFwdKernel(
    const StateT* __restrict__ read, const StateT* __restrict__ neg_read,
    const T* __restrict__ gate, T* __restrict__ output,
    int64_t rows, int groups, int q_heads, int rep, int group_dim, ReadStrides strides) {
  const int lane = threadIdx.x % kWarpSize;
  const int sublane = lane % kSubwarp;
  const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / kSubwarp;
  if (row >= rows) {
    return;
  }
  const int c = int(row % groups);
  const int64_t head_row = row / groups;
  const int h = int(head_row % q_heads);
  const int64_t bt = head_row / q_heads;
  const int kv = h / rep;
  const int64_t read_offset =
      bt * strides.read_bt + kv * strides.read_kv + (h - kv * rep) * strides.read_r + int64_t(c) * group_dim;
  const int64_t neg_offset = bt * strides.neg_bt + kv * strides.neg_kv + int64_t(c) * group_dim;
  const float gate_value = Sigmoid(static_cast<float>(gate[row]));
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int column = (chunk * kSubwarp + sublane) * kVec;
    if (column < group_dim) {
      float a[kVec], y[kVec];
      LoadVec<StateT, kVec>(read + read_offset + column, a);
      LoadVec<StateT, kVec>(neg_read + neg_offset + column, y);
#pragma unroll
      for (int i = 0; i < kVec; ++i) a[i] -= gate_value * y[i];
      StoreVec<T, kVec>(output + row * group_dim + column, a);
    }
  }
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
  __device__ __forceinline__ float operator[](int i) const {
    return static_cast<float>(reinterpret_cast<const T*>(words)[i]);
  }
};

// One kLanes-lane item per (b, t, KV head kv); lane l owns columns
// [8 l, 8 l + 8) of V (lanes past V idle) for the group's rep query heads, in
// rounds of kChunk (one round unless kLooped). y is read once per item; every
// load is unconditional from an in-bounds address and issued before any use
// (guarded loads become branches interleaved with their uses).
template <typename T, typename StateT, int kLanes, bool kLooped>
__global__ void __launch_bounds__(kThreads) GroupSoftDeltaGateFwdGroupKernel(
    const StateT* __restrict__ read, const StateT* __restrict__ neg_read, const T* __restrict__ gate,
    T* __restrict__ output, int items, cutlass::FastDivmod kv_divmod, int rep, int value_dim, int groups,
    int group_dim, ReadStrides strides) {
  constexpr int kChunk = 4;
  const int sub = threadIdx.x % kLanes;
  const int item = int((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / kLanes);
  const bool live = item < items;
  int kv;
  const int bt = kv_divmod.divmod(kv, live ? item : 0);
  const int col = sub * 8;
  const bool active = live && col < value_dim;
  const int load_col = col < value_dim ? col : 0;
  const int group = load_col / group_dim;
  const int64_t q_row = int64_t(bt) * kv_divmod.divisor * rep + int64_t(kv) * rep;
  Packed8<StateT> y;
  y.load(neg_read + bt * strides.neg_bt + kv * strides.neg_kv + load_col);
  const StateT* read_row = read + bt * strides.read_bt + kv * strides.read_kv + load_col;
  const T* gate_row = gate + q_row * groups + group;
  T* out_row = output + q_row * value_dim + col;
  const int rounds = kLooped ? (rep + kChunk - 1) / kChunk : 1;
  for (int round = 0; round < rounds; ++round) {
    const int base = round * kChunk;
    Packed8<StateT> a[kChunk];
    T logits[kChunk];
#pragma unroll
    for (int c = 0; c < kChunk; ++c) {
      const int r = min(base + c, rep - 1);
      a[c].load(read_row + r * strides.read_r);
      logits[c] = gate_row[r * groups];
    }
#pragma unroll
    for (int c = 0; c < kChunk; ++c) {
      const int r = base + c;
      if (r >= rep) {  // uniform: every item shares rep
        break;
      }
      const float s = Sigmoid(static_cast<float>(logits[c]));
      union { uint4 packed; T scalars[8]; } out;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        out.scalars[i] = static_cast<T>(a[c][i] - s * y[i]);
      }
      if (active) {
        *reinterpret_cast<uint4*>(out_row + int64_t(r) * value_dim) = out.packed;
      }
    }
  }
}

template <typename T, typename StateT, int kLanes, bool kLooped>
void LaunchFwdGroup(const void* read, const void* neg_read, const void* gate, void* output, int64_t items,
                    int kv_heads, int rep, int value_dim, int groups, int group_dim, const ReadStrides& strides,
                    cudaStream_t stream) {
  const int64_t blocks = (items * kLanes + kThreads - 1) / kThreads;
  TORCH_CHECK(items <= std::numeric_limits<int>::max() && blocks <= std::numeric_limits<int>::max(),
              "group_softdelta gate: batch x length x KV heads exceeds the 32-bit item index");
  GroupSoftDeltaGateFwdGroupKernel<T, StateT, kLanes, kLooped><<<int(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const StateT*>(read), reinterpret_cast<const StateT*>(neg_read),
      reinterpret_cast<const T*>(gate), reinterpret_cast<T*>(output), int(items),
      cutlass::FastDivmod(kv_heads), rep, value_dim, groups, group_dim, strides);
}

// The grouped forward when V is a multiple of 8 within 256 and value groups are
// whole vectors; returns false otherwise (per-row kernel).
template <typename T, typename StateT>
bool TryLaunchFwdGroup(const void* read, const void* neg_read, const void* gate, void* output, int64_t items,
                       int kv_heads, int rep, int value_dim, int groups, int group_dim, const ReadStrides& strides,
                       cudaStream_t stream) {
  if (value_dim % 8 != 0 || value_dim > 256 || group_dim % 8 != 0) {
    return false;
  }
  const bool looped = rep > 4;
  auto run = [&](auto lanes_tag) {
    constexpr int kLanes = decltype(lanes_tag)::value;
    if (looped) {
      LaunchFwdGroup<T, StateT, kLanes, true>(read, neg_read, gate, output, items, kv_heads, rep, value_dim,
                                              groups, group_dim, strides, stream);
    } else {
      LaunchFwdGroup<T, StateT, kLanes, false>(read, neg_read, gate, output, items, kv_heads, rep, value_dim,
                                               groups, group_dim, strides, stream);
    }
  };
  const int column_groups = value_dim / 8;
  if (column_groups <= 1) run(std::integral_constant<int, 1>{});
  else if (column_groups <= 2) run(std::integral_constant<int, 2>{});
  else if (column_groups <= 4) run(std::integral_constant<int, 4>{});
  else if (column_groups <= 8) run(std::integral_constant<int, 8>{});
  else if (column_groups <= 16) run(std::integral_constant<int, 16>{});
  else run(std::integral_constant<int, 32>{});
  return true;
}

// One kSubwarp-lane group per KV row (b, t, kv, c): loops over the rep query
// heads in order, accumulating the negative read's gradient in registers.
template <typename T, typename StateT, int kVec, int kSubwarp, int kChunks>
__global__ void GroupSoftDeltaGateBwdKernel(
    const T* __restrict__ dy, const StateT* __restrict__ neg_read,
    const T* __restrict__ gate, T* __restrict__ neg_read_grad,
    T* __restrict__ gate_grad,
    int64_t kv_rows, int groups, int kv_heads, int rep, int group_dim, ReadStrides strides) {
  const int lane = threadIdx.x % kWarpSize;
  const int sublane = lane % kSubwarp;
  const int64_t kv_row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / kSubwarp;
  // Every lane of a warp takes part in the shuffles; out-of-range groups read nothing.
  const bool active = kv_row < kv_rows;
  const int c = active ? int(kv_row % groups) : 0;
  const int64_t kv_head_row = kv_row / groups;
  const int kv = int(kv_head_row % kv_heads);
  const int64_t bt = kv_head_row / kv_heads;
  const int q_heads = kv_heads * rep;

  float y[kChunks][kVec], acc[kChunks][kVec];
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int column = (chunk * kSubwarp + sublane) * kVec;
#pragma unroll
    for (int i = 0; i < kVec; ++i) { y[chunk][i] = 0.0f; acc[chunk][i] = 0.0f; }
    if (active && column < group_dim) {
      LoadVec<StateT, kVec>(neg_read + bt * strides.neg_bt + kv * strides.neg_kv +
                                int64_t(c) * group_dim + column,
                            y[chunk]);
    }
  }
  for (int r = 0; r < rep; ++r) {
    const int64_t row = (bt * q_heads + kv * rep + r) * groups + c;
    const float gate_value = active ? Sigmoid(static_cast<float>(gate[row])) : 0.0f;
    float projection = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < kChunks; ++chunk) {
      const int column = (chunk * kSubwarp + sublane) * kVec;
      if (active && column < group_dim) {
        float grad[kVec];
        LoadVec<T, kVec>(dy + row * group_dim + column, grad);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          acc[chunk][i] += gate_value * grad[i];
          projection += grad[i] * y[chunk][i];
        }
      }
    }
#pragma unroll
    for (int offset = kSubwarp / 2; offset > 0; offset /= 2) {
      projection += __shfl_xor_sync(0xffffffff, projection, offset, kSubwarp);
    }
    if (active && sublane == 0) {
      gate_grad[row] = static_cast<T>(-gate_value * (1.0f - gate_value) * projection);
    }
  }
  if (!active) {
    return;
  }
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int column = (chunk * kSubwarp + sublane) * kVec;
    if (column < group_dim) {
#pragma unroll
      for (int i = 0; i < kVec; ++i) acc[chunk][i] = -acc[chunk][i];
      StoreVec<T, kVec>(neg_read_grad + kv_row * group_dim + column, acc[chunk]);
    }
  }
}

// out[i] += other[i] (+ extra[row - extra_row_offset] on rows >= extra_row_offset), 8 elements per thread
// when aligned. Rows are the per-batch time index; row_elems = Hkv * D.
template <typename T, int kVec>
__global__ void GroupSoftDeltaMergeKernel(
    T* __restrict__ out, const T* __restrict__ other, const T* __restrict__ extra,
    int64_t elems, int64_t row_elems, int rows, int extra_rows) {
  const int64_t e = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * kVec;
  if (e >= elems) {
    return;
  }
  float a[kVec], b[kVec];
  LoadVec<T, kVec>(out + e, a);
  LoadVec<T, kVec>(other + e, b);
#pragma unroll
  for (int i = 0; i < kVec; ++i) a[i] += b[i];
  if (extra != nullptr) {
    const int64_t per_batch = int64_t(rows) * row_elems;
    const int64_t batch = e / per_batch;
    const int64_t within = e - batch * per_batch;
    const int64_t row = within / row_elems;
    const int extra_row_offset = rows - extra_rows;
    if (row >= extra_row_offset) {
      float c[kVec];
      LoadVec<T, kVec>(extra + batch * extra_rows * row_elems + (row - extra_row_offset) * row_elems +
                           (within - row * row_elems), c);
#pragma unroll
      for (int i = 0; i < kVec; ++i) a[i] += c[i];
    }
  }
  StoreVec<T, kVec>(out + e, a);
}

int NextPowerOfTwo(int value) {
  int result = 1;
  while (result < value) result *= 2;
  return result;
}

bool Aligned16(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0;
}

struct GateShape {
  int64_t rows;  // gate rows (forward) or KV rows (backward)
  int groups, heads, rep, group_dim;
  ReadStrides strides;  // forward only
};

template <typename T, typename StateT, int kVec, int kSubwarp, int kChunks>
void LaunchFwd(const torch::Tensor& read, const torch::Tensor& neg_read, const torch::Tensor& gate,
               torch::Tensor& output, const GateShape& shape, cudaStream_t stream) {
  const int64_t threads = shape.rows * kSubwarp;
  GroupSoftDeltaGateFwdKernel<T, StateT, kVec, kSubwarp, kChunks>
      <<<int((threads + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
          reinterpret_cast<const StateT*>(read.data_ptr()),
          reinterpret_cast<const StateT*>(neg_read.data_ptr()),
          reinterpret_cast<const T*>(gate.data_ptr()),
          reinterpret_cast<T*>(output.data_ptr()),
          shape.rows, shape.groups, shape.heads, shape.rep, shape.group_dim, shape.strides);
}

template <typename T, typename StateT, int kVec, int kSubwarp, int kChunks>
void LaunchBwd(const torch::Tensor& dy, const torch::Tensor& neg_read, const torch::Tensor& gate,
               torch::Tensor& neg_read_grad, torch::Tensor& gate_grad, const GateShape& shape,
               cudaStream_t stream) {
  const int64_t threads = shape.rows * kSubwarp;
  GroupSoftDeltaGateBwdKernel<T, StateT, kVec, kSubwarp, kChunks>
      <<<int((threads + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
          reinterpret_cast<const T*>(dy.data_ptr()),
          reinterpret_cast<const StateT*>(neg_read.data_ptr()),
          reinterpret_cast<const T*>(gate.data_ptr()),
          reinterpret_cast<T*>(neg_read_grad.data_ptr()),
          reinterpret_cast<T*>(gate_grad.data_ptr()),
          shape.rows, shape.groups, shape.heads, shape.rep, shape.group_dim, shape.strides);
}

// Launcher selection by group width: kVec = 8 vectors when every row start is
// 16-byte aligned, otherwise one element per lane.
template <typename T, typename StateT, bool kForward>
struct GateDispatch {
  template <int kVec, int kSubwarp, int kChunks, typename... Args>
  static void Run(Args&&... args) {
    if constexpr (kForward) {
      LaunchFwd<T, StateT, kVec, kSubwarp, kChunks>(args...);
    } else {
      LaunchBwd<T, StateT, kVec, kSubwarp, kChunks>(args...);
    }
  }

  template <typename... Args>
  static void Dispatch(int group_dim, bool aligned, Args&&... args) {
    if (aligned && group_dim % 8 == 0) {
      switch (NextPowerOfTwo(group_dim / 8)) {
        case 1: Run<8, 1, 1>(args...); return;
        case 2: Run<8, 2, 1>(args...); return;
        case 4: Run<8, 4, 1>(args...); return;
        case 8: Run<8, 8, 1>(args...); return;
        case 16: Run<8, 16, 1>(args...); return;
        case 32: Run<8, 32, 1>(args...); return;
        default: break;
      }
    }
    TORCH_CHECK(group_dim <= 256, "group_softdelta gate group dimension must be <= 256");
    Run<1, 32, 8>(args...);
  }
};

void CheckInputs(const torch::Tensor& read_or_grad, const torch::Tensor& neg_read,
                 const torch::Tensor& gate, const char* first_name) {
  TORCH_CHECK(read_or_grad.is_cuda() && neg_read.is_cuda() && gate.is_cuda(),
              "group_softdelta gate requires CUDA tensors");
  TORCH_CHECK(read_or_grad.dim() == 5 && neg_read.dim() == 5 && gate.dim() == 5,
              "group_softdelta gate expects 5D tensors");
  TORCH_CHECK(gate.scalar_type() == at::kHalf || gate.scalar_type() == at::kBFloat16,
              "group_softdelta gate supports only fp16 and bf16 gates");
  TORCH_CHECK(read_or_grad.is_contiguous() && neg_read.is_contiguous() && gate.is_contiguous(),
              "group_softdelta gate tensors must be contiguous");
  TORCH_CHECK(read_or_grad.device() == gate.device() && neg_read.device() == gate.device(),
              "group_softdelta gate tensors must share a device");
  TORCH_CHECK(neg_read.scalar_type() == gate.scalar_type() ||
                  neg_read.scalar_type() == at::kFloat,
              "group_softdelta neg_read must use the gate dtype or FP32 state");
  const auto B = read_or_grad.size(0), L = read_or_grad.size(1), H = read_or_grad.size(2);
  const auto C = read_or_grad.size(3), Dg = read_or_grad.size(4);
  TORCH_CHECK(gate.size(0) == B && gate.size(1) == L && gate.size(2) == H &&
                  gate.size(3) == C && gate.size(4) == 1,
              "group_softdelta gate shape must be [B, L, H, C, 1]");
  const auto Hkv = neg_read.size(2);
  TORCH_CHECK(neg_read.size(0) == B && neg_read.size(1) == L && neg_read.size(3) == C &&
                  neg_read.size(4) == Dg && Hkv > 0 && H % Hkv == 0,
              "group_softdelta neg_read must be [B, L, Hkv, C, Dg] with H % Hkv == 0");
  TORCH_CHECK(read_or_grad.numel() > 0, first_name, " must be nonempty");
}

template <typename T, bool kForward, typename... Args>
void DispatchState(bool fp32_state, int group_dim, bool aligned, Args&&... args) {
  if (fp32_state) {
    GateDispatch<T, float, kForward>::Dispatch(group_dim, aligned, args...);
  } else {
    GateDispatch<T, T, kForward>::Dispatch(group_dim, aligned, args...);
  }
}

template <typename T>
void LaunchMerge(torch::Tensor& out, const torch::Tensor& other, const c10::optional<torch::Tensor>& extra,
                 cudaStream_t stream) {
  const int64_t elems = out.numel();
  const int rows = int(out.size(1));
  const int64_t row_elems = elems / (out.size(0) * rows);
  const T* extra_ptr = extra.has_value() ? reinterpret_cast<const T*>(extra->data_ptr()) : nullptr;
  const int extra_rows = extra.has_value() ? int(extra->size(1)) : 0;
  const bool aligned = row_elems % 8 == 0 && Aligned16(out) && Aligned16(other) &&
                       (!extra.has_value() || Aligned16(*extra));
  if (aligned) {
    const int64_t threads = elems / 8;
    GroupSoftDeltaMergeKernel<T, 8><<<int((threads + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        reinterpret_cast<T*>(out.data_ptr()), reinterpret_cast<const T*>(other.data_ptr()), extra_ptr,
        elems, row_elems, rows, extra_rows);
  } else {
    GroupSoftDeltaMergeKernel<T, 1><<<int((elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        reinterpret_cast<T*>(out.data_ptr()), reinterpret_cast<const T*>(other.data_ptr()), extra_ptr,
        elems, row_elems, rows, extra_rows);
  }
}

void MergeInPlace(torch::Tensor& out, const torch::Tensor& other, const c10::optional<torch::Tensor>& extra,
                  cudaStream_t stream) {
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && other.is_contiguous() && out.sizes() == other.sizes() &&
                  out.scalar_type() == other.scalar_type() && out.dim() >= 3,
              "group_softdelta gradient merge expects matching contiguous CUDA tensors");
  TORCH_CHECK(out.scalar_type() == at::kHalf || out.scalar_type() == at::kBFloat16,
              "group_softdelta gradient merge supports only fp16 and bf16");
  if (extra.has_value()) {
    TORCH_CHECK(extra->is_contiguous() && extra->scalar_type() == out.scalar_type() &&
                    extra->dim() == out.dim() && extra->size(0) == out.size(0) &&
                    extra->size(1) <= out.size(1) &&
                    extra->numel() / extra->size(1) == out.numel() / out.size(1),
                "group_softdelta negative-query gradient must be [B, Lq, Hkv, D] with Lq <= Lk");
  }
  if (out.numel() == 0) {
    return;
  }
  if (out.scalar_type() == at::kHalf) {
    LaunchMerge<half>(out, other, extra, stream);
  } else {
    LaunchMerge<nv_bfloat16>(out, other, extra, stream);
  }
}

}  // namespace

torch::Tensor GroupSoftDeltaGateFwd(
    const torch::Tensor& read, const torch::Tensor& neg_read, const torch::Tensor& gate) {
  CheckInputs(read, neg_read, gate, "read");
  TORCH_CHECK(read.scalar_type() == neg_read.scalar_type(),
              "group_softdelta read and neg_read must share a dtype");
  const c10::cuda::CUDAGuard guard(read.device());
  auto output = torch::empty(read.sizes(), gate.options());
  const int group_dim = int(read.size(4));
  const int64_t row = read.size(3) * group_dim;
  const int64_t rep = read.size(2) / neg_read.size(2);
  const GateShape shape{read.numel() / group_dim, int(read.size(3)), int(read.size(2)), int(rep), group_dim,
                        ReadStrides{read.size(2) * row, rep * row, row, neg_read.size(2) * row, row}};
  const bool aligned = Aligned16(read) && Aligned16(neg_read) && Aligned16(output);
  const bool fp32_state = read.scalar_type() == at::kFloat;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t items = read.size(0) * read.size(1) * neg_read.size(2);
  const int value_dim = int(row);
  bool done = false;
  if (aligned) {
    if (gate.scalar_type() == at::kHalf) {
      done = fp32_state ? TryLaunchFwdGroup<half, float>(read.data_ptr(), neg_read.data_ptr(), gate.data_ptr(),
                                                         output.data_ptr(), items, int(neg_read.size(2)), int(rep),
                                                         value_dim, int(read.size(3)), group_dim, shape.strides, stream)
                        : TryLaunchFwdGroup<half, half>(read.data_ptr(), neg_read.data_ptr(), gate.data_ptr(),
                                                        output.data_ptr(), items, int(neg_read.size(2)), int(rep),
                                                        value_dim, int(read.size(3)), group_dim, shape.strides, stream);
    } else {
      done = fp32_state ? TryLaunchFwdGroup<nv_bfloat16, float>(read.data_ptr(), neg_read.data_ptr(),
                                                                gate.data_ptr(), output.data_ptr(), items,
                                                                int(neg_read.size(2)), int(rep), value_dim,
                                                                int(read.size(3)), group_dim, shape.strides, stream)
                        : TryLaunchFwdGroup<nv_bfloat16, nv_bfloat16>(read.data_ptr(), neg_read.data_ptr(),
                                                                      gate.data_ptr(), output.data_ptr(), items,
                                                                      int(neg_read.size(2)), int(rep), value_dim,
                                                                      int(read.size(3)), group_dim, shape.strides,
                                                                      stream);
    }
  }
  if (!done) {
    if (gate.scalar_type() == at::kHalf) {
      DispatchState<half, true>(fp32_state, group_dim, aligned, read, neg_read, gate, output, shape, stream);
    } else {
      DispatchState<nv_bfloat16, true>(fp32_state, group_dim, aligned, read, neg_read, gate, output, shape,
                                       stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor GroupSoftDeltaGateFwdReaders(const torch::Tensor& readers, const torch::Tensor& gate) {
  TORCH_CHECK(readers.is_cuda() && gate.is_cuda() && readers.dim() == 5 && gate.dim() == 5 &&
                  readers.is_contiguous() && gate.is_contiguous(),
              "group_softdelta readers gate expects contiguous 5D CUDA tensors");
  TORCH_CHECK(gate.scalar_type() == at::kHalf || gate.scalar_type() == at::kBFloat16,
              "group_softdelta gate supports only fp16 and bf16 gates");
  TORCH_CHECK(readers.scalar_type() == gate.scalar_type() || readers.scalar_type() == at::kFloat,
              "group_softdelta readers must use the gate dtype or FP32 state");
  const auto B = gate.size(0), L = gate.size(1), H = gate.size(2), C = gate.size(3);
  const auto logical_heads = readers.size(2);
  const auto Hkv = logical_heads - H;
  TORCH_CHECK(readers.size(0) == B && readers.size(1) == L && readers.size(3) == C && Hkv > 0 &&
                  H % Hkv == 0 && gate.size(4) == 1,
              "group_softdelta readers must be [B, L, H + Hkv, C, Dg] for gate [B, L, H, C, 1]");
  const c10::cuda::CUDAGuard guard(gate.device());
  const int group_dim = int(readers.size(4));
  auto output = torch::empty({B, L, H, C, group_dim}, gate.options());
  const int64_t row = C * group_dim;
  const int64_t rep = H / Hkv;
  const GateShape shape{output.numel() / group_dim, int(C), int(H), int(rep), group_dim,
                        ReadStrides{logical_heads * row, rep * row, row, logical_heads * row, row}};
  // Readers heads [0, Hkv) hold the negative reads, [Hkv, Hkv + H) the query reads.
  const bool aligned = row % 8 == 0 && Aligned16(readers) && Aligned16(output);
  const bool fp32_state = readers.scalar_type() == at::kFloat;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t items = B * L * Hkv;
  const void* const read_ptr = static_cast<const char*>(readers.data_ptr()) + Hkv * row * readers.element_size();
  bool done = false;
  if (aligned) {
    if (gate.scalar_type() == at::kHalf) {
      done = fp32_state ? TryLaunchFwdGroup<half, float>(read_ptr, readers.data_ptr(), gate.data_ptr(),
                                                         output.data_ptr(), items, int(Hkv), int(rep), int(row),
                                                         int(C), group_dim, shape.strides, stream)
                        : TryLaunchFwdGroup<half, half>(read_ptr, readers.data_ptr(), gate.data_ptr(),
                                                        output.data_ptr(), items, int(Hkv), int(rep), int(row),
                                                        int(C), group_dim, shape.strides, stream);
    } else {
      done = fp32_state ? TryLaunchFwdGroup<nv_bfloat16, float>(read_ptr, readers.data_ptr(), gate.data_ptr(),
                                                                output.data_ptr(), items, int(Hkv), int(rep),
                                                                int(row), int(C), group_dim, shape.strides, stream)
                        : TryLaunchFwdGroup<nv_bfloat16, nv_bfloat16>(read_ptr, readers.data_ptr(),
                                                                      gate.data_ptr(), output.data_ptr(), items,
                                                                      int(Hkv), int(rep), int(row), int(C),
                                                                      group_dim, shape.strides, stream);
    }
  }
  if (!done) {
    const torch::Tensor reads = readers.flatten(0, 1).narrow(1, Hkv, H);
    if (gate.scalar_type() == at::kHalf) {
      DispatchState<half, true>(fp32_state, group_dim, aligned, reads, readers, gate, output, shape, stream);
    } else {
      DispatchState<nv_bfloat16, true>(fp32_state, group_dim, aligned, reads, readers, gate, output, shape,
                                       stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> GroupSoftDeltaGateBwd(
    const torch::Tensor& dy, const torch::Tensor& neg_read, const torch::Tensor& gate) {
  CheckInputs(dy, neg_read, gate, "dy");
  TORCH_CHECK(dy.scalar_type() == gate.scalar_type(),
              "group_softdelta dy must use the gate dtype");
  const c10::cuda::CUDAGuard guard(dy.device());
  auto neg_read_grad = torch::empty(neg_read.sizes(), gate.options());
  auto gate_grad = torch::empty(gate.sizes(), gate.options());
  const int group_dim = int(dy.size(4));
  const int kv_heads = int(neg_read.size(2));
  const int64_t row = dy.size(3) * group_dim;
  const GateShape shape{neg_read.numel() / group_dim, int(dy.size(3)), kv_heads,
                        int(dy.size(2)) / kv_heads, group_dim,
                        ReadStrides{0, 0, 0, kv_heads * row, row}};
  const bool aligned = Aligned16(dy) && Aligned16(neg_read) && Aligned16(neg_read_grad);
  const bool fp32_state = neg_read.scalar_type() == at::kFloat;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (gate.scalar_type() == at::kHalf) {
    DispatchState<half, false>(fp32_state, group_dim, aligned, dy, neg_read, gate,
                               neg_read_grad, gate_grad, shape, stream);
  } else {
    DispatchState<nv_bfloat16, false>(fp32_state, group_dim, aligned, dy, neg_read, gate,
                                      neg_read_grad, gate_grad, shape, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {neg_read_grad, gate_grad};
}

std::tuple<torch::Tensor, torch::Tensor> GroupSoftDeltaMergeGradients(
    torch::Tensor read_key_grad, const torch::Tensor& neg_read_key_grad,
    const c10::optional<torch::Tensor>& dneg_q, torch::Tensor read_value_grad,
    const torch::Tensor& neg_read_value_grad) {
  const c10::cuda::CUDAGuard guard(read_key_grad.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  MergeInPlace(read_key_grad, neg_read_key_grad, dneg_q, stream);
  MergeInPlace(read_value_grad, neg_read_value_grad, c10::nullopt, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {read_key_grad, read_value_grad};
}

std::tuple<torch::Tensor, torch::Tensor> GroupSoftDeltaGateBwdReaders(
    const torch::Tensor& dy, const torch::Tensor& readers, const torch::Tensor& gate) {
  TORCH_CHECK(dy.is_cuda() && readers.is_cuda() && gate.is_cuda() && dy.dim() == 5 &&
                  readers.dim() == 5 && gate.dim() == 5 && dy.is_contiguous() &&
                  readers.is_contiguous() && gate.is_contiguous(),
              "group_softdelta readers gate BWD expects contiguous 5D CUDA tensors");
  TORCH_CHECK(gate.scalar_type() == at::kHalf || gate.scalar_type() == at::kBFloat16,
              "group_softdelta gate supports only fp16 and bf16 gates");
  TORCH_CHECK(dy.scalar_type() == gate.scalar_type() &&
                  (readers.scalar_type() == gate.scalar_type() || readers.scalar_type() == at::kFloat),
              "group_softdelta readers gate BWD dtypes are inconsistent");
  const auto B = dy.size(0), L = dy.size(1), H = dy.size(2);
  const auto C = dy.size(3), Dg = dy.size(4);
  const auto logical_heads = readers.size(2);
  const auto Hkv = logical_heads - H;
  TORCH_CHECK(readers.size(0) == B && readers.size(1) == L && readers.size(3) == C && readers.size(4) == Dg &&
                  Hkv > 0 && H % Hkv == 0 && gate.sizes() == torch::IntArrayRef({B, L, H, C, 1}),
              "group_softdelta readers gate BWD shapes are inconsistent");
  const c10::cuda::CUDAGuard guard(gate.device());
  auto neg_read_grad = torch::empty({B, L, Hkv, C, Dg}, gate.options());
  auto gate_grad = torch::empty(gate.sizes(), gate.options());
  const int group_dim = int(Dg);
  const int64_t row = C * Dg;
  const GateShape shape{neg_read_grad.numel() / group_dim, int(C), int(Hkv), int(H / Hkv), group_dim,
                        ReadStrides{0, 0, 0, logical_heads * row, row}};
  const bool aligned = row % 8 == 0 && Aligned16(dy) && Aligned16(readers) && Aligned16(neg_read_grad);
  const bool fp32_state = readers.scalar_type() == at::kFloat;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (gate.scalar_type() == at::kHalf) {
    DispatchState<half, false>(fp32_state, group_dim, aligned, dy, readers, gate,
                               neg_read_grad, gate_grad, shape, stream);
  } else {
    DispatchState<nv_bfloat16, false>(fp32_state, group_dim, aligned, dy, readers, gate,
                                      neg_read_grad, gate_grad, shape, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {neg_read_grad, gate_grad};
}

}  // namespace ops
}  // namespace xattn

// Author: Shicheng Wen

#include "softdelta/gate.h"

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

// Row strides (elements): read row (bt, kv, r) starts at bt * read_bt + kv * read_kv +
// r * read_r, negative read row (bt, kv) at bt * neg_bt + kv * neg_kv and output row
// (bt, h) at bt * out_bt + h * C * Dg; value group c adds c * Dg. Contiguous tensors and
// head slices of the forward's class-major [B, L, Hkv + H, C, Dg] readers both fit.
struct GateStrides {
  int64_t read_bt, read_kv, read_r, neg_bt, neg_kv, out_bt;
};

// output = read - sign * sigmoid(gate) * neg_read: sign 1 gates the read, -1 restores
// it from the gated output. The output may alias the read (each element is read
// before the same thread writes it), so neither pointer is restrict.
// One kSubwarp-lane group per gate row (b, t, h, c). Lane l owns the kVec-wide
// chunks l, l + kSubwarp, ... of the Dg columns (kChunks of them at most).
template <typename T, typename StateT, int kVec, int kSubwarp, int kChunks>
__global__ void SoftDeltaGateFwdKernel(
    const StateT* read, const StateT* __restrict__ neg_read,
    const T* __restrict__ gate, T* output, float sign,
    int64_t rows, int groups, int q_heads, int rep, int group_dim, GateStrides strides) {
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
  const int64_t out_offset = bt * strides.out_bt + (int64_t(h) * groups + c) * group_dim;
  const float gate_value = sign * Sigmoid(static_cast<float>(gate[row]));
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int column = (chunk * kSubwarp + sublane) * kVec;
    if (column < group_dim) {
      float a[kVec], y[kVec];
      LoadVec<StateT, kVec>(read + read_offset + column, a);
      LoadVec<StateT, kVec>(neg_read + neg_offset + column, y);
#pragma unroll
      for (int i = 0; i < kVec; ++i) a[i] -= gate_value * y[i];
      StoreVec<T, kVec>(output + out_offset + column, a);
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
// (guarded loads become branches interleaved with their uses). The output may
// alias the read, as in SoftDeltaGateFwdKernel.
template <typename T, typename StateT, int kLanes, bool kLooped>
__global__ void __launch_bounds__(kThreads) SoftDeltaGateFwdGroupKernel(
    const StateT* read, const StateT* __restrict__ neg_read, const T* __restrict__ gate,
    T* output, float sign, int items, cutlass::FastDivmod kv_divmod, int rep, int value_dim, int groups,
    int group_dim, GateStrides strides) {
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
  T* out_row = output + bt * strides.out_bt + int64_t(kv) * rep * value_dim + col;
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
      const float s = sign * Sigmoid(static_cast<float>(logits[c]));
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
void LaunchFwdGroup(const void* read, const void* neg_read, const void* gate, void* output, float sign,
                    int64_t items, int kv_heads, int rep, int value_dim, int groups, int group_dim,
                    const GateStrides& strides, cudaStream_t stream) {
  const int64_t blocks = (items * kLanes + kThreads - 1) / kThreads;
  TORCH_CHECK(items <= std::numeric_limits<int>::max() && blocks <= std::numeric_limits<int>::max(),
              "softdelta gate: batch x length x KV heads exceeds the 32-bit item index");
  SoftDeltaGateFwdGroupKernel<T, StateT, kLanes, kLooped><<<int(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const StateT*>(read), reinterpret_cast<const StateT*>(neg_read),
      reinterpret_cast<const T*>(gate), reinterpret_cast<T*>(output), sign, int(items),
      cutlass::FastDivmod(kv_heads), rep, value_dim, groups, group_dim, strides);
}

// The grouped forward when V is a multiple of 8 within 256 and value groups are
// whole vectors; returns false otherwise (per-row kernel).
template <typename T, typename StateT>
bool TryLaunchFwdGroup(const void* read, const void* neg_read, const void* gate, void* output, float sign,
                       int64_t items, int kv_heads, int rep, int value_dim, int groups, int group_dim,
                       const GateStrides& strides, cudaStream_t stream) {
  if (value_dim % 8 != 0 || value_dim > 256 || group_dim % 8 != 0) {
    return false;
  }
  const bool looped = rep > 4;
  auto run = [&](auto lanes_tag) {
    constexpr int kLanes = decltype(lanes_tag)::value;
    if (looped) {
      LaunchFwdGroup<T, StateT, kLanes, true>(read, neg_read, gate, output, sign, items, kv_heads, rep,
                                              value_dim, groups, group_dim, strides, stream);
    } else {
      LaunchFwdGroup<T, StateT, kLanes, false>(read, neg_read, gate, output, sign, items, kv_heads, rep,
                                               value_dim, groups, group_dim, strides, stream);
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
__global__ void SoftDeltaGateBwdKernel(
    const T* __restrict__ dy, const StateT* __restrict__ neg_read,
    const T* __restrict__ gate, T* __restrict__ neg_read_grad,
    T* __restrict__ gate_grad,
    int64_t kv_rows, int groups, int kv_heads, int rep, int group_dim, GateStrides strides) {
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

// out[i] = acc[i] + other[i] (+ extra[row - extra_row_offset] on rows >= extra_row_offset), 8 elements
// per thread when aligned; acc is out (in place) or an FP32 accumulator. Rows are the per-batch time
// index; row_elems = Hkv * D.
template <typename T, typename AccT, int kVec>
__global__ void SoftDeltaMergeKernel(
    T* out, const AccT* acc, const T* __restrict__ other, const T* __restrict__ extra,
    int64_t elems, int64_t row_elems, int rows, int extra_rows) {
  const int64_t e = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * kVec;
  if (e >= elems) {
    return;
  }
  float a[kVec], b[kVec];
  LoadVec<AccT, kVec>(acc + e, a);
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
  GateStrides strides;
};

template <typename T, typename StateT, int kVec, int kSubwarp, int kChunks>
void LaunchFwd(const torch::Tensor& read, const torch::Tensor& neg_read, const torch::Tensor& gate,
               torch::Tensor& output, float sign, const GateShape& shape, cudaStream_t stream) {
  const int64_t threads = shape.rows * kSubwarp;
  SoftDeltaGateFwdKernel<T, StateT, kVec, kSubwarp, kChunks>
      <<<int((threads + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
          reinterpret_cast<const StateT*>(read.data_ptr()),
          reinterpret_cast<const StateT*>(neg_read.data_ptr()),
          reinterpret_cast<const T*>(gate.data_ptr()),
          reinterpret_cast<T*>(output.data_ptr()), sign,
          shape.rows, shape.groups, shape.heads, shape.rep, shape.group_dim, shape.strides);
}

template <typename T, typename StateT, int kVec, int kSubwarp, int kChunks>
void LaunchBwd(const torch::Tensor& dy, const torch::Tensor& neg_read, const torch::Tensor& gate,
               torch::Tensor& neg_read_grad, torch::Tensor& gate_grad, const GateShape& shape,
               cudaStream_t stream) {
  const int64_t threads = shape.rows * kSubwarp;
  SoftDeltaGateBwdKernel<T, StateT, kVec, kSubwarp, kChunks>
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
    TORCH_CHECK(group_dim <= 256, "softdelta gate group dimension must be <= 256");
    Run<1, 32, 8>(args...);
  }
};

template <typename T, bool kForward, typename... Args>
void DispatchState(bool fp32_state, int group_dim, bool aligned, Args&&... args) {
  if (fp32_state) {
    GateDispatch<T, float, kForward>::Dispatch(group_dim, aligned, args...);
  } else {
    GateDispatch<T, T, kForward>::Dispatch(group_dim, aligned, args...);
  }
}

// A [B, L, X, C, Dg] gate operand with unit column stride whose X heads of C * Dg
// elements are packed and whose B and L dimensions merge: a contiguous tensor or a
// head slice of the readers. Returns its (b, t) row stride in elements.
int64_t RowStride(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 5, "softdelta gate ", name, " must be a 5D CUDA tensor");
  const int64_t head = t.size(3) * t.size(4);
  TORCH_CHECK(t.stride(4) == 1 && (t.size(3) == 1 || t.stride(3) == t.size(4)) &&
                  (t.size(2) == 1 || t.stride(2) == head) &&
                  (t.size(0) == 1 || t.size(1) == 1 || t.stride(0) == t.size(1) * t.stride(1)),
              "softdelta gate ", name, " must pack its heads and merge batch and length");
  if (t.size(1) > 1) {
    return t.stride(1);
  }
  return t.size(0) > 1 ? t.stride(0) : t.size(2) * head;
}

bool RowsAligned16(const torch::Tensor& t, int64_t row_stride) {
  return Aligned16(t) && row_stride * t.element_size() % 16 == 0 &&
         t.size(3) * t.size(4) * t.element_size() % 16 == 0;
}

void CheckGate(const torch::Tensor& rows, const torch::Tensor& neg_read, const torch::Tensor& gate) {
  TORCH_CHECK(gate.is_cuda() && gate.dim() == 5 && gate.is_contiguous(),
              "softdelta gate must be a contiguous 5D CUDA tensor");
  TORCH_CHECK(gate.scalar_type() == at::kHalf || gate.scalar_type() == at::kBFloat16,
              "softdelta gate supports only fp16 and bf16 gates");
  TORCH_CHECK(rows.device() == gate.device() && neg_read.device() == gate.device(),
              "softdelta gate tensors must share a device");
  TORCH_CHECK(neg_read.scalar_type() == gate.scalar_type() || neg_read.scalar_type() == at::kFloat,
              "softdelta neg_read must use the gate dtype or FP32");
  const auto B = rows.size(0), L = rows.size(1), H = rows.size(2);
  const auto C = rows.size(3), Dg = rows.size(4);
  TORCH_CHECK(gate.size(0) == B && gate.size(1) == L && gate.size(2) == H &&
                  gate.size(3) == C && gate.size(4) == 1,
              "softdelta gate shape must be [B, L, H, C, 1]");
  const auto Hkv = neg_read.size(2);
  TORCH_CHECK(neg_read.size(0) == B && neg_read.size(1) == L && neg_read.size(3) == C &&
                  neg_read.size(4) == Dg && Hkv > 0 && H % Hkv == 0,
              "softdelta neg_read must be [B, L, Hkv, C, Dg] with H % Hkv == 0");
  TORCH_CHECK(rows.numel() > 0, "softdelta gate operands must be nonempty");
}

// output = read - sign * sigmoid(gate) * neg_read over strided operands; see GateStrides.
void GateCombine(const torch::Tensor& read, const torch::Tensor& neg_read, const torch::Tensor& gate,
                 torch::Tensor& output, float sign) {
  CheckGate(read, neg_read, gate);
  TORCH_CHECK(read.scalar_type() == neg_read.scalar_type(),
              "softdelta read and neg_read must share a dtype");
  TORCH_CHECK(output.scalar_type() == gate.scalar_type() && output.sizes() == read.sizes() &&
                  output.device() == gate.device(),
              "softdelta gate output must be [B, L, H, C, Dg] in the gate dtype");
  const int64_t read_bt = RowStride(read, "read");
  const int64_t neg_bt = RowStride(neg_read, "neg_read");
  const int64_t out_bt = RowStride(output, "output");
  const int group_dim = int(read.size(4));
  const int groups = int(read.size(3));
  const int64_t row = int64_t(groups) * group_dim;
  const int kv_heads = int(neg_read.size(2));
  const int rep = int(read.size(2)) / kv_heads;
  const GateShape shape{read.numel() / group_dim, groups, int(read.size(2)), rep, group_dim,
                        GateStrides{read_bt, rep * row, row, neg_bt, row, out_bt}};
  const bool aligned = RowsAligned16(read, read_bt) && RowsAligned16(neg_read, neg_bt) &&
                       RowsAligned16(output, out_bt);
  const bool fp32_state = read.scalar_type() == at::kFloat;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t items = read.size(0) * read.size(1) * kv_heads;
  bool done = false;
  if (aligned) {
    auto group = [&](auto t_tag, auto state_tag) {
      using T = decltype(t_tag);
      using StateT = decltype(state_tag);
      return TryLaunchFwdGroup<T, StateT>(read.data_ptr(), neg_read.data_ptr(), gate.data_ptr(),
                                          output.data_ptr(), sign, items, kv_heads, rep, int(row), groups,
                                          group_dim, shape.strides, stream);
    };
    if (gate.scalar_type() == at::kHalf) {
      done = fp32_state ? group(half{}, float{}) : group(half{}, half{});
    } else {
      done = fp32_state ? group(nv_bfloat16{}, float{}) : group(nv_bfloat16{}, nv_bfloat16{});
    }
  }
  if (!done) {
    if (gate.scalar_type() == at::kHalf) {
      DispatchState<half, true>(fp32_state, group_dim, aligned, read, neg_read, gate, output, sign, shape,
                                stream);
    } else {
      DispatchState<nv_bfloat16, true>(fp32_state, group_dim, aligned, read, neg_read, gate, output, sign,
                                       shape, stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, typename AccT>
void LaunchMerge(torch::Tensor& out, const torch::Tensor& acc, const torch::Tensor& other,
                 const c10::optional<torch::Tensor>& extra, cudaStream_t stream) {
  const int64_t elems = out.numel();
  const int rows = int(out.size(1));
  const int64_t row_elems = elems / (out.size(0) * rows);
  const T* extra_ptr = extra.has_value() ? reinterpret_cast<const T*>(extra->data_ptr()) : nullptr;
  const int extra_rows = extra.has_value() ? int(extra->size(1)) : 0;
  const bool aligned = row_elems % 8 == 0 && Aligned16(out) && Aligned16(acc) && Aligned16(other) &&
                       (!extra.has_value() || Aligned16(*extra));
  if (aligned) {
    const int64_t threads = elems / 8;
    SoftDeltaMergeKernel<T, AccT, 8><<<int((threads + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        reinterpret_cast<T*>(out.data_ptr()), reinterpret_cast<const AccT*>(acc.data_ptr()),
        reinterpret_cast<const T*>(other.data_ptr()), extra_ptr, elems, row_elems, rows, extra_rows);
  } else {
    SoftDeltaMergeKernel<T, AccT, 1><<<int((elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        reinterpret_cast<T*>(out.data_ptr()), reinterpret_cast<const AccT*>(acc.data_ptr()),
        reinterpret_cast<const T*>(other.data_ptr()), extra_ptr, elems, row_elems, rows, extra_rows);
  }
}

// out = acc + other (+ extra); acc is out itself or an FP32 accumulator of out's shape.
void Merge(torch::Tensor& out, const torch::Tensor& acc, const torch::Tensor& other,
           const c10::optional<torch::Tensor>& extra, cudaStream_t stream) {
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && acc.is_contiguous() && other.is_contiguous() &&
                  out.sizes() == other.sizes() && acc.sizes() == out.sizes() &&
                  out.scalar_type() == other.scalar_type() &&
                  (acc.scalar_type() == out.scalar_type() || acc.scalar_type() == at::kFloat) && out.dim() >= 3,
              "softdelta gradient merge expects matching contiguous CUDA tensors");
  TORCH_CHECK(out.scalar_type() == at::kHalf || out.scalar_type() == at::kBFloat16,
              "softdelta gradient merge supports only fp16 and bf16");
  if (extra.has_value()) {
    TORCH_CHECK(extra->is_contiguous() && extra->scalar_type() == out.scalar_type() &&
                    extra->dim() == out.dim() && extra->size(0) == out.size(0) &&
                    extra->size(1) <= out.size(1) &&
                    extra->numel() / extra->size(1) == out.numel() / out.size(1),
                "softdelta negative-query gradient must be [B, Lq, Hkv, D] with Lq <= Lk");
  }
  if (out.numel() == 0) {
    return;
  }
  const bool fp32 = acc.scalar_type() == at::kFloat;
  if (out.scalar_type() == at::kHalf) {
    fp32 ? LaunchMerge<half, float>(out, acc, other, extra, stream)
         : LaunchMerge<half, half>(out, acc, other, extra, stream);
  } else {
    fp32 ? LaunchMerge<nv_bfloat16, float>(out, acc, other, extra, stream)
         : LaunchMerge<nv_bfloat16, nv_bfloat16>(out, acc, other, extra, stream);
  }
}

}  // namespace

torch::Tensor SoftDeltaGateFwd(const torch::Tensor& read, const torch::Tensor& neg_read,
                               const torch::Tensor& gate, const c10::optional<torch::Tensor>& output) {
  const c10::cuda::CUDAGuard guard(gate.device());
  torch::Tensor y = output.has_value() ? output.value() : torch::empty(read.sizes(), gate.options());
  GateCombine(read, neg_read, gate, y, 1.0f);
  return y;
}

torch::Tensor SoftDeltaGateRestore(const torch::Tensor& y, const torch::Tensor& neg_read,
                                   const torch::Tensor& gate) {
  const c10::cuda::CUDAGuard guard(gate.device());
  torch::Tensor read = torch::empty(y.sizes(), gate.options());
  GateCombine(y, neg_read, gate, read, -1.0f);
  return read;
}

std::tuple<torch::Tensor, torch::Tensor> SoftDeltaGateBwd(
    const torch::Tensor& dy, const torch::Tensor& neg_read, const torch::Tensor& gate) {
  CheckGate(dy, neg_read, gate);
  TORCH_CHECK(dy.is_contiguous() && dy.scalar_type() == gate.scalar_type(),
              "softdelta dy must be contiguous in the gate dtype");
  const c10::cuda::CUDAGuard guard(dy.device());
  const int64_t neg_bt = RowStride(neg_read, "neg_read");
  auto neg_read_grad = torch::empty(neg_read.sizes(), gate.options());
  auto gate_grad = torch::empty(gate.sizes(), gate.options());
  const int group_dim = int(dy.size(4));
  const int kv_heads = int(neg_read.size(2));
  const int64_t row = dy.size(3) * group_dim;
  const GateShape shape{neg_read_grad.numel() / group_dim, int(dy.size(3)), kv_heads,
                        int(dy.size(2)) / kv_heads, group_dim,
                        GateStrides{0, 0, 0, neg_bt, row, 0}};
  const bool aligned = Aligned16(dy) && RowsAligned16(neg_read, neg_bt) && Aligned16(neg_read_grad);
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

std::tuple<torch::Tensor, torch::Tensor> SoftDeltaMergeGradients(
    torch::Tensor read_key_grad, const torch::Tensor& neg_read_key_grad,
    const c10::optional<torch::Tensor>& dneg_q, torch::Tensor read_value_grad,
    const torch::Tensor& neg_read_value_grad) {
  const c10::cuda::CUDAGuard guard(read_key_grad.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto merge = [&](torch::Tensor& read_grad, const torch::Tensor& neg_read_grad,
                   const c10::optional<torch::Tensor>& extra) {
    if (read_grad.scalar_type() != at::kFloat) {
      Merge(read_grad, read_grad, neg_read_grad, extra, stream);
      return read_grad;
    }
    torch::Tensor out = torch::empty_like(neg_read_grad);
    Merge(out, read_grad.contiguous(), neg_read_grad, extra, stream);
    return out;
  };
  torch::Tensor key_grad = merge(read_key_grad, neg_read_key_grad, dneg_q);
  torch::Tensor value_grad = merge(read_value_grad, neg_read_value_grad, c10::nullopt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {key_grad, value_grad};
}

}  // namespace ops
}  // namespace xattn

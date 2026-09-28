#include <ATen/DeviceGuard.h>
#include <c10/core/MemoryFormat.h>
#include <c10/cuda/CUDAGuard.h>

#include <tuple>

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>
#include <vector_types.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "attention/partition/segment_index.h"
#include "cuda_utils.cuh"
#include "flash_sca/sliding_chunk_attention.h"
#include "reduce.cuh"

namespace xattn {
namespace ops {

namespace {

template <typename T>
__device__ __forceinline__ float FlashSCAToFloat(T x) {
  return static_cast<float>(x);
}

template <typename T>
struct FlashSCACutlassElement;

template <>
struct FlashSCACutlassElement<at::Half> {
  using Type = cutlass::half_t;
};

template <>
struct FlashSCACutlassElement<at::BFloat16> {
  using Type = cutlass::bfloat16_t;
};

template <typename T>
__global__ void FlashSCADeltaKernel(
    int64_t L, int64_t H, int64_t V, const T* __restrict__ dy,
    const T* __restrict__ y, float* __restrict__ delta);

constexpr int64_t kFlashSCAThreads = 128;
constexpr int64_t kFlashSCAHalfThreads = 64;
constexpr int64_t kFlashSCADeltaRowsPerBlock = 8;
constexpr int64_t kFlashSCADeltaThreads =
    kFlashSCADeltaRowsPerBlock * cuda_utils::kWarpSize;

template <typename Element>
struct FlashSCAMmaOp;

template <>
struct FlashSCAMmaOp<cutlass::half_t> {
  using Type = cute::SM80_16x8x16_F32F16F16F32_TN;
};

template <>
struct FlashSCAMmaOp<cutlass::bfloat16_t> {
  using Type = cute::SM80_16x8x16_F32BF16BF16F32_TN;
};

template <typename T>
__device__ __forceinline__ T* FlashSCAAllocSmem(char*& smem, int64_t count) {
  constexpr uintptr_t kAlignment =
      alignof(T) > 16 ? uintptr_t(alignof(T)) : uintptr_t(16);
  uintptr_t addr = reinterpret_cast<uintptr_t>(smem);
  addr = (addr + kAlignment - 1) & ~(kAlignment - 1);
  T* ptr = reinterpret_cast<T*>(addr);
  smem = reinterpret_cast<char*>(ptr + count);
  return ptr;
}

template <typename Element, int M, int K>
__device__ __forceinline__ auto FlashSCARowMajorTensor(Element* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(ptr),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::Int<1>{})));
}

template <int BM>
constexpr int64_t FlashSCAThreadsForBM() {
  return BM == 32 ? kFlashSCAHalfThreads : kFlashSCAThreads;
}

template <typename Element, int BM>
__device__ __forceinline__ auto FlashSCATiledMmaQK() {
  using MmaAtom = cute::MMA_Atom<typename FlashSCAMmaOp<Element>::Type>;
  if constexpr (BM == 32) {
    return cute::make_tiled_mma(
        MmaAtom{},
        cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>{});
  } else {
    return cute::make_tiled_mma(
        MmaAtom{},
        cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>{});
  }
}

template <typename Element, int BM, int BN>
__device__ __forceinline__ auto FlashSCATiledMmaDKV() {
  using MmaAtom = cute::MMA_Atom<typename FlashSCAMmaOp<Element>::Type>;
  if constexpr (BM == 32) {
    if constexpr (BN % 32 == 0) {
      return cute::make_tiled_mma(
          MmaAtom{},
          cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>{});
    } else {
      return cute::make_tiled_mma(
          MmaAtom{},
          cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_1>>{});
    }
  } else {
    if constexpr (BN % 32 == 0) {
      return cute::make_tiled_mma(
          MmaAtom{},
          cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{});
    } else {
      return cute::make_tiled_mma(
          MmaAtom{},
          cute::Layout<cute::Shape<cute::_1, cute::_4, cute::_1>>{});
    }
  }
}

template <typename Element, bool Transposed>
struct FlashSCASmemCopyAtom {
  using Type = cute::Copy_Atom<cute::DefaultCopy, Element>;
};

template <typename Element>
struct FlashSCASmemCopyAtom<Element, true> {
  using Type = cute::Copy_Atom<cute::DefaultCopy, Element>;
};

template <class Layout>
__forceinline__ __device__ auto FlashSCARegConvertLayoutAccRowCol(Layout acc_layout) {
  static_assert(decltype(cute::size<0>(acc_layout))::value == 2 ||
                decltype(cute::size<0>(acc_layout))::value == 4);
  static_assert(decltype(cute::rank(acc_layout))::value == 3);
  auto l = cute::logical_divide(acc_layout, cute::Shape<cute::_2>{});
  return cute::make_layout(
      cute::make_layout(cute::get<0, 1>(l), cute::get<1>(l)),
      cute::make_layout(cute::get<0, 0>(l), cute::get<2>(l)));
}

template <typename TiledMma, class Layout>
__forceinline__ __device__ auto FlashSCARegConvertLayoutAccAregs(Layout acc_layout) {
  static_assert(decltype(cute::size<0>(acc_layout))::value == 2 ||
                decltype(cute::size<0>(acc_layout))::value == 4);
  static_assert(decltype(cute::rank(acc_layout))::value == 3);
  constexpr int mma_shape_k =
      cute::get<2>(typename TiledMma::AtomShape_MNK{});
  static_assert(mma_shape_k == 8 || mma_shape_k == 16);
  if constexpr (mma_shape_k == 8) {
    return acc_layout;
  } else {
    using X = cute::Underscore;
    auto l = cute::logical_divide(acc_layout, cute::Shape<X, X, cute::_2>{});
    return cute::make_layout(
        cute::make_layout(cute::get<0>(l), cute::get<2, 0>(l)),
        cute::get<1>(l), cute::get<2, 1>(l));
  }
}

template <int Threads>
struct FlashSCARegAllreduce {
  static_assert(Threads == 32 || Threads == 16 || Threads == 8 ||
                Threads == 4 || Threads == 2);
  template <typename T, typename Op>
  static __device__ __forceinline__ T Run(T x, Op op) {
    constexpr int offset = Threads / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, offset));
    return FlashSCARegAllreduce<offset>::Run(x, op);
  }
};

template <>
struct FlashSCARegAllreduce<1> {
  template <typename T, typename Op>
  static __device__ __forceinline__ T Run(T x, Op) {
    return x;
  }
};

struct FlashSCARegMaxOp {
  __device__ __forceinline__ float operator()(float x, float y) const {
    return fmaxf(x, y);
  }
};

struct FlashSCARegSumOp {
  __device__ __forceinline__ float operator()(float x, float y) const {
    return x + y;
  }
};

template <class Tensor0, class Tensor1, typename Op>
__device__ __forceinline__ void FlashSCARegThreadReduce(
    Tensor0 const& tensor, Tensor1& summary, Op op, bool zero_init) {
  static_assert(Tensor0::layout_type::rank == 2, "tensor must be 2D");
  static_assert(Tensor1::layout_type::rank == 1, "summary must be 1D");
  CUTE_STATIC_ASSERT_V(cute::size<0>(summary) == cute::size<0>(tensor));
  #pragma unroll
  for (int mi = 0; mi < cute::size<0>(tensor); ++mi) {
    summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
    #pragma unroll
    for (int ni = 1; ni < cute::size<1>(tensor); ++ni) {
      summary(mi) = op(summary(mi), tensor(mi, ni));
    }
  }
}

template <class Tensor>
__device__ __forceinline__ void FlashSCARegQuadAllreduceSum(Tensor& tensor) {
  FlashSCARegSumOp op;
  #pragma unroll
  for (int i = 0; i < cute::size(tensor); ++i) {
    tensor(i) = FlashSCARegAllreduce<4>::Run(tensor(i), op);
  }
}

template <class Tensor>
__device__ __forceinline__ void FlashSCARegQuadAllreduceMax(Tensor& tensor) {
  FlashSCARegMaxOp op;
  #pragma unroll
  for (int i = 0; i < cute::size(tensor); ++i) {
    tensor(i) = FlashSCARegAllreduce<4>::Run(tensor(i), op);
  }
}

template <class Tensor0, class Tensor1>
__device__ __forceinline__ void FlashSCARegReduceMax(
    Tensor0 const& tensor, Tensor1& max_frag, bool zero_init) {
  FlashSCARegMaxOp op;
  FlashSCARegThreadReduce(tensor, max_frag, op, zero_init);
  FlashSCARegQuadAllreduceMax(max_frag);
}

template <class Tensor0, class Tensor1>
__device__ __forceinline__ void FlashSCARegReduceSum(
    Tensor0 const& tensor, Tensor1& sum_frag, bool zero_init) {
  FlashSCARegSumOp op;
  FlashSCARegThreadReduce(tensor, sum_frag, op, zero_init);
}

template <class Tensor0, class Tensor1>
__device__ __forceinline__ void FlashSCARegScaleApplyExp2(
    Tensor0& tensor, Tensor1 const& max_frag, float scale_log2) {
  static_assert(Tensor0::layout_type::rank == 2, "tensor must be 2D");
  static_assert(Tensor1::layout_type::rank == 1, "max must be 1D");
  CUTE_STATIC_ASSERT_V(cute::size<0>(max_frag) == cute::size<0>(tensor));
  #pragma unroll
  for (int mi = 0; mi < cute::size<0>(tensor); ++mi) {
    const float max_scaled =
        max_frag(mi) == -INFINITY ? 0.0f : max_frag(mi) * scale_log2;
    #pragma unroll
    for (int ni = 0; ni < cute::size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale_log2 - max_scaled);
    }
  }
}

template <int kNRows>
struct FlashSCARegSoftmax {
  decltype(cute::make_tensor<float>(cute::Int<kNRows>{})) row_max;
  decltype(cute::make_tensor<float>(cute::Int<kNRows>{})) row_sum;

  __device__ __forceinline__ FlashSCARegSoftmax() {}

  template <bool IsFirst, class TensorS, class TensorO>
  __device__ __forceinline__ void SoftmaxRescaleO(
      TensorS& acc_s, TensorO& acc_o, float scale_log2) {
    auto scores =
        cute::make_tensor(acc_s.data(), FlashSCARegConvertLayoutAccRowCol(acc_s.layout()));
    static_assert(decltype(cute::size<0>(scores))::value == kNRows);
    if constexpr (IsFirst) {
      FlashSCARegReduceMax(scores, row_max, true);
      FlashSCARegScaleApplyExp2(scores, row_max, scale_log2);
      FlashSCARegReduceSum(scores, row_sum, true);
    } else {
      auto row_max_prev = cute::make_fragment_like(row_max);
      cute::copy(row_max, row_max_prev);
      FlashSCARegReduceMax(scores, row_max, false);
      auto acc_o_rowcol =
          cute::make_tensor(acc_o.data(), FlashSCARegConvertLayoutAccRowCol(acc_o.layout()));
      static_assert(decltype(cute::size<0>(acc_o_rowcol))::value == kNRows);
      #pragma unroll
      for (int mi = 0; mi < cute::size(row_max); ++mi) {
        const float cur = row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
        const float old_scale = exp2f((row_max_prev(mi) - cur) * scale_log2);
        row_sum(mi) *= old_scale;
        #pragma unroll
        for (int ni = 0; ni < cute::size<1>(acc_o_rowcol); ++ni) {
          acc_o_rowcol(mi, ni) *= old_scale;
        }
      }
      FlashSCARegScaleApplyExp2(scores, row_max, scale_log2);
      FlashSCARegReduceSum(scores, row_sum, false);
    }
  }

  template <class TensorO>
  __device__ __forceinline__ auto Normalize(TensorO& acc_o, float scale) {
    FlashSCARegQuadAllreduceSum(row_sum);
    auto lse = cute::make_fragment_like(row_sum);
    auto acc_o_rowcol =
        cute::make_tensor(acc_o.data(), FlashSCARegConvertLayoutAccRowCol(acc_o.layout()));
    static_assert(decltype(cute::size<0>(acc_o_rowcol))::value == kNRows);
    #pragma unroll
    for (int mi = 0; mi < cute::size<0>(acc_o_rowcol); ++mi) {
      const float sum = row_sum(mi);
      const float inv_sum = (sum == 0.0f || sum != sum) ? 1.0f : 1.0f / sum;
      lse(mi) = (sum == 0.0f || sum != sum)
                    ? INFINITY
                    : row_max(mi) * scale + __logf(sum);
      #pragma unroll
      for (int ni = 0; ni < cute::size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= inv_sum;
      }
    }
    return lse;
  }
};

template <typename To, class Tensor>
__device__ __forceinline__ auto FlashSCARegConvertType(Tensor const& tensor) {
  auto out = cute::make_tensor<To>(tensor.layout());
  #pragma unroll
  for (int i = 0; i < cute::size(tensor); ++i) {
    out(i) = static_cast<To>(tensor(i));
  }
  return out;
}

template <typename Element, bool TransposedA = false,
          bool TransposedB = false, class Acc, class TensorA, class TensorB,
          class TiledMma>
__device__ __forceinline__ void FlashSCARegGemmSS(
    Acc& acc, TensorA const& sA, TensorB const& sB, TiledMma tiled_mma) {
  const int tidx = threadIdx.x;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  auto tCrA = thr_mma.partition_fragment_A(sA);
  auto tCrB = thr_mma.partition_fragment_B(sB);
  auto smem_tiled_copy_A =
      cute::make_tiled_copy_A(
          typename FlashSCASmemCopyAtom<Element, TransposedA>::Type{},
          tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);
  auto tCsA = smem_thr_copy_A.partition_S(sA);
  auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  auto smem_tiled_copy_B =
      cute::make_tiled_copy_B(
          typename FlashSCASmemCopyAtom<Element, TransposedB>::Type{},
          tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tidx);
  auto tCsB = smem_thr_copy_B.partition_S(sB);
  auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  cute::copy(smem_tiled_copy_A, tCsA(cute::_, cute::_, cute::Int<0>{}),
             tCrA_copy_view(cute::_, cute::_, cute::Int<0>{}));
  cute::copy(smem_tiled_copy_B, tCsB(cute::_, cute::_, cute::Int<0>{}),
             tCrB_copy_view(cute::_, cute::_, cute::Int<0>{}));
  constexpr int kBlockMax = decltype(cute::size<2>(tCrA))::value;
  #pragma unroll
  for (int k_block = 0; k_block < kBlockMax; ++k_block) {
    if (k_block < kBlockMax - 1) {
      cute::copy(smem_tiled_copy_A, tCsA(cute::_, cute::_, k_block + 1),
                 tCrA_copy_view(cute::_, cute::_, k_block + 1));
      cute::copy(smem_tiled_copy_B, tCsB(cute::_, cute::_, k_block + 1),
                 tCrB_copy_view(cute::_, cute::_, k_block + 1));
    }
    cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block),
               tCrB(cute::_, cute::_, k_block), acc);
  }
}

template <typename Element, bool TransposedA = false, class TensorAReg,
          class TensorASmem,
          class TiledMma>
__device__ __forceinline__ void FlashSCARegCopySmemToRegA(
    TensorAReg& rA, TensorASmem const& sA, TiledMma tiled_mma) {
  const int tidx = threadIdx.x;
  auto smem_tiled_copy_A =
      cute::make_tiled_copy_A(
          typename FlashSCASmemCopyAtom<Element, TransposedA>::Type{},
          tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);
  auto tCsA = smem_thr_copy_A.partition_S(sA);
  auto tCrA_copy_view = smem_thr_copy_A.retile_D(rA);
  constexpr int kBlockMax = decltype(cute::size<2>(rA))::value;
  #pragma unroll
  for (int k_block = 0; k_block < kBlockMax; ++k_block) {
    cute::copy(smem_tiled_copy_A, tCsA(cute::_, cute::_, k_block),
               tCrA_copy_view(cute::_, cute::_, k_block));
  }
}

template <typename Element, bool TransposedB = false, class Acc, class TensorA,
          class TensorB,
          class TensorBSmem, class TiledMma>
__device__ __forceinline__ void FlashSCARegGemmRS(
    Acc& acc, TensorA const& rA, TensorB& rB, TensorBSmem const& sB,
    TiledMma tiled_mma) {
  const int tidx = threadIdx.x;
  auto smem_tiled_copy_B =
      cute::make_tiled_copy_B(
          typename FlashSCASmemCopyAtom<Element, TransposedB>::Type{},
          tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tidx);
  auto tCsB = smem_thr_copy_B.partition_S(sB);
  auto tCrB_copy_view = smem_thr_copy_B.retile_D(rB);
  cute::copy(smem_tiled_copy_B, tCsB(cute::_, cute::_, cute::Int<0>{}),
             tCrB_copy_view(cute::_, cute::_, cute::Int<0>{}));
  constexpr int kBlockMax = decltype(cute::size<2>(rA))::value;
  #pragma unroll
  for (int k_block = 0; k_block < kBlockMax; ++k_block) {
    if (k_block < kBlockMax - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(cute::_, cute::_, k_block + 1),
                 tCrB_copy_view(cute::_, cute::_, k_block + 1));
    }
    cute::gemm(tiled_mma, rA(cute::_, cute::_, k_block),
               rB(cute::_, cute::_, k_block), acc);
  }
}

__device__ __forceinline__ bool FlashSCASameSegment(
    bool has_segment, const int64_t* __restrict__ q_segment_idx,
    const int64_t* __restrict__ k_segment_idx, int64_t b, int64_t L,
    int64_t k_segment_len, int64_t q_pos, bool k_is_prev,
    int64_t k_pos) {
  if (!has_segment) {
    return true;
  }
  const int64_t k_current_offset = k_segment_len - L;
  const int64_t k_segment_pos = k_is_prev ? k_pos : k_current_offset + k_pos;
  const int64_t q_seg = q_segment_idx[b * L + q_pos];
  const int64_t k_seg = k_segment_idx[b * k_segment_len + k_segment_pos];
  return attention::partition::same_partition(q_seg, k_seg);
}

__device__ __forceinline__ bool FlashSCASegmentTilesMayOverlap(
    bool has_segment, const int64_t* __restrict__ q_segment_idx,
    const int64_t* __restrict__ k_segment_idx, int64_t b, int64_t L,
    int64_t C, int64_t k_segment_len, int64_t q_start, int64_t q_end,
    int64_t k_virtual_start, int64_t k_virtual_end, bool has_prev) {
  if (q_start >= q_end || k_virtual_start >= k_virtual_end) {
    return false;
  }
  if (!has_segment) {
    return true;
  }

  const int64_t q_min = q_segment_idx[b * L + q_start];
  const int64_t q_max = q_segment_idx[b * L + q_end - 1];
  const int64_t k_current_offset = k_segment_len - L;

  if (k_virtual_start < 0) {
    if (!has_prev || k_segment_idx == nullptr) {
      return false;
    }
    if (k_virtual_end > 0) {
      return true;
    }
    const int64_t k_begin = k_virtual_start + C;
    const int64_t k_end = k_virtual_end + C;
    if (k_begin < 0 || k_end > C || k_begin >= k_end) {
      return false;
    }
    const int64_t k_min = k_segment_idx[b * k_segment_len + k_begin];
    const int64_t k_max = k_segment_idx[b * k_segment_len + k_end - 1];
    return attention::partition::partition_ranges_may_overlap(
        q_min, q_max, k_min, k_max);
  }

  const int64_t k_begin = k_virtual_start;
  const int64_t k_end = k_virtual_end < L ? k_virtual_end : L;
  if (k_begin >= k_end) {
    return false;
  }
  const int64_t k_min =
      k_segment_idx[b * k_segment_len + k_current_offset + k_begin];
  const int64_t k_max =
      k_segment_idx[b * k_segment_len + k_current_offset + k_end - 1];
  return attention::partition::partition_ranges_may_overlap(
      q_min, q_max, k_min, k_max);
}

template <typename Element>
__device__ __forceinline__ Element FlashSCAFromFloat(float x) {
  return static_cast<Element>(x);
}

template <typename T, typename Element, int BM, int D>
__device__ __forceinline__ void FlashSCARegLoadQFull(
    int64_t L, int64_t H, int64_t q_start, int64_t b, int64_t h,
    int64_t d_actual, const T* __restrict__ q, Element* __restrict__ q_sm) {
  static_assert(D % 8 == 0, "D must be divisible by 8 for vector loads");
  const Element* q_e = reinterpret_cast<const Element*>(q);
  using Vec = uint4;
  constexpr int kVecElems = int(sizeof(Vec) / sizeof(Element));
  constexpr int kVecCols = D / kVecElems;
  Vec* q_sm_vec = reinterpret_cast<Vec*>(q_sm);
  for (int64_t idx = threadIdx.x; idx < int64_t(BM) * kVecCols;
       idx += blockDim.x) {
    const int64_t row = idx / kVecCols;
    const int64_t vec_col = idx - row * kVecCols;
    const int64_t q_pos = q_start + row;
    Vec val{};
    if (q_pos < L) {
      const Element* row_ptr =
          q_e + ((b * L + q_pos) * H + h) * d_actual;
      val = reinterpret_cast<const Vec*>(row_ptr)[vec_col];
    }
    q_sm_vec[idx] = val;
  }
}

template <typename T, typename Element, int BN, int D>
__device__ __forceinline__ void FlashSCARegLoadKFull(
    int64_t L, int64_t H, int64_t C, int64_t k_virtual_start, int64_t b,
    int64_t h, int64_t d_actual, bool has_prev, const T* __restrict__ k,
    const T* __restrict__ prev_k, Element* __restrict__ k_sm) {
  static_assert(D % 8 == 0, "D must be divisible by 8 for vector loads");
  const Element* k_e = reinterpret_cast<const Element*>(k);
  const Element* prev_k_e = reinterpret_cast<const Element*>(prev_k);
  using Vec = uint4;
  constexpr int kVecElems = int(sizeof(Vec) / sizeof(Element));
  constexpr int kVecCols = D / kVecElems;
  Vec* k_sm_vec = reinterpret_cast<Vec*>(k_sm);
  for (int64_t idx = threadIdx.x; idx < int64_t(BN) * kVecCols;
       idx += blockDim.x) {
    const int64_t row = idx / kVecCols;
    const int64_t vec_col = idx - row * kVecCols;
    const int64_t k_virtual = k_virtual_start + row;
    Vec val{};
    if (k_virtual < 0) {
      const int64_t prev_pos = k_virtual + C;
      if (has_prev && prev_pos >= 0 && prev_pos < C) {
        const Element* row_ptr =
            prev_k_e + ((b * C + prev_pos) * H + h) * d_actual;
        val = reinterpret_cast<const Vec*>(row_ptr)[vec_col];
      }
    } else if (k_virtual < L) {
      const Element* row_ptr =
          k_e + ((b * L + k_virtual) * H + h) * d_actual;
      val = reinterpret_cast<const Vec*>(row_ptr)[vec_col];
    }
    k_sm_vec[idx] = val;
  }
}

template <typename T, typename Element, int BM, int V>
__device__ __forceinline__ void FlashSCARegLoadDOFull(
    int64_t L, int64_t H, int64_t q_start, int64_t b, int64_t h,
    int64_t v_actual, const T* __restrict__ dy, Element* __restrict__ do_sm) {
  static_assert(V % 8 == 0, "V must be divisible by 8 for vector loads");
  const Element* dy_e = reinterpret_cast<const Element*>(dy);
  using Vec = uint4;
  constexpr int kVecElems = int(sizeof(Vec) / sizeof(Element));
  constexpr int kVecCols = V / kVecElems;
  Vec* do_sm_vec = reinterpret_cast<Vec*>(do_sm);
  for (int64_t idx = threadIdx.x; idx < int64_t(BM) * kVecCols;
       idx += blockDim.x) {
    const int64_t row = idx / kVecCols;
    const int64_t vec_col = idx - row * kVecCols;
    const int64_t q_pos = q_start + row;
    Vec val{};
    if (q_pos < L) {
      const Element* row_ptr =
          dy_e + ((b * L + q_pos) * H + h) * v_actual;
      val = reinterpret_cast<const Vec*>(row_ptr)[vec_col];
    }
    do_sm_vec[idx] = val;
  }
}

template <typename T, typename Element, int BN, int V>
__device__ __forceinline__ void FlashSCARegLoadVRowsFull(
    int64_t L, int64_t H, int64_t C, int64_t k_virtual_start, int64_t b,
    int64_t h, int64_t v_actual, bool has_prev, const T* __restrict__ v,
    const T* __restrict__ prev_v, Element* __restrict__ v_sm) {
  static_assert(V % 8 == 0, "V must be divisible by 8 for vector loads");
  const Element* v_e = reinterpret_cast<const Element*>(v);
  const Element* prev_v_e = reinterpret_cast<const Element*>(prev_v);
  using Vec = uint4;
  constexpr int kVecElems = int(sizeof(Vec) / sizeof(Element));
  constexpr int kVecCols = V / kVecElems;
  Vec* v_sm_vec = reinterpret_cast<Vec*>(v_sm);
  for (int64_t idx = threadIdx.x; idx < int64_t(BN) * kVecCols;
       idx += blockDim.x) {
    const int64_t row = idx / kVecCols;
    const int64_t vec_col = idx - row * kVecCols;
    const int64_t k_virtual = k_virtual_start + row;
    Vec val{};
    if (k_virtual < 0) {
      const int64_t prev_pos = k_virtual + C;
      if (has_prev && prev_pos >= 0 && prev_pos < C) {
        const Element* row_ptr =
            prev_v_e + ((b * C + prev_pos) * H + h) * v_actual;
        val = reinterpret_cast<const Vec*>(row_ptr)[vec_col];
      }
    } else if (k_virtual < L) {
      const Element* row_ptr =
          v_e + ((b * L + k_virtual) * H + h) * v_actual;
      val = reinterpret_cast<const Vec*>(row_ptr)[vec_col];
    }
    v_sm_vec[idx] = val;
  }
}

template <typename Element, int M, int N, int StrideM, int StrideN>
__device__ __forceinline__ auto FlashSCARegRowMajorStridedTensor(
    Element* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(ptr),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<N>{}),
                        cute::make_stride(cute::Int<StrideM>{},
                                          cute::Int<StrideN>{})));
}

template <int BM, int BN, class TensorS>
__device__ __forceinline__ void FlashSCARegApplyCausalChunkMask(
    TensorS& acc_s, int64_t L, int64_t C, int64_t chunk_end, int64_t q_start,
    int64_t k_virtual_start, int64_t b, bool has_prev, bool has_segment,
    const int64_t* __restrict__ q_segment_idx,
    const int64_t* __restrict__ k_segment_idx, int64_t k_segment_len) {
  auto scores =
      cute::make_tensor(acc_s.data(), FlashSCARegConvertLayoutAccRowCol(acc_s.layout()));
  const int lane_id = threadIdx.x & 31;
  const int row_idx_offset =
      int(q_start) + (threadIdx.x / 32) * 16 + (lane_id / 4);
  const int col_idx_offset = int(k_virtual_start) + (lane_id % 4) * 2;
  constexpr int kWarpRowStride =
      decltype(cute::size<0>(acc_s.layout()))::value * 16;
  #pragma unroll
  for (int mi_outer = 0; mi_outer < cute::size<0, 1>(scores); ++mi_outer) {
    const int row_base = row_idx_offset + mi_outer * kWarpRowStride;
    #pragma unroll
    for (int i = 0; i < cute::size<0, 0>(scores); ++i) {
      const int q_pos = row_base + i * 8;
      #pragma unroll
      for (int nj = 0; nj < cute::size<1, 1>(scores); ++nj) {
        const int col_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < cute::size<1, 0>(scores); ++j) {
          const int k_virtual = col_base + j;
          const bool key_valid =
              k_virtual < 0 ? has_prev && (k_virtual + C) >= 0 &&
                                   (k_virtual + C) < C
                             : k_virtual < L;
          bool same_segment = true;
          if (has_segment && q_pos < L && key_valid) {
            const bool k_is_prev = k_virtual < 0;
            const int64_t k_pos = k_is_prev ? k_virtual + C : k_virtual;
            same_segment = FlashSCASameSegment(
                true, q_segment_idx, k_segment_idx, b, L, k_segment_len,
                q_pos, k_is_prev, k_pos);
          }
          const bool valid = q_pos < L && q_pos < chunk_end && key_valid &&
                             k_virtual <= q_pos && same_segment;
          if (!valid) {
            scores(cute::make_coord(i, mi_outer),
                   cute::make_coord(j, nj)) = -INFINITY;
          }
        }
      }
    }
  }
}

template <class TensorS>
__device__ __forceinline__ void FlashSCARegApplySoftmaxFromLSE(
    TensorS& acc_s, int64_t L, int64_t H, int64_t q_start, int64_t b,
    int64_t h, float scale_log2, const float* __restrict__ lse) {
  auto scores =
      cute::make_tensor(acc_s.data(), FlashSCARegConvertLayoutAccRowCol(acc_s.layout()));
  const int lane_id = threadIdx.x & 31;
  const int row_idx_offset =
      int(q_start) + (threadIdx.x / 32) * 16 + (lane_id / 4);
  constexpr int kWarpRowStride =
      decltype(cute::size<0>(acc_s.layout()))::value * 16;
  #pragma unroll
  for (int mi_outer = 0; mi_outer < cute::size<0, 1>(scores); ++mi_outer) {
    const int row_base = row_idx_offset + mi_outer * kWarpRowStride;
    #pragma unroll
    for (int i = 0; i < cute::size<0, 0>(scores); ++i) {
      const int q_pos = row_base + i * 8;
      const float row_lse =
          q_pos < L ? lse[(b * H + h) * L + q_pos] : INFINITY;
      const float lse_log2 = row_lse * float(M_LOG2E);
      #pragma unroll
      for (int ni = 0; ni < cute::size<1>(scores); ++ni) {
        float s = scores(cute::make_coord(i, mi_outer), ni);
        scores(cute::make_coord(i, mi_outer), ni) =
            isinf(s) && s < 0.0f ? 0.0f : exp2f(s * scale_log2 - lse_log2);
      }
    }
  }
}

template <class TensorP, class TensorDP>
__device__ __forceinline__ void FlashSCARegMakeDS(
    TensorP& acc_p, TensorDP& acc_dp, int64_t L, int64_t H, int64_t q_start,
    int64_t b, int64_t h, float scale, const float* __restrict__ delta) {
  auto p = cute::make_tensor(acc_p.data(), FlashSCARegConvertLayoutAccRowCol(acc_p.layout()));
  auto dp =
      cute::make_tensor(acc_dp.data(), FlashSCARegConvertLayoutAccRowCol(acc_dp.layout()));
  const int lane_id = threadIdx.x & 31;
  const int row_idx_offset =
      int(q_start) + (threadIdx.x / 32) * 16 + (lane_id / 4);
  constexpr int kWarpRowStride =
      decltype(cute::size<0>(acc_dp.layout()))::value * 16;
  #pragma unroll
  for (int mi_outer = 0; mi_outer < cute::size<0, 1>(dp); ++mi_outer) {
    const int row_base = row_idx_offset + mi_outer * kWarpRowStride;
    #pragma unroll
    for (int i = 0; i < cute::size<0, 0>(dp); ++i) {
      const int q_pos = row_base + i * 8;
      const float d = q_pos < L ? delta[(b * H + h) * L + q_pos] : 0.0f;
      #pragma unroll
      for (int ni = 0; ni < cute::size<1>(dp); ++ni) {
        const auto coord = cute::make_coord(i, mi_outer);
        dp(coord, ni) = scale * p(coord, ni) * (dp(coord, ni) - d);
      }
    }
  }
}

template <typename Element, int BM, int BN, class TensorScore>
__device__ __forceinline__ void FlashSCARegStoreTransposedScoreTile(
    TensorScore& acc, int64_t q_start, int64_t k_virtual_start,
    Element* __restrict__ out_t_sm) {
  auto scores =
      cute::make_tensor(acc.data(), FlashSCARegConvertLayoutAccRowCol(acc.layout()));
  const int lane_id = threadIdx.x & 31;
  const int row_idx_offset =
      int(q_start) + (threadIdx.x / 32) * 16 + (lane_id / 4);
  const int col_idx_offset = int(k_virtual_start) + (lane_id % 4) * 2;
  constexpr int kWarpRowStride =
      decltype(cute::size<0>(acc.layout()))::value * 16;
  #pragma unroll
  for (int mi_outer = 0; mi_outer < cute::size<0, 1>(scores); ++mi_outer) {
    const int row_base = row_idx_offset + mi_outer * kWarpRowStride;
    #pragma unroll
    for (int i = 0; i < cute::size<0, 0>(scores); ++i) {
      const int row = row_base + i * 8 - int(q_start);
      #pragma unroll
      for (int nj = 0; nj < cute::size<1, 1>(scores); ++nj) {
        const int col_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < cute::size<1, 0>(scores); ++j) {
          const int col = col_base + j - int(k_virtual_start);
          if (row >= 0 && row < BM && col >= 0 && col < BN) {
            out_t_sm[col * BM + row] =
                FlashSCAFromFloat<Element>(
                    scores(cute::make_coord(i, mi_outer),
                           cute::make_coord(j, nj)));
          }
        }
      }
    }
  }
}

template <int BM, int D, class Acc, class TiledMma, typename Element>
__device__ __forceinline__ void FlashSCARegStoreDQ(
    Acc const& dq_frag, TiledMma tiled_mma, int64_t L, int64_t H,
    int64_t DActual, int64_t q_start, int64_t chunk_end, int64_t b,
    int64_t h, Element* __restrict__ dq) {
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  auto cDQ = cute::make_identity_tensor(
      cute::Shape<cute::Int<BM>, cute::Int<D>>{});
  auto taccDQcDQ = thr_mma.partition_C(cDQ);
  #pragma unroll
  for (int i = 0; i < cute::size(dq_frag); ++i) {
    const int row = cute::get<0>(taccDQcDQ(i));
    const int col = cute::get<1>(taccDQcDQ(i));
    const int64_t q_pos = q_start + row;
    if (q_pos < L && q_pos < chunk_end && col < DActual) {
      dq[((b * L + q_pos) * H + h) * DActual + col] = dq_frag(i);
    }
  }
}

template <int BN, int Dim, class Acc, class TiledMma, typename Element>
__device__ __forceinline__ void FlashSCARegStoreKVGrad(
    Acc const& grad_frag, TiledMma tiled_mma, int64_t L, int64_t H,
    int64_t DimActual, int64_t k_start, int64_t b, int64_t h,
    Element* __restrict__ grad) {
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  auto cGrad = cute::make_identity_tensor(
      cute::Shape<cute::Int<BN>, cute::Int<Dim>>{});
  auto taccGcG = thr_mma.partition_C(cGrad);
  #pragma unroll
  for (int i = 0; i < cute::size(grad_frag); ++i) {
    const int row = cute::get<0>(taccGcG(i));
    const int col = cute::get<1>(taccGcG(i));
    const int64_t k_pos = k_start + row;
    if (k_pos < L && col < DimActual) {
      grad[((b * L + k_pos) * H + h) * DimActual + col] = grad_frag(i);
    }
  }
}

template <int BM, int D, class Acc, class TiledMma>
__device__ __forceinline__ void FlashSCARegAtomicAddDQ(
    Acc const& acc_dq, TiledMma tiled_mma, int64_t L, int64_t H,
    int64_t DActual, int64_t q_start, int64_t chunk_end, int64_t b,
    int64_t h, float* __restrict__ dq_accum) {
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  auto cDQ = cute::make_identity_tensor(
      cute::Shape<cute::Int<BM>, cute::Int<D>>{});
  auto taccDQcDQ = thr_mma.partition_C(cDQ);
  #pragma unroll
  for (int i = 0; i < cute::size(acc_dq); ++i) {
    const int row = cute::get<0>(taccDQcDQ(i));
    const int col = cute::get<1>(taccDQcDQ(i));
    const int64_t q_pos = q_start + row;
    if (q_pos < L && q_pos < chunk_end && col < DActual) {
      atomicAdd(dq_accum + ((b * L + q_pos) * H + h) * DActual + col,
                acc_dq(i));
    }
  }
}

template <typename T, typename Element, int BM, int BN, int D, int V,
          int CStatic>
__global__ void FlashSCARegFwdKernel(
    int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t DActual,
    int64_t VActual,
    int64_t CArg, int64_t k_segment_len, bool has_prev, bool has_segment,
    float scale,
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    const T* __restrict__ prev_k, const T* __restrict__ prev_v,
    const int64_t* __restrict__ q_segment_idx,
    const int64_t* __restrict__ k_segment_idx,
    T* __restrict__ y, float* __restrict__ lse) {
  extern __shared__ char raw_smem[];
  char* smem = raw_smem;
  Element* q_sm = FlashSCAAllocSmem<Element>(smem, BM * D);
  Element* k_sm = FlashSCAAllocSmem<Element>(smem, BN * D);
  Element* v_sm = FlashSCAAllocSmem<Element>(smem, BN * V);
  Element* o_sm = FlashSCAAllocSmem<Element>(smem, BM * V);

  const int64_t C = CStatic > 0 ? int64_t(CStatic) : CArg;
  const int64_t q_block = blockIdx.x;
  const int64_t q_head = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t kv_head = q_head / (Hq / Hkv);
  const int64_t q_tiles_per_chunk = (C + BM - 1) / BM;
  const int64_t chunk_id = q_block / q_tiles_per_chunk;
  const int64_t q_tile_in_chunk = q_block - chunk_id * q_tiles_per_chunk;
  const int64_t chunk_start = chunk_id * C;
  const int64_t chunk_end = chunk_start + C;
  const int64_t q_start = chunk_start + q_tile_in_chunk * BM;
  const int64_t q_tile_end =
      (q_start + BM < chunk_end ? q_start + BM : chunk_end) < L
          ? (q_start + BM < chunk_end ? q_start + BM : chunk_end)
          : L;
  const int64_t window_start =
      (chunk_id == 0) ? (has_prev ? -C : 0) : (chunk_start - C);
  const int64_t window_end = q_tile_end;

  FlashSCARegLoadQFull<T, Element, BM, D>(
      L, Hq, q_start, b, q_head, DActual, q, q_sm);
  __syncthreads();

  auto tiled_mma = FlashSCATiledMmaQK<Element, BM>();
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  auto sQ = FlashSCARowMajorTensor<Element, BM, D>(q_sm);
  auto sK = FlashSCARowMajorTensor<Element, BN, D>(k_sm);
  auto sVt = FlashSCARegRowMajorStridedTensor<Element, V, BN, 1, V>(v_sm);
  auto sO = FlashSCARowMajorTensor<Element, BM, V>(o_sm);
  auto rQ = thr_mma.partition_fragment_A(sQ);
  FlashSCARegCopySmemToRegA<Element>(rQ, sQ, tiled_mma);
  auto tSrK = thr_mma.partition_fragment_B(sK);
  auto tOrVt = thr_mma.partition_fragment_B(sVt);
  auto acc_o = cute::partition_fragment_C(
      tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<V>>{});
  cute::clear(acc_o);
  FlashSCARegSoftmax<2 * decltype(cute::size<1>(acc_o))::value> softmax;
  const float scale_log2 = scale * float(M_LOG2E);

  bool is_first = true;
  for (int64_t k_start = window_start; k_start < window_end; k_start += BN) {
    const int64_t k_tile_end =
        k_start + BN < window_end ? k_start + BN : window_end;
    if (!FlashSCASegmentTilesMayOverlap(
            has_segment, q_segment_idx, k_segment_idx, b, L, C,
            k_segment_len, q_start, q_tile_end, k_start, k_tile_end,
            has_prev)) {
      continue;
    }
    FlashSCARegLoadKFull<T, Element, BN, D>(
        L, Hkv, C, k_start, b, kv_head, DActual, has_prev, k, prev_k, k_sm);
    FlashSCARegLoadVRowsFull<T, Element, BN, V>(
        L, Hkv, C, k_start, b, kv_head, VActual, has_prev, v, prev_v, v_sm);
    __syncthreads();

    auto acc_s = cute::partition_fragment_C(
        tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
    cute::clear(acc_s);
    FlashSCARegGemmRS<Element>(acc_s, rQ, tSrK, sK, tiled_mma);
    FlashSCARegApplyCausalChunkMask<BM, BN>(
        acc_s, L, C, chunk_end, q_start, k_start, b, has_prev, has_segment,
        q_segment_idx, k_segment_idx, k_segment_len);
    if (is_first) {
      softmax.template SoftmaxRescaleO<true>(acc_s, acc_o, scale_log2);
      is_first = false;
    } else {
      softmax.template SoftmaxRescaleO<false>(acc_s, acc_o, scale_log2);
    }
    auto rP = FlashSCARegConvertType<Element>(acc_s);
    auto tOrP =
        cute::make_tensor(rP.data(),
                          FlashSCARegConvertLayoutAccAregs<decltype(tiled_mma)>(
                              rP.layout()));
    FlashSCARegGemmRS<Element, true>(acc_o, tOrP, tOrVt, sVt, tiled_mma);
    __syncthreads();
  }

  auto lse_frag = softmax.Normalize(acc_o, scale);
  auto rO = FlashSCARegConvertType<Element>(acc_o);
  auto smem_tiled_copy_O =
      cute::make_tiled_copy_C(cute::Copy_Atom<cute::DefaultCopy, Element>{},
                              tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
  auto taccOrO = smem_thr_copy_O.retile_S(rO);
  auto taccOsO = smem_thr_copy_O.partition_D(sO);
  __syncthreads();
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  auto cO = cute::make_identity_tensor(
      cute::Shape<cute::Int<BM>, cute::Int<V>>{});
  auto taccOcO = thr_mma.partition_C(cO);
  auto taccOcO_row =
      cute::logical_divide(taccOcO, cute::Shape<cute::_2>{})(
          cute::make_coord(cute::Int<0>{}, cute::_), cute::_, 0);
  if (cute::get<1>(taccOcO_row(0)) == 0) {
    #pragma unroll
    for (int mi = 0; mi < cute::size(lse_frag); ++mi) {
      const int row = cute::get<0>(taccOcO_row(mi));
      const int64_t q_pos = q_start + row;
      if (q_pos < L && q_pos < chunk_end) {
        lse[(b * Hq + q_head) * L + q_pos] = lse_frag(mi);
      }
    }
  }
  __syncthreads();

  Element* y_e = reinterpret_cast<Element*>(y);
  for (int64_t idx = threadIdx.x; idx < int64_t(BM) * VActual;
       idx += blockDim.x) {
    const int64_t row = idx / VActual;
    const int64_t col = idx - row * VActual;
    const int64_t q_pos = q_start + row;
    if (q_pos < L && q_pos < chunk_end) {
      y_e[((b * L + q_pos) * Hq + q_head) * VActual + col] =
          o_sm[row * V + col];
    }
  }
}

template <typename T, typename Element, int BM, int BN, int D, int V,
          int CStatic>
__global__ void FlashSCARegDQKernel(
    int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t DActual,
    int64_t VActual,
    int64_t CArg, int64_t k_segment_len, bool has_prev, bool has_segment,
    float scale,
    const T* __restrict__ dy, const T* __restrict__ q,
    const T* __restrict__ k, const T* __restrict__ v,
    const T* __restrict__ prev_k, const T* __restrict__ prev_v,
    const float* __restrict__ lse, const float* __restrict__ delta,
    const int64_t* __restrict__ q_segment_idx,
    const int64_t* __restrict__ k_segment_idx, T* __restrict__ dq) {
  extern __shared__ char raw_smem[];
  char* smem = raw_smem;
  Element* q_sm = FlashSCAAllocSmem<Element>(smem, BM * D);
  Element* do_sm = FlashSCAAllocSmem<Element>(smem, BM * V);
  Element* k_sm = FlashSCAAllocSmem<Element>(smem, BN * D);
  Element* v_sm = FlashSCAAllocSmem<Element>(smem, BN * V);

  const int64_t C = CStatic > 0 ? int64_t(CStatic) : CArg;
  const int64_t q_block = blockIdx.x;
  const int64_t q_head = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t kv_head = q_head / (Hq / Hkv);
  const int64_t q_tiles_per_chunk = (C + BM - 1) / BM;
  const int64_t chunk_id = q_block / q_tiles_per_chunk;
  const int64_t q_tile_in_chunk = q_block - chunk_id * q_tiles_per_chunk;
  const int64_t chunk_start = chunk_id * C;
  const int64_t chunk_end = chunk_start + C;
  const int64_t q_start = chunk_start + q_tile_in_chunk * BM;
  const int64_t q_tile_end =
      (q_start + BM < chunk_end ? q_start + BM : chunk_end) < L
          ? (q_start + BM < chunk_end ? q_start + BM : chunk_end)
          : L;
  const int64_t window_start =
      (chunk_id == 0) ? (has_prev ? -C : 0) : (chunk_start - C);
  const int64_t window_end = q_tile_end;

  FlashSCARegLoadQFull<T, Element, BM, D>(
      L, Hq, q_start, b, q_head, DActual, q, q_sm);
  FlashSCARegLoadDOFull<T, Element, BM, V>(
      L, Hq, q_start, b, q_head, VActual, dy, do_sm);
  __syncthreads();

  auto tiled_mma = FlashSCATiledMmaQK<Element, BM>();
  auto sQ = FlashSCARowMajorTensor<Element, BM, D>(q_sm);
  auto sDO = FlashSCARowMajorTensor<Element, BM, V>(do_sm);
  auto sK = FlashSCARowMajorTensor<Element, BN, D>(k_sm);
  auto sV = FlashSCARowMajorTensor<Element, BN, V>(v_sm);
  auto sKt = FlashSCARegRowMajorStridedTensor<Element, D, BN, 1, D>(k_sm);
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  auto rQ = thr_mma.partition_fragment_A(sQ);
  auto rDO = thr_mma.partition_fragment_A(sDO);
  FlashSCARegCopySmemToRegA<Element>(rQ, sQ, tiled_mma);
  FlashSCARegCopySmemToRegA<Element>(rDO, sDO, tiled_mma);
  auto tSrK = thr_mma.partition_fragment_B(sK);
  auto tSrV = thr_mma.partition_fragment_B(sV);
  auto tOrKt = thr_mma.partition_fragment_B(sKt);
  auto acc_dq = cute::partition_fragment_C(
      tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<D>>{});
  cute::clear(acc_dq);
  const float scale_log2 = scale * float(M_LOG2E);

  for (int64_t k_start = window_start; k_start < window_end; k_start += BN) {
    const int64_t k_tile_end =
        k_start + BN < window_end ? k_start + BN : window_end;
    if (!FlashSCASegmentTilesMayOverlap(
            has_segment, q_segment_idx, k_segment_idx, b, L, C,
            k_segment_len, q_start, q_tile_end, k_start, k_tile_end,
            has_prev)) {
      continue;
    }
    FlashSCARegLoadKFull<T, Element, BN, D>(
        L, Hkv, C, k_start, b, kv_head, DActual, has_prev, k, prev_k, k_sm);
    FlashSCARegLoadVRowsFull<T, Element, BN, V>(
        L, Hkv, C, k_start, b, kv_head, VActual, has_prev, v, prev_v, v_sm);
    __syncthreads();

    auto acc_s = cute::partition_fragment_C(
        tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
    cute::clear(acc_s);
    FlashSCARegGemmRS<Element>(acc_s, rQ, tSrK, sK, tiled_mma);
    FlashSCARegApplyCausalChunkMask<BM, BN>(
        acc_s, L, C, chunk_end, q_start, k_start, b, has_prev, has_segment,
        q_segment_idx, k_segment_idx, k_segment_len);
    FlashSCARegApplySoftmaxFromLSE(
        acc_s, L, Hq, q_start, b, q_head, scale_log2, lse);

    auto acc_dp = cute::partition_fragment_C(
        tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
    cute::clear(acc_dp);
    FlashSCARegGemmRS<Element>(acc_dp, rDO, tSrV, sV, tiled_mma);
    FlashSCARegMakeDS(
        acc_s, acc_dp, L, Hq, q_start, b, q_head, scale, delta);

    auto rDS = FlashSCARegConvertType<Element>(acc_dp);
    auto tOrDS =
        cute::make_tensor(rDS.data(),
                          FlashSCARegConvertLayoutAccAregs<decltype(tiled_mma)>(
                              rDS.layout()));
    FlashSCARegGemmRS<Element, true>(acc_dq, tOrDS, tOrKt, sKt, tiled_mma);
    __syncthreads();
  }

  auto rDQ = FlashSCARegConvertType<Element>(acc_dq);
  Element* dq_e = reinterpret_cast<Element*>(dq);
  if (q_start + BM <= chunk_end && q_start + BM <= L) {
    auto gDQ = cute::make_tensor(
        cute::make_gmem_ptr(
            dq_e + ((b * L + q_start) * Hq + q_head) * DActual),
        cute::make_layout(cute::make_shape(cute::Int<BM>{}, cute::Int<D>{}),
                          cute::make_stride(Hq * DActual, cute::Int<1>{})));
    auto gmem_tiled_copy_DQ =
        cute::make_tiled_copy_C(cute::Copy_Atom<cute::DefaultCopy, Element>{},
                                tiled_mma);
    auto gmem_thr_copy_DQ = gmem_tiled_copy_DQ.get_thread_slice(threadIdx.x);
    auto taccDrD = gmem_thr_copy_DQ.retile_S(rDQ);
    auto taccDgD = gmem_thr_copy_DQ.partition_D(gDQ);
    cute::copy(gmem_tiled_copy_DQ, taccDrD, taccDgD);
  } else {
    FlashSCARegStoreDQ<BM, D>(
        rDQ, tiled_mma, L, Hq, DActual, q_start, chunk_end, b, q_head, dq_e);
  }
}

template <typename T, typename Element, int BM, int BN, int D, int V,
          bool IsPrev, bool MergedDQ, int CStatic>
__global__ void FlashSCARegDKVKernel(
    int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t DActual,
    int64_t VActual,
    int64_t CArg, int64_t k_segment_len, bool has_segment, float scale,
    const T* __restrict__ dy, const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, const T* __restrict__ prev_k,
    const T* __restrict__ prev_v, const float* __restrict__ lse,
    const float* __restrict__ delta,
    const int64_t* __restrict__ q_segment_idx,
    const int64_t* __restrict__ k_segment_idx, T* __restrict__ dk,
    T* __restrict__ dv, float* __restrict__ dq_accum) {
  extern __shared__ char raw_smem[];
  char* smem = raw_smem;
  Element* q_sm = FlashSCAAllocSmem<Element>(smem, BM * D);
  Element* do_sm = FlashSCAAllocSmem<Element>(smem, BM * V);
  Element* k_sm = FlashSCAAllocSmem<Element>(smem, BN * D);
  Element* v_sm = FlashSCAAllocSmem<Element>(smem, BN * V);
  Element* p_t_sm = FlashSCAAllocSmem<Element>(smem, BN * BM);
  Element* ds_t_sm = FlashSCAAllocSmem<Element>(smem, BN * BM);

  const int64_t C = CStatic > 0 ? int64_t(CStatic) : CArg;
  const int64_t k_tile = blockIdx.x;
  const int64_t kv_head = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t k_tiles_per_chunk = (C + BN - 1) / BN;
  const int64_t key_chunk = IsPrev ? int64_t(0) : (k_tile / k_tiles_per_chunk);
  const int64_t k_tile_in_chunk =
      IsPrev ? k_tile : (k_tile - key_chunk * k_tiles_per_chunk);
  const int64_t key_chunk_start = key_chunk * C;
  const int64_t key_chunk_end = key_chunk_start + C;
  const int64_t k_start =
      IsPrev ? (k_tile * BN) : (key_chunk_start + k_tile_in_chunk * BN);
  const int64_t k_virtual_start = IsPrev ? (k_start - C) : k_start;

  FlashSCARegLoadKFull<T, Element, BN, D>(
      L, Hkv, C, k_virtual_start, b, kv_head, DActual, IsPrev, k, prev_k,
      k_sm);
  FlashSCARegLoadVRowsFull<T, Element, BN, V>(
      L, Hkv, C, k_virtual_start, b, kv_head, VActual, IsPrev, v, prev_v,
      v_sm);
  __syncthreads();

  auto tiled_mma_qk = FlashSCATiledMmaQK<Element, BM>();
  auto tiled_mma_dkv = FlashSCATiledMmaDKV<Element, BM, BN>();
  auto sQ = FlashSCARowMajorTensor<Element, BM, D>(q_sm);
  auto sDO = FlashSCARowMajorTensor<Element, BM, V>(do_sm);
  auto sK = FlashSCARowMajorTensor<Element, BN, D>(k_sm);
  auto sV = FlashSCARowMajorTensor<Element, BN, V>(v_sm);
  auto sPt = FlashSCARowMajorTensor<Element, BN, BM>(p_t_sm);
  auto sDSt = FlashSCARowMajorTensor<Element, BN, BM>(ds_t_sm);
  auto sDOt = FlashSCARegRowMajorStridedTensor<Element, V, BM, 1, V>(do_sm);
  auto sQt = FlashSCARegRowMajorStridedTensor<Element, D, BM, 1, D>(q_sm);
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x);
  auto tSrK = thr_mma_qk.partition_fragment_B(sK);
  auto tSrV = thr_mma_qk.partition_fragment_B(sV);
  auto acc_dk = cute::partition_fragment_C(
      tiled_mma_dkv, cute::Shape<cute::Int<BN>, cute::Int<D>>{});
  auto acc_dv = cute::partition_fragment_C(
      tiled_mma_dkv, cute::Shape<cute::Int<BN>, cute::Int<V>>{});
  cute::clear(acc_dk);
  cute::clear(acc_dv);
  const float scale_log2 = scale * float(M_LOG2E);

  const int64_t num_chunks = (L + C - 1) / C;
  const int64_t q_chunk_begin = IsPrev ? int64_t(0) : key_chunk;
  const int64_t next_chunk_end =
      (key_chunk + int64_t(2) < num_chunks) ? key_chunk + int64_t(2)
                                            : num_chunks;
  const int64_t q_chunk_end = IsPrev ? int64_t(1) : next_chunk_end;
  const int64_t q_heads_per_kv = Hq / Hkv;
  const int64_t q_head_begin = kv_head * q_heads_per_kv;
  const int64_t q_head_end = q_head_begin + q_heads_per_kv;
  for (int64_t q_head = q_head_begin; q_head < q_head_end; ++q_head) {
    for (int64_t q_chunk = q_chunk_begin; q_chunk < q_chunk_end; ++q_chunk) {
      const int64_t q_chunk_start = q_chunk * C;
      const int64_t q_chunk_end_pos = q_chunk_start + C;
      int64_t q_loop_start = q_chunk_start;
      if (!IsPrev && q_chunk == key_chunk) {
        const int64_t q_causal_start =
            q_chunk_start + ((k_start - q_chunk_start) / BM) * BM;
        q_loop_start =
            q_loop_start < q_causal_start ? q_causal_start : q_loop_start;
      }
      for (int64_t q_start = q_loop_start; q_start < q_chunk_end_pos;
           q_start += BM) {
        const int64_t q_tile_end =
            (q_start + BM < q_chunk_end_pos ? q_start + BM
                                             : q_chunk_end_pos) < L
                ? (q_start + BM < q_chunk_end_pos ? q_start + BM
                                                   : q_chunk_end_pos)
                : L;
        const int64_t k_tile_end =
            IsPrev ? (k_virtual_start + BN)
                   : (k_virtual_start + BN < key_chunk_end
                          ? k_virtual_start + BN
                          : key_chunk_end);
        if (!FlashSCASegmentTilesMayOverlap(
                has_segment, q_segment_idx, k_segment_idx, b, L, C,
                k_segment_len, q_start, q_tile_end, k_virtual_start,
                k_tile_end, IsPrev)) {
          continue;
        }
        FlashSCARegLoadQFull<T, Element, BM, D>(
            L, Hq, q_start, b, q_head, DActual, q, q_sm);
        FlashSCARegLoadDOFull<T, Element, BM, V>(
            L, Hq, q_start, b, q_head, VActual, dy, do_sm);
        __syncthreads();
        auto rQ = thr_mma_qk.partition_fragment_A(sQ);
        auto rDO = thr_mma_qk.partition_fragment_A(sDO);
        FlashSCARegCopySmemToRegA<Element>(rQ, sQ, tiled_mma_qk);
        FlashSCARegCopySmemToRegA<Element>(rDO, sDO, tiled_mma_qk);

        auto acc_s = cute::partition_fragment_C(
            tiled_mma_qk, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
        cute::clear(acc_s);
        FlashSCARegGemmRS<Element>(acc_s, rQ, tSrK, sK, tiled_mma_qk);
        FlashSCARegApplyCausalChunkMask<BM, BN>(
            acc_s, L, C, q_chunk_end_pos, q_start, k_virtual_start, b,
            IsPrev, has_segment, q_segment_idx, k_segment_idx,
            k_segment_len);
        FlashSCARegApplySoftmaxFromLSE(
            acc_s, L, Hq, q_start, b, q_head, scale_log2, lse);
        FlashSCARegStoreTransposedScoreTile<Element, BM, BN>(
            acc_s, q_start, k_virtual_start, p_t_sm);
        __syncthreads();

        auto acc_dp = cute::partition_fragment_C(
            tiled_mma_qk, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
        cute::clear(acc_dp);
        FlashSCARegGemmRS<Element>(acc_dp, rDO, tSrV, sV, tiled_mma_qk);
        FlashSCARegMakeDS(
            acc_s, acc_dp, L, Hq, q_start, b, q_head, scale, delta);
        if constexpr (MergedDQ) {
          auto sKt =
              FlashSCARegRowMajorStridedTensor<Element, D, BN, 1, D>(k_sm);
          auto tOrKt = thr_mma_qk.partition_fragment_B(sKt);
          auto acc_dq = cute::partition_fragment_C(
              tiled_mma_qk, cute::Shape<cute::Int<BM>, cute::Int<D>>{});
          cute::clear(acc_dq);
          auto rDS = FlashSCARegConvertType<Element>(acc_dp);
          auto tOrDS = cute::make_tensor(
              rDS.data(),
              FlashSCARegConvertLayoutAccAregs<decltype(tiled_mma_qk)>(
                  rDS.layout()));
          FlashSCARegGemmRS<Element, true>(
              acc_dq, tOrDS, tOrKt, sKt, tiled_mma_qk);
          FlashSCARegAtomicAddDQ<BM, D>(
              acc_dq, tiled_mma_qk, L, Hq, DActual, q_start,
              q_chunk_end_pos, b, q_head, dq_accum);
        }
        FlashSCARegStoreTransposedScoreTile<Element, BM, BN>(
            acc_dp, q_start, k_virtual_start, ds_t_sm);
        __syncthreads();

        FlashSCARegGemmSS<Element, false, true>(
            acc_dv, sPt, sDOt, tiled_mma_dkv);
        FlashSCARegGemmSS<Element, false, true>(
            acc_dk, sDSt, sQt, tiled_mma_dkv);
        __syncthreads();
      }
    }
  }

  auto rDK = FlashSCARegConvertType<Element>(acc_dk);
  auto rDV = FlashSCARegConvertType<Element>(acc_dv);
  Element* dk_e = reinterpret_cast<Element*>(dk);
  Element* dv_e = reinterpret_cast<Element*>(dv);
  const int64_t out_L = IsPrev ? C : L;
  if (k_start + BN <= out_L) {
    auto gDK = cute::make_tensor(
        cute::make_gmem_ptr(
            dk_e + ((b * out_L + k_start) * Hkv + kv_head) * DActual),
        cute::make_layout(cute::make_shape(cute::Int<BN>{}, cute::Int<D>{}),
                          cute::make_stride(Hkv * DActual, cute::Int<1>{})));
    auto gDV = cute::make_tensor(
        cute::make_gmem_ptr(
            dv_e + ((b * out_L + k_start) * Hkv + kv_head) * VActual),
        cute::make_layout(cute::make_shape(cute::Int<BN>{}, cute::Int<V>{}),
                          cute::make_stride(Hkv * VActual, cute::Int<1>{})));
    auto gmem_tiled_copy_DK =
        cute::make_tiled_copy_C(cute::Copy_Atom<cute::DefaultCopy, Element>{},
                                tiled_mma_dkv);
    auto gmem_thr_copy_DK = gmem_tiled_copy_DK.get_thread_slice(threadIdx.x);
    auto taccDKr = gmem_thr_copy_DK.retile_S(rDK);
    auto taccDKg = gmem_thr_copy_DK.partition_D(gDK);
    auto taccDVr = gmem_thr_copy_DK.retile_S(rDV);
    auto taccDVg = gmem_thr_copy_DK.partition_D(gDV);
    cute::copy(gmem_tiled_copy_DK, taccDKr, taccDKg);
    cute::copy(gmem_tiled_copy_DK, taccDVr, taccDVg);
  } else {
    FlashSCARegStoreKVGrad<BN, D>(
        rDK, tiled_mma_dkv, out_L, Hkv, DActual, k_start, b, kv_head, dk_e);
    FlashSCARegStoreKVGrad<BN, V>(
        rDV, tiled_mma_dkv, out_L, Hkv, VActual, k_start, b, kv_head, dv_e);
  }
}

template <int D, int V, int BM, int BN>
constexpr int64_t FlashSCARegFwdSmemBytes() {
  return (BM * D + BN * D + V * BN + BM * V) *
             int64_t(sizeof(cutlass::half_t)) +
         512;
}

template <int D, int V, int BM, int BN>
constexpr int64_t FlashSCARegDQSmemBytes() {
  return (BM * D + BM * V + BN * D + BN * V) *
             int64_t(sizeof(cutlass::half_t)) +
         512;
}

template <int D, int V, int BM, int BN>
constexpr int64_t FlashSCARegDKVSmemBytes() {
  return (BM * D + BM * V + BN * D + BN * V + 2 * BN * BM) *
             int64_t(sizeof(cutlass::half_t)) +
         512;
}

#define LAUNCH_FLASH_SCA_FWD_KERNEL(T, D_VALUE, V_VALUE, BM, BN, C_STATIC) \
  do {                                                                     \
    static_assert((BM) == 32 || ((BM) >= 64 && (BM) % 64 == 0),            \
                  "FlashSCA FWD BM must be 32 or a multiple of 64");       \
    static_assert((BN) >= 32 && (BN) % 16 == 0,                            \
                  "FlashSCA FWD BN must be a multiple of 16");             \
    const int64_t c = (C_STATIC) > 0 ? int64_t(C_STATIC) : chunk_size;      \
    TORCH_CHECK(c == chunk_size, "FlashSCA static chunk dispatch expected ", \
                c, ", got ", chunk_size);                                  \
    const int64_t q_tiles_per_chunk = (c + (BM) - 1) / (BM);               \
    const int64_t num_chunks = (L + c - 1) / c;                            \
    const dim3 grid(num_chunks * q_tiles_per_chunk, Hq, B);                \
    cuda_utils::LaunchKernel(                                              \
        FlashSCARegFwdKernel<T, typename FlashSCACutlassElement<T>::Type,   \
                             BM, BN, D_VALUE, V_VALUE, C_STATIC>,          \
        grid, dim3(FlashSCAThreadsForBM<BM>()),                            \
        FlashSCARegFwdSmemBytes<D_VALUE, V_VALUE, BM, BN>(), cuda_stream,  \
        B, L, Hq, Hkv, DActual, VActual, c, k_segment_len, has_prev,       \
        has_segment, scale, q_data, k_data, v_data, prev_k_data,           \
        prev_v_data, q_segment_idx_data, k_segment_idx_data, y_data,       \
        lse_data);                                                         \
  } while (false)

#define LAUNCH_FLASH_SCA_DQ_KERNEL(T, D_VALUE, V_VALUE, BM, BN, C_STATIC)  \
  do {                                                                     \
    static_assert((BM) == 32 || ((BM) >= 64 && (BM) % 64 == 0),            \
                  "FlashSCA DQ BM must be 32 or a multiple of 64");        \
    static_assert((BN) >= 32 && (BN) % 16 == 0,                            \
                  "FlashSCA DQ BN must be a multiple of 16");              \
    const int64_t c = (C_STATIC) > 0 ? int64_t(C_STATIC) : chunk_size;      \
    TORCH_CHECK(c == chunk_size, "FlashSCA static chunk dispatch expected ", \
                c, ", got ", chunk_size);                                  \
    const int64_t q_tiles_per_chunk = (c + (BM) - 1) / (BM);               \
    const int64_t num_chunks = (L + c - 1) / c;                            \
    const dim3 grid(num_chunks * q_tiles_per_chunk, Hq, B);                \
    cuda_utils::LaunchKernel(                                              \
        FlashSCARegDQKernel<T, typename FlashSCACutlassElement<T>::Type,    \
                            BM, BN, D_VALUE, V_VALUE, C_STATIC>,           \
        grid, dim3(FlashSCAThreadsForBM<BM>()),                            \
        FlashSCARegDQSmemBytes<D_VALUE, V_VALUE, BM, BN>(), cuda_stream,   \
        B, L, Hq, Hkv, DActual, VActual, c, k_segment_len, has_prev,       \
        has_segment, scale, dy_data, q_data, k_data, v_data, prev_k_data,  \
        prev_v_data, lse_data, delta_data, q_segment_idx_data,             \
        k_segment_idx_data, dq_data);                                      \
  } while (false)

#define LAUNCH_FLASH_SCA_DKV_KERNEL(                                       \
    T, D_VALUE, V_VALUE, BM, BN, IS_PREV, MERGED_DQ, C_STATIC, DK_DATA,    \
    DV_DATA)                                                               \
  do {                                                                     \
    static_assert((BM) == 32 || ((BM) >= 64 && (BM) % 64 == 0),            \
                  "FlashSCA DKV BM must be 32 or a multiple of 64");       \
    static_assert((BN) >= 32 && (BN) % 16 == 0,                            \
                  "FlashSCA DKV BN must be a multiple of 16");             \
    const int64_t c = (C_STATIC) > 0 ? int64_t(C_STATIC) : chunk_size;      \
    TORCH_CHECK(c == chunk_size, "FlashSCA static chunk dispatch expected ", \
                c, ", got ", chunk_size);                                  \
    const int64_t k_tiles_per_chunk = (c + (BN) - 1) / (BN);               \
    const int64_t num_chunks = (L + c - 1) / c;                            \
    const dim3 grid((IS_PREV) ? k_tiles_per_chunk                          \
                             : num_chunks * k_tiles_per_chunk,             \
                    Hkv, B);                                               \
    cuda_utils::LaunchKernel(                                              \
        FlashSCARegDKVKernel<T, typename FlashSCACutlassElement<T>::Type,   \
                             BM, BN, D_VALUE, V_VALUE, IS_PREV, MERGED_DQ, \
                             C_STATIC>,                                    \
        grid, dim3(FlashSCAThreadsForBM<BM>()),                            \
        FlashSCARegDKVSmemBytes<D_VALUE, V_VALUE, BM, BN>(), cuda_stream,  \
        B, L, Hq, Hkv, DActual, VActual, c, k_segment_len, has_segment,    \
        scale,                                                             \
        dy_data, q_data, k_data, v_data, prev_k_data, prev_v_data,         \
        lse_data, delta_data, q_segment_idx_data, k_segment_idx_data,      \
        DK_DATA, DV_DATA, (MERGED_DQ) ? dq_accum_data : nullptr);          \
  } while (false)

#define LAUNCH_FLASH_SCA_DET_BWD_KERNEL(                                   \
    T, D_VALUE, V_VALUE, C_STATIC, DQ_BM, DQ_BN, DKV_BM, DKV_BN)           \
  do {                                                                     \
    TORCH_CHECK(chunk_size >= (DKV_BN) && chunk_size % (DKV_BN) == 0,      \
                "FlashSCA bwd unsupported chunk_size=", chunk_size,        \
                " for DKV_BN=", (DKV_BN));                                 \
    LAUNCH_FLASH_SCA_DQ_KERNEL(T, D_VALUE, V_VALUE, DQ_BM, DQ_BN,          \
                               C_STATIC);                                  \
    LAUNCH_FLASH_SCA_DKV_KERNEL(T, D_VALUE, V_VALUE, DKV_BM, DKV_BN,       \
                                false, false, C_STATIC, dk_data, dv_data); \
    if (has_prev) {                                                        \
      LAUNCH_FLASH_SCA_DKV_KERNEL(T, D_VALUE, V_VALUE, DKV_BM, DKV_BN,     \
                                  true, false, C_STATIC, prev_dk_data,     \
                                  prev_dv_data);                           \
    }                                                                      \
  } while (false)

#define LAUNCH_FLASH_SCA_NONDET_BWD_KERNEL(                                \
    T, D_VALUE, V_VALUE, C_STATIC, DKV_BM, DKV_BN)                         \
  do {                                                                     \
    TORCH_CHECK(chunk_size >= (DKV_BN) && chunk_size % (DKV_BN) == 0,      \
                "FlashSCA bwd unsupported chunk_size=", chunk_size,        \
                " for DKV_BN=", (DKV_BN));                                 \
    LAUNCH_FLASH_SCA_DKV_KERNEL(T, D_VALUE, V_VALUE, DKV_BM, DKV_BN,       \
                                false, true, C_STATIC, dk_data, dv_data);  \
    if (has_prev) {                                                        \
      LAUNCH_FLASH_SCA_DKV_KERNEL(T, D_VALUE, V_VALUE, DKV_BM, DKV_BN,     \
                                  true, true, C_STATIC, prev_dk_data,      \
                                  prev_dv_data);                           \
    }                                                                      \
  } while (false)

#define DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(                                \
    T, D_VALUE, V_VALUE, C_STATIC, BM, BN)                                 \
  do {                                                                     \
    LAUNCH_FLASH_SCA_FWD_KERNEL(T, D_VALUE, V_VALUE, BM, BN, C_STATIC);    \
  } while (false)

#define DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(                                \
    T, D_VALUE, V_VALUE, C_STATIC, DQ_BM, DQ_BN, DET_DKV_BM, DET_DKV_BN,   \
    NONDET_DKV_BM, NONDET_DKV_BN)                                          \
  do {                                                                     \
    if (deterministic) {                                                   \
      LAUNCH_FLASH_SCA_DET_BWD_KERNEL(T, D_VALUE, V_VALUE, C_STATIC,       \
                                      DQ_BM, DQ_BN, DET_DKV_BM,            \
                                      DET_DKV_BN);                         \
    } else {                                                               \
      LAUNCH_FLASH_SCA_NONDET_BWD_KERNEL(T, D_VALUE, V_VALUE, C_STATIC,    \
                                         NONDET_DKV_BM, NONDET_DKV_BN);    \
    }                                                                      \
  } while (false)

#define DISPATCH_FLASH_SCA_FWD_KERNEL(T, C_STATIC)                         \
  do {                                                                     \
    if (D == 32 && V == 32) {                                              \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 32, 32, C_STATIC, 32, 32);     \
    } else if (D == 32 && V == 64) {                                       \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 32, 64, C_STATIC, 32, 32);     \
    } else if (D == 32 && V == 128) {                                      \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 32, 128, C_STATIC, 32, 32);    \
    } else if (D == 32 && V == 256) {                                      \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 32, 256, C_STATIC, 32, 32);    \
    } else if (D == 64 && V == 32) {                                       \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 64, 32, C_STATIC, 32, 32);     \
    } else if (D == 64 && V == 64) {                                       \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 64, 64, C_STATIC, 32, 32);     \
    } else if (D == 64 && V == 128) {                                      \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 64, 128, C_STATIC, 32, 32);    \
    } else if (D == 64 && V == 256) {                                      \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 64, 256, C_STATIC, 32, 32);    \
    } else if (D == 128 && V == 32) {                                      \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 128, 32, C_STATIC, 32, 32);    \
    } else if (D == 128 && V == 64) {                                      \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 128, 64, C_STATIC, 32, 32);    \
    } else if (D == 128 && V == 128) {                                     \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 128, 128, C_STATIC, 32, 32);   \
    } else if (D == 128 && V == 256) {                                     \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 128, 256, C_STATIC, 32, 32);   \
    } else if (D == 256 && V == 32) {                                      \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 32, C_STATIC, 32, 32);    \
    } else if (D == 256 && V == 64) {                                      \
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 64, C_STATIC, 32, 32);    \
    } else if (D == 256 && V == 128) {                                     \
      if (chunk_size == 4096) {                                            \
        DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 128, C_STATIC, 64, 64); \
      } else {                                                             \
        DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 128, C_STATIC, 32, 32); \
      }                                                                    \
    } else if (D == 256 && V == 256) {                                     \
      if (chunk_size == 4096) {                                            \
        DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 256, C_STATIC, 64, 48); \
      } else {                                                             \
        DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 256, C_STATIC, 64, 32); \
      }                                                                    \
    } else {                                                               \
      TORCH_CHECK(false, "FlashSCA fwd unsupported D=", D,                 \
                  ", V=", V, ", chunk_size=", chunk_size);                 \
    }                                                                      \
  } while (false)

#define DISPATCH_FLASH_SCA_BWD_KERNEL(T, C_STATIC)                         \
  do {                                                                     \
    if (D == 32 && V == 32) {                                              \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 32, 32, C_STATIC, 32, 32,      \
                                         32, 32, 32, 64);                  \
    } else if (D == 32 && V == 64) {                                       \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 32, 64, C_STATIC, 32, 32,      \
                                         32, 64, 32, 64);                  \
    } else if (D == 32 && V == 128) {                                      \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 32, 128, C_STATIC, 64, 32,     \
                                         32, 64, 32, 64);                  \
    } else if (D == 32 && V == 256) {                                      \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 32, 256, C_STATIC, 32, 32,     \
                                         32, 32, 32, 32);                  \
    } else if (D == 64 && V == 32) {                                       \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 64, 32, C_STATIC, 32, 32,      \
                                         32, 64, 32, 64);                  \
    } else if (D == 64 && V == 64) {                                       \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 64, 64, C_STATIC, 32, 32,      \
                                         32, 64, 32, 64);                  \
    } else if (D == 64 && V == 128) {                                      \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 64, 128, C_STATIC, 32, 32,     \
                                         32, 32, 32, 32);                  \
    } else if (D == 64 && V == 256) {                                      \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 64, 256, C_STATIC, 32, 32,     \
                                         32, 32, 32, 32);                  \
    } else if (D == 128 && V == 32) {                                      \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 128, 32, C_STATIC, 32, 32,     \
                                         32, 64, 32, 32);                  \
    } else if (D == 128 && V == 64) {                                      \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 128, 64, C_STATIC, 32, 32,     \
                                         32, 32, 32, 32);                  \
    } else if (D == 128 && V == 128) {                                     \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 128, 128, C_STATIC, 32, 32,    \
                                         32, 32, 32, 32);                  \
    } else if (D == 128 && V == 256) {                                     \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 128, 256, C_STATIC, 32, 32,    \
                                         32, 32, 32, 32);                  \
    } else if (D == 256 && V == 32) {                                      \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 32, C_STATIC, 32, 32,     \
                                         32, 32, 32, 32);                  \
    } else if (D == 256 && V == 64) {                                      \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 64, C_STATIC, 32, 32,     \
                                         32, 32, 32, 32);                  \
    } else if (D == 256 && V == 128) {                                     \
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 128, C_STATIC, 32, 32,    \
                                         32, 32, 32, 32);                  \
    } else if (D == 256 && V == 256) {                                     \
      if (chunk_size == 4096) {                                            \
        DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 256, C_STATIC, 64, 48,  \
                                           32, 32, 32, 32);                \
      } else {                                                             \
        DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 256, C_STATIC, 64, 32,  \
                                           32, 32, 32, 32);                \
      }                                                                    \
    } else {                                                               \
      TORCH_CHECK(false, "FlashSCA bwd unsupported D=", D,                 \
                  ", V=", V, ", chunk_size=", chunk_size);                 \
    }                                                                      \
  } while (false)

inline bool FlashSCARegSupportsHeadDim(int64_t dim) {
  return dim == 32 || dim == 64 || dim == 128 || dim == 256;
}

template <typename T, int DValue, int VValue>
void FlashSCAFwdCase(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& y, torch::Tensor& lse) {
  static_assert(DValue == 32 || DValue == 64 || DValue == 128 ||
                DValue == 256);
  static_assert(VValue == 32 || VValue == 64 || VValue == 128 ||
                VValue == 256);
  const int64_t B = q.size(0);
  const int64_t L = q.size(1);
  const int64_t Hq = q.size(2);
  const int64_t Hkv = k.size(2);
  const int64_t DActual = q.size(3);
  const int64_t VActual = v.size(3);
  const int64_t k_segment_len = has_segment ? k_segment_idx.size(1) : 0;
  const T* q_data = q.data_ptr<T>();
  const T* k_data = k.data_ptr<T>();
  const T* v_data = v.data_ptr<T>();
  const T* prev_k_data = has_prev ? prev_k.data_ptr<T>() : nullptr;
  const T* prev_v_data = has_prev ? prev_v.data_ptr<T>() : nullptr;
  const int64_t* q_segment_idx_data =
      has_segment ? q_segment_idx.data_ptr<int64_t>() : nullptr;
  const int64_t* k_segment_idx_data =
      has_segment ? k_segment_idx.data_ptr<int64_t>() : nullptr;
  T* y_data = y.data_ptr<T>();
  float* lse_data = lse.data_ptr<float>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(DActual == DValue && VActual == VValue,
              "FlashSCA fwd case mismatch: expected D=", DValue,
              ", V=", VValue, "; got D=", DActual, ", V=", VActual);

  if constexpr (DValue == 32 && VValue == 32) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 32, 32, 0, 32, 32);
  } else if constexpr (DValue == 32 && VValue == 64) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 32, 64, 0, 32, 32);
  } else if constexpr (DValue == 32 && VValue == 128) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 32, 128, 0, 32, 32);
  } else if constexpr (DValue == 32 && VValue == 256) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 32, 256, 0, 32, 32);
  } else if constexpr (DValue == 64 && VValue == 32) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 64, 32, 0, 32, 32);
  } else if constexpr (DValue == 64 && VValue == 64) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 64, 64, 0, 32, 32);
  } else if constexpr (DValue == 64 && VValue == 128) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 64, 128, 0, 32, 32);
  } else if constexpr (DValue == 64 && VValue == 256) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 64, 256, 0, 32, 32);
  } else if constexpr (DValue == 128 && VValue == 32) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 128, 32, 0, 32, 32);
  } else if constexpr (DValue == 128 && VValue == 64) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 128, 64, 0, 32, 32);
  } else if constexpr (DValue == 128 && VValue == 128) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 128, 128, 0, 32, 32);
  } else if constexpr (DValue == 128 && VValue == 256) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 128, 256, 0, 32, 32);
  } else if constexpr (DValue == 256 && VValue == 32) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 32, 0, 32, 32);
  } else if constexpr (DValue == 256 && VValue == 64) {
    DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 64, 0, 32, 32);
  } else if constexpr (DValue == 256 && VValue == 128) {
    if (chunk_size == 4096) {
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 128, 0, 64, 64);
    } else {
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 128, 0, 32, 32);
    }
  } else if constexpr (DValue == 256 && VValue == 256) {
    if (chunk_size == 4096) {
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 256, 0, 64, 48);
    } else {
      DISPATCH_FLASH_SCA_FWD_TILE_KERNEL(T, 256, 256, 0, 64, 32);
    }
  }
}

template <typename T, int DValue, int VValue>
void FlashSCABwdCase(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& dq, torch::Tensor& dk, torch::Tensor& dv,
    c10::optional<torch::Tensor>& prev_dk,
    c10::optional<torch::Tensor>& prev_dv, bool deterministic) {
  static_assert(DValue == 32 || DValue == 64 || DValue == 128 ||
                DValue == 256);
  static_assert(VValue == 32 || VValue == 64 || VValue == 128 ||
                VValue == 256);
  const int64_t B = q.size(0);
  const int64_t L = q.size(1);
  const int64_t Hq = q.size(2);
  const int64_t Hkv = k.size(2);
  const int64_t DActual = q.size(3);
  const int64_t VActual = v.size(3);
  const int64_t k_segment_len = has_segment ? k_segment_idx.size(1) : 0;
  TORCH_CHECK(DActual == DValue && VActual == VValue,
              "FlashSCA bwd case mismatch: expected D=", DValue,
              ", V=", VValue, "; got D=", DActual, ", V=", VActual);

  torch::Tensor dq_accum;
  if (!deterministic) {
    dq_accum =
        torch::zeros({B, L, Hq, DActual},
                     q.options().dtype(at::kFloat).memory_format(
                         at::MemoryFormat::Contiguous));
  }

  torch::Tensor delta =
      torch::empty({B, Hq, L},
                   lse.options().memory_format(at::MemoryFormat::Contiguous));

  const T* dy_data = dy.data_ptr<T>();
  const T* q_data = q.data_ptr<T>();
  const T* k_data = k.data_ptr<T>();
  const T* v_data = v.data_ptr<T>();
  const T* y_data = y.data_ptr<T>();
  const T* prev_k_data = has_prev ? prev_k.data_ptr<T>() : nullptr;
  const T* prev_v_data = has_prev ? prev_v.data_ptr<T>() : nullptr;
  const float* lse_data = lse.data_ptr<float>();
  float* delta_data = delta.data_ptr<float>();
  const int64_t* q_segment_idx_data =
      has_segment ? q_segment_idx.data_ptr<int64_t>() : nullptr;
  const int64_t* k_segment_idx_data =
      has_segment ? k_segment_idx.data_ptr<int64_t>() : nullptr;
  T* dq_data = deterministic ? dq.data_ptr<T>() : nullptr;
  T* dk_data = dk.data_ptr<T>();
  T* dv_data = dv.data_ptr<T>();
  T* prev_dk_data = has_prev ? prev_dk.value().data_ptr<T>() : nullptr;
  T* prev_dv_data = has_prev ? prev_dv.value().data_ptr<T>() : nullptr;
  float* dq_accum_data =
      deterministic ? nullptr : dq_accum.data_ptr<float>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  FlashSCADeltaKernel<T>
      <<<dim3((L + kFlashSCADeltaRowsPerBlock - 1) /
                  kFlashSCADeltaRowsPerBlock,
              Hq, B),
         kFlashSCADeltaThreads, 0, cuda_stream>>>(
          L, Hq, VActual, dy_data, y_data, delta_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if constexpr (DValue == 32 && VValue == 32) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 32, 32, 0, 32, 32, 32, 32,
                                       32, 64);
  } else if constexpr (DValue == 32 && VValue == 64) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 32, 64, 0, 32, 32, 32, 64,
                                       32, 64);
  } else if constexpr (DValue == 32 && VValue == 128) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 32, 128, 0, 64, 32, 32, 64,
                                       32, 64);
  } else if constexpr (DValue == 32 && VValue == 256) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 32, 256, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 64 && VValue == 32) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 64, 32, 0, 32, 32, 32, 64,
                                       32, 64);
  } else if constexpr (DValue == 64 && VValue == 64) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 64, 64, 0, 32, 32, 32, 64,
                                       32, 64);
  } else if constexpr (DValue == 64 && VValue == 128) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 64, 128, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 64 && VValue == 256) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 64, 256, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 128 && VValue == 32) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 128, 32, 0, 32, 32, 32, 64,
                                       32, 32);
  } else if constexpr (DValue == 128 && VValue == 64) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 128, 64, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 128 && VValue == 128) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 128, 128, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 128 && VValue == 256) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 128, 256, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 256 && VValue == 32) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 32, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 256 && VValue == 64) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 64, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 256 && VValue == 128) {
    DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 128, 0, 32, 32, 32, 32,
                                       32, 32);
  } else if constexpr (DValue == 256 && VValue == 256) {
    if (chunk_size == 4096) {
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 256, 0, 64, 48, 32, 32,
                                         32, 32);
    } else {
      DISPATCH_FLASH_SCA_BWD_TILE_KERNEL(T, 256, 256, 0, 64, 32, 32, 32,
                                         32, 32);
    }
  }

  if (!deterministic) {
    dq = dq_accum.to(q.scalar_type());
  }
}

#undef DISPATCH_FLASH_SCA_BWD_KERNEL
#undef DISPATCH_FLASH_SCA_FWD_KERNEL
#undef DISPATCH_FLASH_SCA_BWD_TILE_KERNEL
#undef DISPATCH_FLASH_SCA_FWD_TILE_KERNEL
#undef LAUNCH_FLASH_SCA_NONDET_BWD_KERNEL
#undef LAUNCH_FLASH_SCA_DET_BWD_KERNEL
#undef LAUNCH_FLASH_SCA_DKV_KERNEL
#undef LAUNCH_FLASH_SCA_DQ_KERNEL
#undef LAUNCH_FLASH_SCA_FWD_KERNEL

template <typename T>
__global__ void FlashSCADeltaKernel(
    int64_t L, int64_t H, int64_t V, const T* __restrict__ dy,
    const T* __restrict__ y, float* __restrict__ delta) {
  const int lane = threadIdx.x & (cuda_utils::kWarpSize - 1);
  const int64_t row_in_block = threadIdx.x / cuda_utils::kWarpSize;
  const int64_t q_pos =
      blockIdx.x * kFlashSCADeltaRowsPerBlock + row_in_block;
  if (q_pos >= L) {
    return;
  }
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  const int64_t base = ((batch * L + q_pos) * H + head) * V;
  float partial = 0.0f;
  for (int64_t d = lane; d < V; d += cuda_utils::kWarpSize) {
    partial += FlashSCAToFloat(dy[base + d]) * FlashSCAToFloat(y[base + d]);
  }
  const float sum = reduce::WarpReduce(partial, reduce::SumOp<float>());
  if (lane == 0) {
    delta[(batch * H + head) * L + q_pos] = sum;
  }
}

}  // namespace

}  // namespace ops
}  // namespace xattn

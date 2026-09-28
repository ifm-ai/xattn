#include "flash_sca/sliding_chunk_attention.h"

#include <ATen/cuda/CUDAContext.h>
#include "attention/hopper/input_layout.h"
#include <ATen/cuda/Exceptions.h>
#include <ATen/ops/cumsum.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cub/device/device_select.cuh>
#include "attention/partition/cub_iterators.h"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "attention/partition/segment_metadata.h"
#include "attention/sequence/varlen_metadata.h"
#include "flash_sca/hopper/bwd.h"
#include "flash_sca/hopper/fwd.h"
#include "flash_sca/hopper/plans/sca_compat/registry.h"

namespace xattn {
namespace ops {

namespace {

struct FlashSCABosPlanCacheEntry {
  torch::Tensor bos;
  torch::Tensor cu_seqlens;
  torch::Tensor segment_idx;
  int64_t version = -1;
  int64_t max_seqlen = 0;
  int64_t num_sequences = 0;
  int device = -1;
  cudaStream_t stream = nullptr;
};

thread_local std::array<FlashSCABosPlanCacheEntry, 2>
    g_flash_sca_bos_plan_cache;
thread_local int g_flash_sca_next_bos_plan_cache_entry = 0;

struct FlashSCASegmentDensePlanCacheEntry {
  torch::Tensor q_segment_idx;
  torch::Tensor k_segment_idx;
  int64_t q_version = -1;
  int64_t k_version = -1;
  int64_t q_length = -1;
  int64_t k_length = -1;
  int device = -1;
  cudaStream_t stream = nullptr;
  bool is_dense = false;
  bool initialized = false;
};

thread_local std::array<FlashSCASegmentDensePlanCacheEntry, 4>
    g_flash_sca_segment_dense_plan_cache;
thread_local int g_flash_sca_next_segment_dense_plan_cache_entry = 0;

struct FlashSCAVarlenDensePlanCacheEntry {
  torch::Tensor cu_seqlens_q;
  torch::Tensor cu_seqlens_k;
  torch::Tensor position_offsets;
  int64_t q_version = -1;
  int64_t k_version = -1;
  int64_t position_version = -1;
  int64_t batch = -1;
  int64_t q_row_length = -1;
  int64_t k_row_length = -1;
  int device = -1;
  cudaStream_t stream = nullptr;
  bool is_dense = false;
  bool initialized = false;
};

thread_local std::array<FlashSCAVarlenDensePlanCacheEntry, 4>
    g_flash_sca_varlen_dense_plan_cache;
thread_local int g_flash_sca_next_varlen_dense_plan_cache_entry = 0;

bool FlashSCATensorCacheable(const torch::Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->version_counter().enabled();
}

bool FlashSCABosPlanCacheable(const torch::Tensor& bos) {
  return FlashSCATensorCacheable(bos);
}

bool FlashSCABosPlanCacheMatches(
    const FlashSCABosPlanCacheEntry& entry, const torch::Tensor& bos,
    cudaStream_t stream) {
  return FlashSCABosPlanCacheable(bos) && entry.bos.defined() &&
      entry.bos.data_ptr() == bos.data_ptr() &&
      entry.bos.sizes().equals(bos.sizes()) &&
      entry.bos.strides().equals(bos.strides()) &&
      entry.version == bos._version() && entry.device == bos.get_device() &&
      entry.stream == stream;
}

FlashSCABosPlanCacheEntry* FlashSCAFindBosPlanCacheEntry(
    const torch::Tensor& bos, cudaStream_t stream) {
  for (FlashSCABosPlanCacheEntry& entry : g_flash_sca_bos_plan_cache) {
    if (FlashSCABosPlanCacheMatches(entry, bos, stream)) {
      return &entry;
    }
  }
  return nullptr;
}

FlashSCABosPlanCacheEntry& FlashSCAGetOrCreateBosPlanCacheEntry(
    const torch::Tensor& bos, cudaStream_t stream) {
  TORCH_INTERNAL_ASSERT(FlashSCABosPlanCacheable(bos));
  if (FlashSCABosPlanCacheEntry* entry =
          FlashSCAFindBosPlanCacheEntry(bos, stream)) {
    return *entry;
  }
  FlashSCABosPlanCacheEntry& entry =
      g_flash_sca_bos_plan_cache[g_flash_sca_next_bos_plan_cache_entry];
  g_flash_sca_next_bos_plan_cache_entry =
      (g_flash_sca_next_bos_plan_cache_entry + 1) %
      static_cast<int>(g_flash_sca_bos_plan_cache.size());
  entry = FlashSCABosPlanCacheEntry{};
  entry.bos = bos;
  entry.version = bos._version();
  entry.device = bos.get_device();
  entry.stream = stream;
  return entry;
}

bool FlashSCASegmentDensePlanCacheable(
    const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx) {
  return FlashSCATensorCacheable(q_segment_idx) &&
      FlashSCATensorCacheable(k_segment_idx);
}

bool FlashSCASegmentDensePlanCacheMatches(
    const FlashSCASegmentDensePlanCacheEntry& entry,
    const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx,
    cudaStream_t stream) {
  return FlashSCASegmentDensePlanCacheable(q_segment_idx, k_segment_idx) &&
      entry.initialized &&
      entry.q_segment_idx.unsafeGetTensorImpl() ==
          q_segment_idx.unsafeGetTensorImpl() &&
      entry.k_segment_idx.unsafeGetTensorImpl() ==
          k_segment_idx.unsafeGetTensorImpl() &&
      entry.q_version == q_segment_idx._version() &&
      entry.k_version == k_segment_idx._version() &&
      entry.q_length == q_segment_idx.size(1) &&
      entry.k_length == k_segment_idx.size(1) &&
      entry.device == q_segment_idx.get_device() && entry.stream == stream;
}

FlashSCASegmentDensePlanCacheEntry*
FlashSCAFindSegmentDensePlanCacheEntry(
    const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx,
    cudaStream_t stream) {
  for (FlashSCASegmentDensePlanCacheEntry& entry :
       g_flash_sca_segment_dense_plan_cache) {
    if (FlashSCASegmentDensePlanCacheMatches(
            entry, q_segment_idx, k_segment_idx, stream)) {
      return &entry;
    }
  }
  return nullptr;
}

void FlashSCAStoreSegmentDensePlan(
    const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx,
    cudaStream_t stream,
    bool is_dense) {
  if (!FlashSCASegmentDensePlanCacheable(q_segment_idx, k_segment_idx)) {
    return;
  }
  FlashSCASegmentDensePlanCacheEntry& entry =
      g_flash_sca_segment_dense_plan_cache
          [g_flash_sca_next_segment_dense_plan_cache_entry];
  g_flash_sca_next_segment_dense_plan_cache_entry =
      (g_flash_sca_next_segment_dense_plan_cache_entry + 1) %
      static_cast<int>(g_flash_sca_segment_dense_plan_cache.size());
  entry = FlashSCASegmentDensePlanCacheEntry{};
  entry.q_segment_idx = q_segment_idx;
  entry.k_segment_idx = k_segment_idx;
  entry.q_version = q_segment_idx._version();
  entry.k_version = k_segment_idx._version();
  entry.q_length = q_segment_idx.size(1);
  entry.k_length = k_segment_idx.size(1);
  entry.device = q_segment_idx.get_device();
  entry.stream = stream;
  entry.is_dense = is_dense;
  entry.initialized = true;
}

bool FlashSCAVarlenDensePlanCacheable(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const c10::optional<torch::Tensor>& position_offsets) {
  return FlashSCATensorCacheable(cu_seqlens_q) &&
      FlashSCATensorCacheable(cu_seqlens_k) &&
      (!position_offsets.has_value() ||
       FlashSCATensorCacheable(position_offsets.value()));
}

bool FlashSCAVarlenDensePlanCacheMatches(
    const FlashSCAVarlenDensePlanCacheEntry& entry,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t batch,
    int64_t q_row_length,
    int64_t k_row_length,
    const c10::optional<torch::Tensor>& position_offsets,
    cudaStream_t stream) {
  if (!entry.initialized ||
      entry.cu_seqlens_q.unsafeGetTensorImpl() !=
          cu_seqlens_q.unsafeGetTensorImpl() ||
      entry.cu_seqlens_k.unsafeGetTensorImpl() !=
          cu_seqlens_k.unsafeGetTensorImpl() ||
      (position_offsets.has_value()
           ? !entry.position_offsets.defined() ||
               entry.position_offsets.unsafeGetTensorImpl() !=
                   position_offsets.value().unsafeGetTensorImpl()
           : entry.position_offsets.defined())) {
    return false;
  }
  const bool positions_match =
      position_offsets.has_value()
      ? entry.position_version == position_offsets.value()._version()
      : true;
  return entry.q_version == cu_seqlens_q._version() &&
      entry.k_version == cu_seqlens_k._version() &&
      positions_match && entry.batch == batch &&
      entry.q_row_length == q_row_length &&
      entry.k_row_length == k_row_length &&
      entry.device == cu_seqlens_q.get_device() && entry.stream == stream;
}

FlashSCAVarlenDensePlanCacheEntry*
FlashSCAFindVarlenDensePlanCacheEntry(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t batch,
    int64_t q_row_length,
    int64_t k_row_length,
    const c10::optional<torch::Tensor>& position_offsets,
    cudaStream_t stream) {
  for (FlashSCAVarlenDensePlanCacheEntry& entry :
       g_flash_sca_varlen_dense_plan_cache) {
    if (FlashSCAVarlenDensePlanCacheMatches(
            entry, cu_seqlens_q, cu_seqlens_k, batch, q_row_length,
            k_row_length, position_offsets, stream)) {
      return &entry;
    }
  }
  return nullptr;
}

void FlashSCAStoreVarlenDensePlan(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t batch,
    int64_t q_row_length,
    int64_t k_row_length,
    const c10::optional<torch::Tensor>& position_offsets,
    cudaStream_t stream,
    bool is_dense) {
  if (!FlashSCAVarlenDensePlanCacheable(
          cu_seqlens_q, cu_seqlens_k, position_offsets)) {
    return;
  }
  FlashSCAVarlenDensePlanCacheEntry& entry =
      g_flash_sca_varlen_dense_plan_cache
          [g_flash_sca_next_varlen_dense_plan_cache_entry];
  g_flash_sca_next_varlen_dense_plan_cache_entry =
      (g_flash_sca_next_varlen_dense_plan_cache_entry + 1) %
      static_cast<int>(g_flash_sca_varlen_dense_plan_cache.size());
  entry = FlashSCAVarlenDensePlanCacheEntry{};
  entry.cu_seqlens_q = cu_seqlens_q;
  entry.cu_seqlens_k = cu_seqlens_k;
  if (position_offsets.has_value()) {
    entry.position_offsets = position_offsets.value();
    entry.position_version = position_offsets.value()._version();
  }
  entry.q_version = cu_seqlens_q._version();
  entry.k_version = cu_seqlens_k._version();
  entry.batch = batch;
  entry.q_row_length = q_row_length;
  entry.k_row_length = k_row_length;
  entry.device = cu_seqlens_q.get_device();
  entry.stream = stream;
  entry.is_dense = is_dense;
  entry.initialized = true;
}

__global__ void FlashSCACheckSegmentMetadataDenseKernel(
    const int64_t* q_segment_idx,
    const int64_t* k_segment_idx,
    int batch,
    int q_length,
    int k_length,
    int64_t total_values,
    int* mismatch) {
  for (int64_t linear =
           int64_t(blockIdx.x) * blockDim.x + int64_t(threadIdx.x);
       linear < total_values;
       linear += int64_t(gridDim.x) * blockDim.x) {
    if (linear < int64_t(batch) * q_length) {
      const int row = static_cast<int>(linear / q_length);
      if (q_segment_idx[linear] != q_segment_idx[int64_t(row) * q_length]) {
        atomicExch(mismatch, 1);
      }
      continue;
    }
    const int64_t k_linear = linear - int64_t(batch) * q_length;
    if (k_linear < int64_t(batch) * k_length) {
      const int row = static_cast<int>(k_linear / k_length);
      if (k_segment_idx[k_linear] !=
          k_segment_idx[int64_t(row) * k_length]) {
        atomicExch(mismatch, 1);
      }
      continue;
    }
    const int row = static_cast<int>(
        k_linear - int64_t(batch) * k_length);
    if (q_segment_idx[int64_t(row) * q_length] !=
        k_segment_idx[int64_t(row) * k_length]) {
      atomicExch(mismatch, 1);
    }
  }
}

struct FlashSCABosStartPredicate {
  const bool* bos;
  int row_length;

  __host__ __device__ __forceinline__ bool operator()(const int& token) const {
    return bos[token] || token % row_length == 0;
  }
};

struct FlashSCABosSuffixStartPredicate {
  const bool* bos;
  int q_row_length;
  int k_row_length;
  int q_offset;

  __host__ __device__ __forceinline__ bool operator()(const int& token) const {
    const int row = token / q_row_length;
    const int column = token % q_row_length;
    return column == 0 ||
        bos[int64_t(row) * k_row_length + q_offset + column];
  }
};

__global__ void FlashSCAFinalizeBosToCuSeqlensKernel(
    int total_tokens, int* cu_seqlens, int* plan_stats) {
  const int num_sequences = plan_stats[0];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cu_seqlens[num_sequences] = total_tokens;
  }
  for (int64_t sequence =
           int64_t(blockIdx.x) * blockDim.x + int64_t(threadIdx.x);
       sequence < num_sequences;
       sequence += int64_t(gridDim.x) * blockDim.x) {
    const int start = cu_seqlens[sequence];
    const int end = sequence + 1 < num_sequences
        ? cu_seqlens[sequence + 1]
        : total_tokens;
    atomicMax(plan_stats + 1, end - start);
  }
}

constexpr int FlashSCABwdHeadDimBucketSm90(int headdim) {
  return headdim <= 32
      ? 32
      : (headdim <= 64
             ? 64
             : (headdim <= 96
                    ? 96
                    : (headdim <= 128
                           ? 128
                           : (headdim <= 160
                                  ? 160
                                  : (headdim <= 192 ? 192 : 256)))));
}

constexpr int kD64PersistentChunk = 512;
constexpr int kDenseSmallChunk = 64;

template <
    typename Element, typename ElementOut, bool kVarlen, bool kHasSegment,
    int kHeadDim,
    int kHeadDimV>
void DispatchFlashSCAFwdSm90VD(
    AttentionFwdParams& params, cudaStream_t stream) {
  RunFlashSCAFwdSm90VD<Element, ElementOut, kVarlen, kHasSegment,
                       kHeadDim, kHeadDimV>(
      params, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void DispatchFlashSCABwdSm90Variant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  RunFlashSCABwdSm90Variant<Element, kHeadDim, kHeadDimV, kBlockN, kBlockM>(
      params, deterministic, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM, int kStages, int kStagesDO, int kStagesDS,
          bool kPersistentScheduler>
void DispatchFlashSCABwdSm90DenseDet(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSCABwdSm90DenseDet<
      Element, kHeadDim, kHeadDimV, kBlockN, kBlockM, kStages,
      kStagesDO, kStagesDS, kPersistentScheduler>(params, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM, int kStages, int kStagesDO, int kStagesDS,
          bool kDeterministic, bool kDirectChunkRange>
void DispatchFlashSCABwdSm90DenseGQA(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSCABwdSm90DenseGQA<
      Element, kHeadDim, kHeadDimV, kBlockN, kBlockM, kStages,
      kStagesDO, kStagesDS, kDeterministic, kDirectChunkRange>(
          params, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void DispatchFlashSCABwdSm90NonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSCABwdSm90NonDetVariant<
      Element, kHeadDim, kHeadDimV, kBlockN, kBlockM>(
      params, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void DispatchFlashSCABwdSm90VarlenVariant(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream) {
  RunFlashSCABwdSm90VarlenVariant<
      Element, kHeadDim, kHeadDimV, kBlockN, kBlockM>(
      params, deterministic, stream);
}

template <typename Element, int kHeadDim, int kHeadDimV, int kBlockN,
          int kBlockM = 0>
void DispatchFlashSCABwdSm90VarlenNonDetVariant(
    AttentionBwdParams& params, cudaStream_t stream) {
  RunFlashSCABwdSm90VarlenNonDetVariant<
      Element, kHeadDim, kHeadDimV, kBlockN, kBlockM>(
      params, stream);
}

torch::Tensor FlashSCASM90PadLastDim(
    const torch::Tensor& tensor, int64_t target_dim) {
  TORCH_CHECK(tensor.dim() >= 2,
              "FlashSCA SM90 expected a tensor with at least 2 dims");
  const int64_t last_dim = tensor.dim() - 1;
  if (tensor.size(last_dim) == target_dim) {
    return tensor.contiguous();
  }
  TORCH_CHECK(tensor.size(last_dim) < target_dim,
              "FlashSCA SM90 pad target must be >= tensor last dim");
  std::vector<int64_t> padded_sizes(tensor.sizes().begin(),
                                    tensor.sizes().end());
  padded_sizes[last_dim] = target_dim;
  torch::Tensor padded = torch::zeros(
      padded_sizes,
      tensor.options().memory_format(at::MemoryFormat::Contiguous));
  padded.narrow(last_dim, 0, tensor.size(last_dim)).copy_(tensor);
  return padded;
}

torch::Tensor FlashSCASM90EmptyLastDimLike(
    const torch::Tensor& tensor, int64_t target_dim) {
  const int64_t last_dim = tensor.dim() - 1;
  std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
  sizes[last_dim] = target_dim;
  return torch::empty(
      sizes,
      tensor.options().memory_format(at::MemoryFormat::Contiguous));
}

template <typename Element, typename ElementOut, bool kVarlen,
          bool kHasSegment, int kHeadDim>
void RunFlashSCAFwdSm90V(
    AttentionFwdParams& params, cudaStream_t stream) {
  if (params.dv <= 32) {
    DispatchFlashSCAFwdSm90VD<Element, ElementOut, kVarlen, kHasSegment,
                         kHeadDim, 32>(
        params, stream);
    return;
  }
  if (params.dv <= 64) {
    DispatchFlashSCAFwdSm90VD<Element, ElementOut, kVarlen, kHasSegment,
                         kHeadDim, 64>(
        params, stream);
  } else if (params.dv <= 96) {
    DispatchFlashSCAFwdSm90VD<Element, ElementOut, kVarlen, kHasSegment,
                         kHeadDim, 96>(
        params, stream);
  } else if (params.dv <= 128) {
    DispatchFlashSCAFwdSm90VD<Element, ElementOut, kVarlen, kHasSegment,
                         kHeadDim, 128>(
        params, stream);
  } else if (params.dv <= 160) {
    DispatchFlashSCAFwdSm90VD<Element, ElementOut, kVarlen, kHasSegment,
                         kHeadDim, 160>(
        params, stream);
  } else if (params.dv <= 192) {
    DispatchFlashSCAFwdSm90VD<Element, ElementOut, kVarlen, kHasSegment,
                         kHeadDim, 192>(
        params, stream);
  } else if (params.dv <= 256) {
    DispatchFlashSCAFwdSm90VD<Element, ElementOut, kVarlen, kHasSegment,
                         kHeadDim, 256>(
        params, stream);
  } else {
    TORCH_CHECK(false, "FlashSCA SM90 FWD supports V <= 256; got V=",
                params.dv);
  }
}

template <typename Element, typename ElementOut>
void RunFlashSCAFwdSm90D(AttentionFwdParams& params, cudaStream_t stream,
                         bool varlen, bool has_segment) {
  if (params.d <= 32) {
    if (varlen) {
      RunFlashSCAFwdSm90V<Element, ElementOut, true, false, 32>(
          params, stream);
    } else if (has_segment) {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, true, 32>(
          params, stream);
    } else {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, false, 32>(
          params, stream);
    }
  } else if (params.d <= 64) {
    if (varlen) {
      RunFlashSCAFwdSm90V<Element, ElementOut, true, false, 64>(
          params, stream);
    } else if (has_segment) {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, true, 64>(
          params, stream);
    } else {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, false, 64>(
          params, stream);
    }
  } else if (params.d <= 96) {
    if (varlen) {
      RunFlashSCAFwdSm90V<Element, ElementOut, true, false, 96>(
          params, stream);
    } else if (has_segment) {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, true, 96>(
          params, stream);
    } else {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, false, 96>(
          params, stream);
    }
  } else if (params.d <= 128) {
    if (varlen) {
      RunFlashSCAFwdSm90V<Element, ElementOut, true, false, 128>(
          params, stream);
    } else if (has_segment) {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, true, 128>(
          params, stream);
    } else {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, false, 128>(
          params, stream);
    }
  } else if (params.d <= 160) {
    if (varlen) {
      RunFlashSCAFwdSm90V<Element, ElementOut, true, false, 160>(
          params, stream);
    } else if (has_segment) {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, true, 160>(
          params, stream);
    } else {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, false, 160>(
          params, stream);
    }
  } else if (params.d <= 192) {
    if (varlen) {
      RunFlashSCAFwdSm90V<Element, ElementOut, true, false, 192>(
          params, stream);
    } else if (has_segment) {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, true, 192>(
          params, stream);
    } else {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, false, 192>(
          params, stream);
    }
  } else if (params.d <= 256) {
    if (varlen) {
      RunFlashSCAFwdSm90V<Element, ElementOut, true, false, 256>(
          params, stream);
    } else if (has_segment) {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, true, 256>(
          params, stream);
    } else {
      RunFlashSCAFwdSm90V<Element, ElementOut, false, false, 256>(
          params, stream);
    }
  } else {
    TORCH_CHECK(false, "FlashSCA SM90 FWD supports D <= 256; got D=",
                params.d);
  }
}

template <typename Element, typename ElementOut>
void RunFlashSCAFwdSm90(AttentionFwdParams& params, cudaStream_t stream,
                        bool varlen) {
  RunFlashSCAFwdSm90D<Element, ElementOut>(
      params, stream, varlen, params.q_segment_idx != nullptr);
}

// Native rectangular tiles.
inline bool UseNativeSmallChunkDetBwd(int d, int v, int chunk) {
  return chunk == 256 &&
      ((d == 160 && (v == 128 || v == 160 || v == 192)) ||
       (d == 192 && (v == 128 || v == 160)));
}

template <typename Element>
void RunFlashSCABwdSm90(
    AttentionBwdParams& params, bool deterministic, cudaStream_t stream,
    bool native_small_chunk_det = false) {
  const bool varlen = params.cu_seqlens_q != nullptr;
  TORCH_CHECK(params.dv <= 256,
              "FlashSCA SM90 BWD supports padded V <= 256; got ",
              params.dv);
  const bool dense =
      !varlen && params.q_segment_idx == nullptr &&
      params.seqlen_q == params.seqlen_k;
  if (params.odd_head_window_right_delta == 0) {
  if (native_small_chunk_det && dense && deterministic &&
      UseNativeSmallChunkDetBwd(params.d, params.dv, params.attention_chunk)) {
    if (params.d == 160 && params.dv == 128) {
      DispatchFlashSCABwdSm90Variant<Element, 160, 128, 128>(
          params, deterministic, stream);
      return;
    }
    if (params.d == 160 && params.dv == 160) {
      if (params.h == params.h_k) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 160, 160, 128, 64, 0, 0, 0, false>(params, stream);
      } else {
        DispatchFlashSCABwdSm90DenseGQA<
            Element, 160, 160, 128, 64, 0, 0, 0, true, true>(params, stream);
      }
      return;
    }
    if (params.d == 160 && params.dv == 192) {
      DispatchFlashSCABwdSm90Variant<Element, 160, 192, 128>(
          params, deterministic, stream);
      return;
    }
    if (params.d == 192 && params.dv == 128) {
      DispatchFlashSCABwdSm90Variant<Element, 192, 128, 128>(
          params, deterministic, stream);
      return;
    }
    if (params.d == 192 && params.dv == 160) {
      DispatchFlashSCABwdSm90Variant<Element, 192, 160, 128>(
          params, deterministic, stream);
      return;
    }
  }
  if (dense && params.d == 64 && params.dv == 96) {
    if (!deterministic) {
      DispatchFlashSCABwdSm90Variant<Element, 96, 96, 128>(
          params, deterministic, stream);
    } else if (params.h == params.h_k) {
      if (params.attention_chunk >= kD64PersistentChunk) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 64, 128, 128, 64, 4, 1, 1, true>(params, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 64, 96, 128>(
            params, deterministic, stream);
      }
    } else {
      DispatchFlashSCABwdSm90Variant<Element, 64, 96, 128>(
          params, deterministic, stream);
    }
    return;
  }
  if (dense && deterministic && params.h != params.h_k) {
    if (params.d == 128 && params.dv == 64) {
      DispatchFlashSCABwdSm90DenseGQA<
          Element, 128, 64, 128, 64, 2, 2, 2, true, false>(
              params, stream);
      return;
    }
    if (params.d == 96 && params.dv == 96 &&
        params.h == 4 * params.h_k) {
      DispatchFlashSCABwdSm90DenseGQA<
          Element, 96, 96, 128, 64, 2, 2, 2, true, false>(
              params, stream);
      return;
    }
    if (params.d == 96 && params.dv == 128 &&
        params.h == 4 * params.h_k) {
      DispatchFlashSCABwdSm90DenseGQA<
          Element, 96, 128, 128, 64, 2, 2, 2, true, false>(
              params, stream);
      return;
    }
  }
  if (dense && !deterministic && params.d == 192 && params.dv == 128 &&
      params.h == 2 * params.h_k &&
      params.attention_chunk <= kDenseSmallChunk) {
    DispatchFlashSCABwdSm90DenseGQA<
        Element, 192, 128, 128, 64, 1, 1, 1, false, false>(
            params, stream);
    return;
  }
  if (dense && deterministic && params.h != params.h_k &&
      params.d == 160 && params.dv == 192) {
    DispatchFlashSCABwdSm90Variant<Element, 192, 192, 96>(
        params, deterministic, stream);
    return;
  }
  if (dense &&
      ((params.d == 192 && params.dv == 128) ||
       (params.d == 160 &&
        (params.dv == 128 || params.dv == 160)))) {
    DispatchFlashSCABwdSm90Variant<Element, 192, 192, 96>(
        params, deterministic, stream);
    return;
  }
  if (deterministic && !varlen &&
      params.q_segment_idx == nullptr && params.h == params.h_k &&
      params.seqlen_q == params.seqlen_k) {
    if (params.d == 192 && params.dv == 32 &&
        params.attention_chunk <= kDenseSmallChunk) {
      DispatchFlashSCABwdSm90DenseDet<
          Element, 192, 32, 128, 64, 1, 1, 1, false>(params, stream);
      return;
    }
    if (params.d == 192 && params.dv == 64 &&
        params.attention_chunk <= kDenseSmallChunk) {
      DispatchFlashSCABwdSm90DenseDet<
          Element, 192, 64, 128, 64, 1, 1, 1, false>(params, stream);
      return;
    }
    if (params.d == 192 && params.dv == 192 &&
        params.attention_chunk <= kDenseSmallChunk) {
      DispatchFlashSCABwdSm90DenseDet<
          Element, 192, 192, 96, 64, 1, 1, 1, false>(params, stream);
      return;
    }
    if (params.d == 64 && params.dv == 128) {
      if (params.attention_chunk < kD64PersistentChunk) {
        DispatchFlashSCABwdSm90Variant<Element, 64, 128, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 64, 128, 128, 64, 4, 1, 1, true>(params, stream);
      }
      return;
    }
    if (params.d == 96) {
      if (params.dv == 32) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 96, 32, 128, 128, 2, 2, 2, true>(params, stream);
        return;
      } else if (params.dv == 128) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 96, 128, 128, 64, 4, 1, 1, true>(params, stream);
        return;
      } else if (params.dv == 160) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 96, 160, 128, 64, 4, 1, 1, false>(params, stream);
        return;
      } else if (params.dv == 192) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 96, 192, 128, 64, 4, 1, 1, true>(params, stream);
        return;
      } else if (params.dv == 256 &&
                 params.attention_chunk <= kDenseSmallChunk) {
        DispatchFlashSCABwdSm90Variant<Element, 96, 256, 128>(
            params, deterministic, stream);
        return;
      }
    }
    if (params.d == 128) {
      if (params.dv == 32) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 128, 32, 128, 64, 4, 1, 1, false>(params, stream);
        return;
      } else if (params.dv == 64) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 128, 64, 128, 64, 2, 2, 2, true>(params, stream);
        return;
      } else if (params.dv == 96) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 128, 96, 128, 64, 4, 1, 1, false>(params, stream);
        return;
      } else if (params.dv == 128) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 128, 128, 128, 64, 4, 1, 1, false>(params, stream);
        return;
      } else if (params.dv == 160) {
        DispatchFlashSCABwdSm90DenseDet<
            Element, 128, 160, 128, 64, 4, 1, 1, false>(params, stream);
        return;
      }
    }
  }
  if (deterministic && !varlen && params.q_segment_idx == nullptr &&
      params.h == params.h_k && params.d == 160 && params.dv > 96 &&
      params.dv <= 192) {
    DispatchFlashSCABwdSm90Variant<Element, 192, 192, 96>(
        params, deterministic, stream);
    return;
  }
  if (deterministic && !varlen && params.q_segment_idx == nullptr &&
      params.h == params.h_k && params.d == 192 && params.dv > 64 &&
      params.dv <= 192) {
    DispatchFlashSCABwdSm90Variant<Element, 192, 192, 96>(
        params, deterministic, stream);
    return;
  }
  if (deterministic && !varlen && params.q_segment_idx == nullptr &&
      params.h == params.h_k && params.seqlen_q == params.seqlen_k &&
      params.d == 128 && params.dv == 192) {
    DispatchFlashSCABwdSm90Variant<Element, 192, 192, 96>(
        params, deterministic, stream);
    return;
  }
  if (deterministic && !varlen && params.q_segment_idx == nullptr &&
      params.h == params.h_k && params.seqlen_q == params.seqlen_k &&
      (params.d == 96 || params.d == 128) &&
      params.dv == 256) {
    DispatchFlashSCABwdSm90Variant<Element, 192, 256, 64>(
        params, deterministic, stream);
    return;
  }
  }
  if (params.d <= 32) {
    if (params.dv <= 32) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 32, 32, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 32, 32, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 64) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 32, 64, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 32, 64, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 96) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 32, 96, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 32, 96, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 128) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 32, 128, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 32, 128, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 160) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 32, 160, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 32, 160, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 192) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 32, 192, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 32, 192, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 256) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 32, 256, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 32, 256, 128>(
            params, deterministic, stream);
      }
    }
  } else if (params.d <= 64) {
    if (params.dv <= 32) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 64, 32, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 64, 32, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 64) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 64, 64, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 64, 64, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 96) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 64, 96, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 64, 96, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 128) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 64, 128, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 64, 128, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 160) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 64, 160, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 64, 160, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 192) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 64, 192, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 64, 192, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 256) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 64, 256, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 64, 256, 128>(
            params, deterministic, stream);
      }
    }
  } else if (params.d <= 96) {
    if (params.dv <= 32) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 96, 32, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 96, 32, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 64) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 96, 64, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 96, 64, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 96) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 96, 96, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 96, 96, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 128) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 96, 128, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 96, 128, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 160) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 96, 160, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 96, 160, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 192) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 96, 192, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 96, 192, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 256) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 96, 256, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 96, 256, 128>(
            params, deterministic, stream);
      }
    }
  } else if (params.d <= 128) {
    if (params.dv <= 32) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 128, 32, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 128, 32, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 64) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 128, 64, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 128, 64, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 96) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 128, 96, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 128, 96, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 128) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 128, 128, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 128, 128, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 160) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 128, 160, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 128, 160, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 192) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 128, 192, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 128, 192, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 256) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 128, 256, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 128, 256, 128>(
            params, deterministic, stream);
      }
    }
  } else if (params.d <= 160) {
    if (params.dv <= 32) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 160, 32, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 160, 32, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 64) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 160, 64, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 160, 64, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 96) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 160, 96, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 160, 96, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 128) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 160, 128, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 160, 128, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 160) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 160, 160, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 160, 160, 64>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 192) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 160, 192, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 160, 192, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 256) {
      if (varlen) {
        TORCH_CHECK(false,
                    "FlashSCA SM90 BWD requires rounded D=160,V=256 to be "
                    "padded to rounded D=256 before dispatch");
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 160, 256, 64>(
            params, deterministic, stream);
      }
    }
  } else if (params.d <= 192) {
    if (params.dv <= 32) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 192, 32, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 192, 32, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 64) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 192, 64, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 192, 64, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 96) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 192, 96, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 192, 96, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 128) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 192, 128, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 192, 128, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 160) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 192, 160, 128>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 192, 160, 128>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 192) {
      if (varlen) {
        DispatchFlashSCABwdSm90VarlenVariant<Element, 192, 192, 96>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 192, 192, 96>(
            params, deterministic, stream);
      }
    } else if (params.dv <= 256) {
      if (varlen) {
        TORCH_CHECK(false,
                    "FlashSCA SM90 BWD requires rounded D=192,V=256 to be "
                    "padded to rounded D=256 before dispatch");
      } else {
        DispatchFlashSCABwdSm90Variant<Element, 192, 256, 64>(
            params, deterministic, stream);
      }
    }
  } else if (params.d <= 256) {
    if (!varlen && params.dv <= 32) {
      if (deterministic) {
        DispatchFlashSCABwdSm90Variant<Element, 256, 32, 64>(
            params, deterministic, stream);
      } else {
        DispatchFlashSCABwdSm90NonDetVariant<Element, 256, 32, 128>(
            params, stream);
      }
    } else if (params.dv <= 64) {
      if (deterministic) {
        if (varlen) {
          DispatchFlashSCABwdSm90VarlenVariant<Element, 256, 128, 64>(
              params, deterministic, stream);
        } else {
          DispatchFlashSCABwdSm90Variant<Element, 256, 64, 64>(
              params, deterministic, stream);
        }
      } else if (varlen) {
        DispatchFlashSCABwdSm90VarlenNonDetVariant<Element, 256, 64, 128>(
            params, stream);
      } else {
        DispatchFlashSCABwdSm90NonDetVariant<Element, 256, 64, 128>(
            params, stream);
      }
    } else if (params.dv <= 96) {
      if (deterministic) {
        if (varlen) {
          DispatchFlashSCABwdSm90VarlenVariant<Element, 256, 128, 64>(
              params, deterministic, stream);
        } else {
          DispatchFlashSCABwdSm90Variant<Element, 256, 96, 64>(
              params, deterministic, stream);
        }
      } else if (varlen) {
        DispatchFlashSCABwdSm90VarlenNonDetVariant<Element, 256, 96, 128>(
            params, stream);
      } else {
        DispatchFlashSCABwdSm90NonDetVariant<Element, 256, 96, 128>(
            params, stream);
      }
    } else if (params.dv <= 128) {
      if (deterministic) {
        if (varlen) {
          DispatchFlashSCABwdSm90VarlenVariant<Element, 256, 128, 64>(
              params, deterministic, stream);
        } else {
          DispatchFlashSCABwdSm90Variant<Element, 256, 128, 64>(
              params, deterministic, stream);
        }
      } else if (varlen) {
        DispatchFlashSCABwdSm90VarlenNonDetVariant<Element, 256, 128, 128>(
            params, stream);
      } else {
        DispatchFlashSCABwdSm90NonDetVariant<Element, 256, 128, 128>(
            params, stream);
      }
    } else if (params.dv <= 160) {
      if (deterministic) {
        if (varlen) {
          DispatchFlashSCABwdSm90VarlenVariant<Element, 256, 256, 64>(
              params, deterministic, stream);
        } else {
          DispatchFlashSCABwdSm90Variant<Element, 256, 160, 64>(
              params, deterministic, stream);
        }
      } else if (varlen) {
        DispatchFlashSCABwdSm90VarlenNonDetVariant<
            Element, 256, 160, 128>(params, stream);
      } else {
        DispatchFlashSCABwdSm90NonDetVariant<Element, 256, 160, 128>(
            params, stream);
      }
    } else if (params.dv <= 192) {
      if (deterministic) {
        if (varlen) {
          DispatchFlashSCABwdSm90VarlenVariant<Element, 256, 256, 64>(
              params, deterministic, stream);
        } else {
          DispatchFlashSCABwdSm90Variant<Element, 256, 192, 64>(
              params, deterministic, stream);
        }
      } else if (varlen) {
        DispatchFlashSCABwdSm90VarlenNonDetVariant<Element, 256, 192, 128>(
            params, stream);
      } else {
        DispatchFlashSCABwdSm90NonDetVariant<Element, 256, 192, 128>(
            params, stream);
      }
    } else if (params.dv <= 256) {
      if (deterministic) {
        if (varlen) {
          DispatchFlashSCABwdSm90VarlenVariant<Element, 256, 256, 64>(
              params, deterministic, stream);
        } else {
          DispatchFlashSCABwdSm90Variant<Element, 256, 256, 64>(
              params, deterministic, stream);
        }
      } else if (varlen) {
        DispatchFlashSCABwdSm90VarlenNonDetVariant<Element, 256, 256, 80>(
            params, stream);
      } else {
        DispatchFlashSCABwdSm90NonDetVariant<Element, 256, 256, 80>(
            params, stream);
      }
    }
  } else {
    TORCH_CHECK(false, "FlashSCA SM90 BWD supports padded D <= 256; got ",
                params.d);
  }
}

int FlashSCACurrentDeviceArch(const torch::Tensor& q) {
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  return prop->major * 10 + prop->minor;
}

torch::Tensor FlashSCASM90MaybeCastCompute(
    const torch::Tensor& tensor, c10::ScalarType dtype, bool do_cast) {
  return (do_cast ? tensor.to(dtype) : tensor).contiguous();
}

void FlashSCACheckDenseQKVSM90(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "FlashSCA SM90 q/k/v must be 4D tensors");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == v.scalar_type(),
              "FlashSCA SM90 q/k/v must have the same dtype");
  TORCH_CHECK(q.scalar_type() == at::kHalf ||
                  q.scalar_type() == at::kBFloat16,
              "FlashSCA SM90 supports only fp16 and bf16 q, k, v input");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "FlashSCA SM90 q/k/v batch sizes must match");
  TORCH_CHECK(q.size(1) > 0 && q.size(1) <= k.size(1) && k.size(1) == v.size(1),
              "FlashSCA SM90 requires 0 < Q length <= matching K/V lengths before "
              "prev-chunk concatenation");
  TORCH_CHECK(k.size(2) == v.size(2),
              "FlashSCA SM90 k/v head counts must match");
  TORCH_CHECK(q.size(2) > 0 && k.size(2) > 0 &&
                  q.size(2) % k.size(2) == 0,
              "FlashSCA SM90 q head count must be divisible by kv head count; "
              "got Hq=", q.size(2), ", Hkv=", k.size(2));
  TORCH_CHECK(q.size(3) == k.size(3),
              "FlashSCA SM90 q and k head dims must match");
}

void FlashSCACheckPackedQKVSM90(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
              "FlashSCA SM90 varlen q/k/v must be packed 3D tensors "
              "[total_tokens, H, D]");
  TORCH_CHECK(q.device().type() == torch::kCUDA &&
                  k.device().type() == torch::kCUDA &&
                  v.device().type() == torch::kCUDA,
              "FlashSCA SM90 varlen q/k/v must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == v.scalar_type(),
              "FlashSCA SM90 q/k/v must have the same dtype");
  TORCH_CHECK(q.scalar_type() == at::kHalf ||
                  q.scalar_type() == at::kBFloat16,
              "FlashSCA SM90 supports only fp16 and bf16 q, k, v input");
  TORCH_CHECK(k.size(1) == v.size(1),
              "FlashSCA SM90 k/v head counts must match");
  TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0 &&
                  q.size(1) % k.size(1) == 0,
              "FlashSCA SM90 q head count must be divisible by kv head count; "
              "got Hq=", q.size(1), ", Hkv=", k.size(1));
  TORCH_CHECK(k.size(0) == v.size(0),
              "FlashSCA SM90 varlen k/v total token counts must match");
  TORCH_CHECK(q.size(2) == k.size(2),
              "FlashSCA SM90 q and k head dims must match");
  TORCH_CHECK(q.size(2) > 0,
              "FlashSCA SM90 q/k head dim must be positive");
}

void FlashSCACheckBwdTensorsSM90(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse) {
  FlashSCACheckPackedQKVSM90(q, k, v);
  TORCH_CHECK(y_grad.dim() == 3 && y.dim() == 3,
              "FlashSCA SM90 varlen y_grad and y must be packed 3D tensors");
  TORCH_CHECK(y_grad.scalar_type() == q.scalar_type() &&
                  (y.scalar_type() == q.scalar_type() ||
                   y.scalar_type() == at::kFloat),
              "FlashSCA SM90 y_grad dtype must match q and y must either "
              "match q or be the float32 backward output state");
  TORCH_CHECK(y_grad.sizes() == y.sizes(),
              "FlashSCA SM90 y_grad and y shapes must match");
  TORCH_CHECK(y_grad.size(0) == q.size(0) && y_grad.size(1) == q.size(1) &&
                  y_grad.size(2) == v.size(2),
              "FlashSCA SM90 varlen y_grad/y shape must be [total_q, H, V]");
  TORCH_CHECK(lse.dim() == 2 && lse.scalar_type() == at::kFloat,
              "FlashSCA SM90 varlen lse must be a float32 [H, total_q] tensor");
  TORCH_CHECK(lse.size(0) == q.size(1) && lse.size(1) == q.size(0),
              "FlashSCA SM90 varlen lse shape must be [H, total_q]");
}

void FlashSCACheckDenseBwdTensorsSM90(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse) {
  FlashSCACheckDenseQKVSM90(q, k, v);
  TORCH_CHECK(y_grad.dim() == 4 && y.dim() == 4,
              "FlashSCA SM90 y_grad and y must be 4D tensors");
  TORCH_CHECK(y_grad.scalar_type() == q.scalar_type() &&
                  (y.scalar_type() == q.scalar_type() ||
                   y.scalar_type() == at::kFloat),
              "FlashSCA SM90 y_grad dtype must match q and y must either "
              "match q or be the float32 backward output state");
  TORCH_CHECK(y_grad.size(0) == y.size(0) && y_grad.size(1) == y.size(1) &&
                  y_grad.size(2) == y.size(2) && y_grad.size(3) == y.size(3),
              "FlashSCA SM90 y_grad and y shapes must match");
  TORCH_CHECK(y_grad.size(0) == q.size(0) && y_grad.size(1) == q.size(1) &&
                  y_grad.size(2) == q.size(2) && y_grad.size(3) == v.size(3),
              "FlashSCA SM90 y_grad/y shape must be [B, L, H, V]");
  TORCH_CHECK(lse.dim() == 3 && lse.scalar_type() == at::kFloat,
              "FlashSCA SM90 lse must be a float32 [B, H, L] tensor");
  TORCH_CHECK(lse.size(0) == q.size(0) && lse.size(1) == q.size(2) &&
                  lse.size(2) == q.size(1),
              "FlashSCA SM90 lse shape must be [B, H, L]");
}

void FlashSCACheckPrevChunkSM90(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& prev_k, const torch::Tensor& prev_v,
    int64_t chunk_size) {
  TORCH_CHECK(prev_k.dim() == 4 && prev_v.dim() == 4,
              "FlashSCA SM90 prev_k and prev_v must be 4D tensors");
  TORCH_CHECK(prev_k.scalar_type() == q.scalar_type() &&
                  prev_v.scalar_type() == q.scalar_type(),
              "FlashSCA SM90 prev_k/prev_v dtype must match q/k/v");
  TORCH_CHECK(prev_k.size(0) == q.size(0) && prev_v.size(0) == q.size(0),
              "FlashSCA SM90 prev_k/prev_v batch size must match q");
  TORCH_CHECK(prev_k.size(1) == chunk_size && prev_v.size(1) == chunk_size,
              "FlashSCA SM90 prev_k/prev_v sequence length must equal chunk_size");
  TORCH_CHECK(prev_k.size(2) == k.size(2) && prev_v.size(2) == v.size(2),
              "FlashSCA SM90 prev_k/prev_v head count must match k/v");
  TORCH_CHECK(prev_k.size(3) == k.size(3),
              "FlashSCA SM90 prev_k head dim must match k");
  TORCH_CHECK(prev_v.size(3) == v.size(3),
              "FlashSCA SM90 prev_v head dim must match v");
}

using FlashSCAVarlenMetadataSM90 =
    attention::sequence::VarlenMetadata;

FlashSCAVarlenMetadataSM90 FlashSCACheckVarlenMetadataSM90(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& cu_seqlens_q, const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k) {
  return attention::sequence::CheckVarlenMetadata(
      q, k, cu_seqlens_q, cu_seqlens_k,
      max_seqlen_q, max_seqlen_k, "FlashSCA SM90");
}

int64_t FlashSCARoundUp(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

void FlashSCAFillSM90DenseFwdParams(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, torch::Tensor& y, torch::Tensor& lse,
    const c10::optional<torch::Tensor>& output_state,
    const int64_t* q_segment_idx, const int64_t* k_segment_idx,
    int64_t k_segment_len, const int* q_chunk_positions,
    bool reset_chunk_pos_per_seq, bool strict_past,
    AttentionFwdParams& params) {
  std::memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = y.data_ptr();
  params.output_fp32 = y.scalar_type() == at::kFloat;
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    params.o_state_ptr = state.data_ptr<float>();
    params.o_state_batch_stride = state.stride(0);
    params.o_state_row_stride = state.stride(1);
    params.o_state_head_stride = state.stride(2);
  }
  params.softmax_lse_ptr = lse.data_ptr<float>();

  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.q_row_stride = q.stride(1);
  params.k_row_stride = k.stride(1);
  params.v_row_stride = v.stride(1);
  params.q_head_stride = q.stride(2);
  params.k_head_stride = k.stride(2);
  params.v_head_stride = v.stride(2);
  params.o_batch_stride = y.stride(0);
  params.o_row_stride = y.stride(1);
  params.o_head_stride = y.stride(2);

  params.b = static_cast<int>(q.size(0));
  params.num_sequences = params.b;
  params.seqlen_q = static_cast<int>(q.size(1));
  params.seqlen_k = static_cast<int>(k.size(1));
  params.max_seqlen_q = params.seqlen_q;
  params.max_seqlen_k = params.seqlen_k;
  params.d = static_cast<int>(q.size(3));
  params.dv = static_cast<int>(v.size(3));
  params.h = static_cast<int>(q.size(2));
  params.h_k = static_cast<int>(k.size(2));

  params.scale_softmax = scale;
  params.window_size_left = static_cast<int>(2 * chunk_size - 1);
  params.window_size_right = strict_past ? -1 : 0;
  params.attention_chunk = reset_chunk_pos_per_seq
      ? 0
      : static_cast<int>(chunk_size);
  params.q_segment_idx = q_segment_idx;
  params.k_segment_idx = k_segment_idx;
  params.k_segment_len = static_cast<int>(k_segment_len);
  params.q_position_offsets = nullptr;
  params.q_chunk_positions = q_chunk_positions;
  params.reset_attention_chunk = reset_chunk_pos_per_seq
      ? static_cast<int>(chunk_size)
      : 0;
  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqused_k = nullptr;
  params.tile_count_semaphore = nullptr;

  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  params.num_sm = prop->multiProcessorCount;
}

void FlashSCAFillSM90VarlenFwdParams(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, torch::Tensor& y, torch::Tensor& lse,
    const c10::optional<torch::Tensor>& output_state,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    const int* seqused_k,
    int num_sequences, int64_t max_seqlen_q, int64_t max_seqlen_k,
    const int* q_position_offsets, bool strict_past,
    AttentionFwdParams& params) {
  std::memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = y.data_ptr();
  params.output_fp32 = y.scalar_type() == at::kFloat;
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    params.o_state_ptr = state.data_ptr<float>();
    params.o_state_batch_stride = 0;
    params.o_state_row_stride = state.stride(0);
    params.o_state_head_stride = state.stride(1);
  }
  params.softmax_lse_ptr = lse.data_ptr<float>();

  params.q_batch_stride = 0;
  params.k_batch_stride = 0;
  params.v_batch_stride = 0;
  params.q_row_stride = q.stride(0);
  params.k_row_stride = k.stride(0);
  params.v_row_stride = v.stride(0);
  params.q_head_stride = q.stride(1);
  params.k_head_stride = k.stride(1);
  params.v_head_stride = v.stride(1);
  params.o_batch_stride = 0;
  params.o_row_stride = y.stride(0);
  params.o_head_stride = y.stride(1);

  params.b = 1;
  params.num_sequences = num_sequences;
  params.seqlen_q = static_cast<int>(q.size(0));
  params.seqlen_k = static_cast<int>(k.size(0));
  params.max_seqlen_q = static_cast<int>(max_seqlen_q);
  params.max_seqlen_k = static_cast<int>(max_seqlen_k);
  params.d = static_cast<int>(q.size(2));
  params.dv = static_cast<int>(v.size(2));
  params.h = static_cast<int>(q.size(1));
  params.h_k = static_cast<int>(k.size(1));

  params.scale_softmax = scale;
  params.window_size_left = static_cast<int>(2 * chunk_size - 1);
  params.window_size_right = strict_past ? -1 : 0;
  params.attention_chunk = static_cast<int>(chunk_size);
  params.q_segment_idx = nullptr;
  params.k_segment_idx = nullptr;
  params.k_segment_len = 0;
  params.q_position_offsets = q_position_offsets;
  params.q_chunk_positions = nullptr;
  params.reset_attention_chunk = 0;
  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_k = cu_seqlens_k;
  params.seqused_k = seqused_k;

  params.tile_count_semaphore = nullptr;

  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  params.num_sm = prop->multiProcessorCount;
}

void FlashSCAFillSM90DenseBwdParams(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, torch::Tensor& dq, torch::Tensor& dk,
    torch::Tensor& dv, const int64_t* q_segment_idx,
    const int64_t* k_segment_idx, int64_t k_segment_len,
    const int* q_chunk_positions, bool reset_chunk_pos_per_seq,
    bool strict_past, int odd_head_window_right_delta,
    AttentionBwdParams& params) {
  std::memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.dy_ptr = dy.data_ptr();
  params.y_ptr = y.data_ptr();
  params.y_is_fp32 = y.scalar_type() == at::kFloat;
  params.lse_ptr = lse.data_ptr<float>();
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();

  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.dy_batch_stride = dy.stride(0);
  params.y_batch_stride = y.stride(0);
  params.dq_batch_stride = dq.stride(0);
  params.dk_batch_stride = dk.stride(0);
  params.dv_batch_stride = dv.stride(0);
  params.q_row_stride = q.stride(1);
  params.k_row_stride = k.stride(1);
  params.v_row_stride = v.stride(1);
  params.dy_row_stride = dy.stride(1);
  params.y_row_stride = y.stride(1);
  params.dq_row_stride = dq.stride(1);
  params.dk_row_stride = dk.stride(1);
  params.dv_row_stride = dv.stride(1);
  params.q_head_stride = q.stride(2);
  params.k_head_stride = k.stride(2);
  params.v_head_stride = v.stride(2);
  params.dy_head_stride = dy.stride(2);
  params.y_head_stride = y.stride(2);
  params.dq_head_stride = dq.stride(2);
  params.dk_head_stride = dk.stride(2);
  params.dv_head_stride = dv.stride(2);

  params.b = static_cast<int>(q.size(0));
  params.num_sequences = params.b;
  params.seqlen_q = static_cast<int>(q.size(1));
  params.seqlen_k = static_cast<int>(k.size(1));
  params.max_seqlen_q = params.seqlen_q;
  params.max_seqlen_k = params.seqlen_k;
  params.seqlen_q_padded = static_cast<int>(
      FlashSCARoundUp(params.seqlen_q, 128));
  params.d = static_cast<int>(q.size(3));
  params.dv = static_cast<int>(v.size(3));
  params.h = static_cast<int>(q.size(2));
  params.h_k = static_cast<int>(k.size(2));

  params.scale_softmax = scale;
  params.window_size_left = static_cast<int>(2 * chunk_size - 1);
  params.window_size_right = strict_past ? -1 : 0;
  params.odd_head_window_right_delta = odd_head_window_right_delta;
  params.attention_chunk = reset_chunk_pos_per_seq
      ? 0
      : static_cast<int>(chunk_size);
  params.q_segment_idx = q_segment_idx;
  params.k_segment_idx = k_segment_idx;
  params.k_segment_len = static_cast<int>(k_segment_len);
  params.q_position_offsets = nullptr;
  params.q_chunk_positions = q_chunk_positions;
  params.reset_attention_chunk = reset_chunk_pos_per_seq
      ? static_cast<int>(chunk_size)
      : 0;
  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqused_k = nullptr;

  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  params.num_sm = prop->multiProcessorCount;
}

void FlashSCAFillSM90VarlenBwdParams(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, torch::Tensor& dq, torch::Tensor& dk,
    torch::Tensor& dv, const int* cu_seqlens_q, const int* cu_seqlens_k,
    const int* seqused_k,
    int num_sequences, int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t seqlen_q_padded, const int* q_position_offsets,
    bool strict_past, AttentionBwdParams& params) {
  std::memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.dy_ptr = dy.data_ptr();
  params.y_ptr = y.data_ptr();
  params.y_is_fp32 = y.scalar_type() == at::kFloat;
  params.lse_ptr = lse.data_ptr<float>();
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();

  params.q_batch_stride = 0;
  params.k_batch_stride = 0;
  params.v_batch_stride = 0;
  params.dy_batch_stride = 0;
  params.y_batch_stride = 0;
  params.dq_batch_stride = 0;
  params.dk_batch_stride = 0;
  params.dv_batch_stride = 0;
  params.q_row_stride = q.stride(0);
  params.k_row_stride = k.stride(0);
  params.v_row_stride = v.stride(0);
  params.dy_row_stride = dy.stride(0);
  params.y_row_stride = y.stride(0);
  params.dq_row_stride = dq.stride(0);
  params.dk_row_stride = dk.stride(0);
  params.dv_row_stride = dv.stride(0);
  params.q_head_stride = q.stride(1);
  params.k_head_stride = k.stride(1);
  params.v_head_stride = v.stride(1);
  params.dy_head_stride = dy.stride(1);
  params.y_head_stride = y.stride(1);
  params.dq_head_stride = dq.stride(1);
  params.dk_head_stride = dk.stride(1);
  params.dv_head_stride = dv.stride(1);

  params.b = 1;
  params.num_sequences = num_sequences;
  params.seqlen_q = static_cast<int>(q.size(0));
  params.seqlen_k = static_cast<int>(k.size(0));
  params.max_seqlen_q = static_cast<int>(max_seqlen_q);
  params.max_seqlen_k = static_cast<int>(max_seqlen_k);
  params.seqlen_q_padded = static_cast<int>(seqlen_q_padded);
  params.d = static_cast<int>(q.size(2));
  params.dv = static_cast<int>(v.size(2));
  params.h = static_cast<int>(q.size(1));
  params.h_k = static_cast<int>(k.size(1));

  params.scale_softmax = scale;
  params.window_size_left = static_cast<int>(2 * chunk_size - 1);
  params.window_size_right = strict_past ? -1 : 0;
  params.attention_chunk = static_cast<int>(chunk_size);
  params.q_segment_idx = nullptr;
  params.k_segment_idx = nullptr;
  params.k_segment_len = 0;
  params.q_position_offsets = q_position_offsets;
  params.q_chunk_positions = nullptr;
  params.reset_attention_chunk = 0;
  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_k = cu_seqlens_k;
  params.seqused_k = seqused_k;

  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  params.num_sm = prop->multiProcessorCount;
}

template <typename Element, typename ElementOut>
void FlashSCASM90DenseFwdImpl(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, const int64_t* q_segment_idx,
    const int64_t* k_segment_idx, int64_t k_segment_len,
    const int* q_chunk_positions, bool reset_chunk_pos_per_seq,
    bool strict_past, torch::Tensor& y, torch::Tensor& lse,
    const c10::optional<torch::Tensor>& output_state) {
  TORCH_CHECK(q.size(2) % k.size(2) == 0,
              "FlashSCA SM90 FWD requires Hq divisible by Hkv");
  TORCH_CHECK(q.size(3) <= 256 && v.size(3) <= 256,
              "FlashSCA SM90 FWD expects D,V <= 256; got D=",
              q.size(3), ", V=", v.size(3));
  TORCH_CHECK(chunk_size <= (int64_t(1) << 30),
              "FlashSCA SM90 FWD chunk_size is too large: ",
              chunk_size);
  TORCH_CHECK(k_segment_len <= std::numeric_limits<int>::max(),
              "FlashSCA SM90 FWD k_segment_idx is too long: ",
              k_segment_len);

  AttentionFwdParams params;
  FlashSCAFillSM90DenseFwdParams(
      q, k, v, chunk_size, scale, y, lse, output_state,
      q_segment_idx, k_segment_idx,
      k_segment_len, q_chunk_positions, reset_chunk_pos_per_seq,
      strict_past, params);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  using CompatEntry =
      flash_sca::plans::ScaDenseFwdCompatRegistryEntrySm90;
  using CompatPlan = typename CompatEntry::Plan;
  static_assert(
      CompatEntry::kAllowsSegment && !CompatEntry::kVarlen,
      "FlashSCA SM90 dense FWD registry entry is inconsistent");
  RunFlashSCAFwdSm90<Element, ElementOut>(
      params, cuda_stream, CompatPlan::kVarlen);
}

template <typename Element, typename ElementOut>
void FlashSCASM90FwdImpl(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, const int* cu_seqlens_q,
    const int* cu_seqlens_k, const int* seqused_k, int num_sequences,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    const int* q_position_offsets, bool strict_past, torch::Tensor& y,
    torch::Tensor& lse,
    const c10::optional<torch::Tensor>& output_state) {
  TORCH_CHECK(q.size(1) % k.size(1) == 0,
              "FlashSCA SM90 FWD requires Hq divisible by Hkv");
  TORCH_CHECK(q.size(2) <= 256 && v.size(2) <= 256,
              "FlashSCA SM90 FWD expects D,V <= 256; got D=",
              q.size(2), ", V=", v.size(2));
  TORCH_CHECK(chunk_size <= (int64_t(1) << 30),
              "FlashSCA SM90 FWD chunk_size is too large: ",
              chunk_size);

  AttentionFwdParams params;
  FlashSCAFillSM90VarlenFwdParams(
      q, k, v, chunk_size, scale, y, lse, output_state,
      cu_seqlens_q, cu_seqlens_k,
      seqused_k,
      num_sequences, max_seqlen_q, max_seqlen_k, q_position_offsets,
      strict_past, params);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  using CompatEntry =
      flash_sca::plans::ScaVarlenFwdCompatRegistryEntrySm90;
  using CompatPlan = typename CompatEntry::Plan;
  static_assert(
      CompatEntry::kVarlen && !CompatEntry::kAllowsSegment,
      "FlashSCA SM90 varlen FWD registry entry is inconsistent");
  RunFlashSCAFwdSm90<Element, ElementOut>(
      params, cuda_stream, CompatPlan::kVarlen);
}

template <typename Element>
void FlashSCASM90DenseBwdImpl(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, const int64_t* q_segment_idx,
    const int64_t* k_segment_idx, int64_t k_segment_len,
    const int* q_chunk_positions, bool reset_chunk_pos_per_seq,
    torch::Tensor& dq, torch::Tensor& dk, torch::Tensor& dv,
    bool deterministic, bool strict_past,
    int odd_head_window_right_delta, bool native_small_chunk_det) {
  TORCH_CHECK(q.size(2) % k.size(2) == 0,
              "FlashSCA SM90 BWD requires Hq divisible by Hkv");
  TORCH_CHECK(q.size(3) == k.size(3),
              "FlashSCA SM90 BWD expects padded q/k to share D");
  TORCH_CHECK(v.size(3) == dy.size(3) && v.size(3) == y.size(3),
              "FlashSCA SM90 BWD expects padded v/dy/y to share V");
  TORCH_CHECK(q.size(3) <= 256 && v.size(3) <= 256,
              "FlashSCA SM90 BWD expects padded D,V <= 256; got D=",
              q.size(3), ", V=", v.size(3));
  TORCH_CHECK(chunk_size <= (int64_t(1) << 30),
              "FlashSCA SM90 BWD chunk_size is too large: ",
              chunk_size);
  TORCH_CHECK(k_segment_len <= std::numeric_limits<int>::max(),
              "FlashSCA SM90 BWD k_segment_idx is too long: ",
              k_segment_len);

  AttentionBwdParams params;
  FlashSCAFillSM90DenseBwdParams(
      dy, q, k, v, y, lse, chunk_size, scale, dq, dk, dv,
      q_segment_idx, k_segment_idx, k_segment_len, q_chunk_positions,
      reset_chunk_pos_per_seq, strict_past,
      odd_head_window_right_delta, params);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  using CompatEntry =
      flash_sca::plans::ScaDenseBwdCompatRegistryEntrySm90;
  using CompatPlan = typename CompatEntry::Plan;
  static_assert(
      CompatEntry::kAllowsSegment && !CompatEntry::kVarlen &&
          CompatPlan::kRuntimeDeterminism,
      "FlashSCA SM90 dense BWD registry entry is inconsistent");
  RunFlashSCABwdSm90<Element>(
      params, deterministic, cuda_stream, native_small_chunk_det);
}

template <typename Element>
void FlashSCASM90BwdImpl(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, const int* cu_seqlens_q,
    const int* cu_seqlens_k, const int* seqused_k, int num_sequences,
    int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t seqlen_q_padded,
    const int* q_position_offsets, bool strict_past,
    torch::Tensor& dq, torch::Tensor& dk, torch::Tensor& dv,
    bool deterministic) {
  TORCH_CHECK(q.size(1) % k.size(1) == 0,
              "FlashSCA SM90 BWD requires Hq divisible by Hkv");
  TORCH_CHECK(q.size(2) == k.size(2),
              "FlashSCA SM90 BWD expects padded q/k to share D");
  TORCH_CHECK(v.size(2) == dy.size(2) && v.size(2) == y.size(2),
              "FlashSCA SM90 BWD expects padded v/dy/y to share V");
  TORCH_CHECK(q.size(2) <= 256 && v.size(2) <= 256,
              "FlashSCA SM90 BWD expects padded D,V <= 256; got D=",
              q.size(2), ", V=", v.size(2));
  TORCH_CHECK(chunk_size <= (int64_t(1) << 30),
              "FlashSCA SM90 BWD chunk_size is too large: ",
              chunk_size);

  AttentionBwdParams params;
  FlashSCAFillSM90VarlenBwdParams(
      dy, q, k, v, y, lse, chunk_size, scale, dq, dk, dv,
      cu_seqlens_q, cu_seqlens_k, seqused_k, num_sequences, max_seqlen_q,
      max_seqlen_k,
      seqlen_q_padded, q_position_offsets, strict_past, params);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  using CompatEntry =
      flash_sca::plans::ScaVarlenBwdCompatRegistryEntrySm90;
  using CompatPlan = typename CompatEntry::Plan;
  static_assert(
      CompatEntry::kVarlen && !CompatEntry::kAllowsSegment &&
          CompatPlan::kRuntimeDeterminism,
      "FlashSCA SM90 varlen BWD registry entry is inconsistent");
  RunFlashSCABwdSm90<Element>(params, deterministic, cuda_stream);
}

__device__ int FlashSCASegmentLowerBoundDevice(
    const int64_t* segment, int begin, int end, int64_t value) {
  int lo = begin;
  int hi = end;
  while (lo < hi) {
    int const mid = lo + ((hi - lo) >> 1);
    if (segment[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ int FlashSCASegmentUpperBoundDevice(
    const int64_t* segment, int begin, int end, int64_t value) {
  int lo = begin;
  int hi = end;
  while (lo < hi) {
    int const mid = lo + ((hi - lo) >> 1);
    if (segment[mid] <= value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__global__ void FlashSCABuildDenseResetChunkPositionsKernel(
    const int64_t* q_segment_idx, const int64_t* k_segment_idx,
    int* q_chunk_positions, int batch, int seqlen_q, int seqlen_k) {
  int const linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int const q_total = batch * seqlen_q;
  if (linear < q_total) {
    int const bidb = linear / seqlen_q;
    int const pos = linear - bidb * seqlen_q;
    const int64_t* q_row = q_segment_idx + bidb * seqlen_q;
    const int64_t* k_row = k_segment_idx + bidb * seqlen_k;
    int64_t const segment = q_row[pos];
    int const q_start =
        FlashSCASegmentLowerBoundDevice(q_row, 0, pos + 1, segment);
    int const q_end =
        FlashSCASegmentUpperBoundDevice(q_row, pos, seqlen_q, segment);
    int const k_start =
        FlashSCASegmentLowerBoundDevice(k_row, 0, seqlen_k, segment);
    int const k_end =
        FlashSCASegmentUpperBoundDevice(k_row, k_start, seqlen_k, segment);
    // Varlen causal alignment: Q/K run ends coincide.
    q_chunk_positions[linear] =
        pos - q_start + (k_end - k_start) - (q_end - q_start);
  }
}

__global__ void FlashSCAZeroRightAlignedKVPrefixKernel(
    uint16_t* dk, uint16_t* dv, int64_t dk_elements_per_token,
    int64_t dv_elements_per_token, const int* cu_seqlens_k_aligned) {
  const int64_t prefix_tokens = cu_seqlens_k_aligned[0];
  const int64_t dk_elements = prefix_tokens * dk_elements_per_token;
  const int64_t total_elements =
      dk_elements + prefix_tokens * dv_elements_per_token;
  for (int64_t linear =
           int64_t(blockIdx.x) * blockDim.x + int64_t(threadIdx.x);
       linear < total_elements;
       linear += int64_t(gridDim.x) * blockDim.x) {
    if (linear < dk_elements) {
      dk[linear] = 0;
    } else {
      dv[linear - dk_elements] = 0;
    }
  }
}

__device__ int FlashSCALowerBoundInt32Device(
    const int* values, int count, int target) {
  int lo = 0;
  int hi = count;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    if (values[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__global__ void FlashSCABuildRowAlignedMetadataKernel(
    const int* cu_seqlens_q, int cu_q_count,
    const int* cu_seqlens_k, int cu_k_count,
    int batch, int q_row_length, int k_row_length,
    int* k_run_starts, int* k_run_lengths, int* k_prefix_ends,
    int* q_row_positions) {
  const int q_run = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  const int num_q_runs = cu_q_count - 1;
  if (q_run >= num_q_runs) {
    return;
  }

  const int q_start = cu_seqlens_q[q_run];
  const int row = q_start / q_row_length;
  if (row < 0 || row >= batch) {
    // Invalid row metadata maps to an empty preprocessing range.
    k_run_starts[q_run] = 0;
    k_run_lengths[q_run] = 0;
    q_row_positions[q_run] = 0;
    return;
  }

  const int q_row_start = row * q_row_length;
  const int q_row_end = q_row_start + q_row_length;
  const int k_row_start = row * k_row_length;
  const int k_row_end = k_row_start + k_row_length;
  q_row_positions[q_run] = q_start - q_row_start;
  const int q_first =
      FlashSCALowerBoundInt32Device(cu_seqlens_q, cu_q_count, q_row_start);
  const int q_end =
      FlashSCALowerBoundInt32Device(cu_seqlens_q, cu_q_count, q_row_end);
  const int k_first =
      FlashSCALowerBoundInt32Device(cu_seqlens_k, cu_k_count, k_row_start);
  const int k_end =
      FlashSCALowerBoundInt32Device(cu_seqlens_k, cu_k_count, k_row_end);
  const int reverse_run = q_end - 1 - q_run;
  const int k_run = k_end - 1 - reverse_run;

  // Row-aligned metadata contains every row boundary and each row has
  // at least as many K runs as Q runs.
  const bool valid =
      q_first < cu_q_count && q_end < cu_q_count &&
      k_first < cu_k_count && k_end < cu_k_count &&
      cu_seqlens_q[q_first] == q_row_start &&
      cu_seqlens_q[q_end] == q_row_end &&
      cu_seqlens_k[k_first] == k_row_start &&
      cu_seqlens_k[k_end] == k_row_end &&
      k_run >= k_first && k_run + 1 < cu_k_count;
  if (!valid) {
    k_run_starts[q_run] = 0;
    k_run_lengths[q_run] = 0;
    if (q_start == q_row_start) {
      k_prefix_ends[row] = k_row_start;
    }
    return;
  }

  const int k_start = cu_seqlens_k[k_run];
  k_run_starts[q_run] = k_start;
  k_run_lengths[q_run] = cu_seqlens_k[k_run + 1] - k_start;
  if (q_start == q_row_start) {
    k_prefix_ends[row] = k_start;
  }
}

__global__ void FlashSCAZeroRowAlignedKVPrefixesKernel(
    uint16_t* dk, uint16_t* dv, int64_t dk_elements_per_token,
    int64_t dv_elements_per_token, const int* k_prefix_ends,
    int k_row_length, int blocks_per_row) {
  const int row = int(blockIdx.x) / blocks_per_row;
  const int row_block = int(blockIdx.x) - row * blocks_per_row;
  const int row_start = row * k_row_length;
  const int64_t prefix_tokens =
      int64_t(k_prefix_ends[row]) - int64_t(row_start);
  const int64_t dk_elements = prefix_tokens * dk_elements_per_token;
  const int64_t total_elements =
      dk_elements + prefix_tokens * dv_elements_per_token;
  for (int64_t linear =
           int64_t(row_block) * blockDim.x + int64_t(threadIdx.x);
       linear < total_elements;
       linear += int64_t(blocks_per_row) * blockDim.x) {
    if (linear < dk_elements) {
      dk[int64_t(row_start) * dk_elements_per_token + linear] = 0;
    } else {
      dv[int64_t(row_start) * dv_elements_per_token +
         linear - dk_elements] = 0;
    }
  }
}

void FlashSCAZeroRightAlignedKVPrefix(
    torch::Tensor& dk, torch::Tensor& dv,
    const int* cu_seqlens_k_aligned) {
  TORCH_CHECK(dk.element_size() == sizeof(uint16_t) &&
                  dv.element_size() == sizeof(uint16_t),
              "FlashSCA SM90 prefix zero expects 16-bit compute gradients");
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 32;
  const int64_t capacity_elements = dk.numel() + dv.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      kMaxBlocks, (capacity_elements + kThreads - 1) / kThreads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  FlashSCAZeroRightAlignedKVPrefixKernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<uint16_t*>(dk.data_ptr()),
      reinterpret_cast<uint16_t*>(dv.data_ptr()), dk.size(1) * dk.size(2),
      dv.size(1) * dv.size(2), cu_seqlens_k_aligned);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void FlashSCAZeroRowAlignedKVPrefixes(
    torch::Tensor& dk, torch::Tensor& dv,
    const torch::Tensor& k_prefix_ends, int64_t k_row_length) {
  TORCH_CHECK(dk.element_size() == sizeof(uint16_t) &&
                  dv.element_size() == sizeof(uint16_t),
              "FlashSCA SM90 row-prefix zero expects 16-bit compute gradients");
  TORCH_CHECK(k_prefix_ends.numel() > 0 && k_row_length > 0,
              "FlashSCA SM90 row-prefix metadata must be nonempty");
  constexpr int kThreads = 256;
  constexpr int kBlocksPerRow = 8;
  const int64_t grid = k_prefix_ends.numel() * kBlocksPerRow;
  TORCH_CHECK(grid <= std::numeric_limits<int>::max(),
              "FlashSCA SM90 row-prefix zero grid must fit int32");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  FlashSCAZeroRowAlignedKVPrefixesKernel
      <<<static_cast<int>(grid), kThreads, 0, stream>>>(
          reinterpret_cast<uint16_t*>(dk.data_ptr()),
          reinterpret_cast<uint16_t*>(dv.data_ptr()),
          dk.size(1) * dk.size(2), dv.size(1) * dv.size(2),
          k_prefix_ends.data_ptr<int>(), static_cast<int>(k_row_length),
          kBlocksPerRow);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

struct FlashSCADenseResetPositionCache {
  torch::Tensor q_segment_idx;
  torch::Tensor k_segment_idx;
  torch::Tensor q_chunk_positions;
  int64_t q_version = -1;
  int64_t k_version = -1;
  int device = -1;
  cudaStream_t stream = nullptr;
};

torch::Tensor FlashSCAGetDenseResetChunkPositions(
    const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static thread_local FlashSCADenseResetPositionCache cache;
  bool const cache_hit =
      cache.q_chunk_positions.defined() &&
      cache.q_segment_idx.unsafeGetTensorImpl() ==
          q_segment_idx.unsafeGetTensorImpl() &&
      cache.k_segment_idx.unsafeGetTensorImpl() ==
          k_segment_idx.unsafeGetTensorImpl() &&
      cache.q_version == q_segment_idx._version() &&
      cache.k_version == k_segment_idx._version() &&
      cache.device == q_segment_idx.get_device() && cache.stream == stream;
  if (cache_hit) {
    return cache.q_chunk_positions;
  }

  torch::Tensor q_positions = torch::empty(
      q_segment_idx.sizes(),
      q_segment_idx.options().dtype(at::kInt).memory_format(
          at::MemoryFormat::Contiguous));
  int const batch = static_cast<int>(q_segment_idx.size(0));
  int const seqlen_q = static_cast<int>(q_segment_idx.size(1));
  int const seqlen_k = static_cast<int>(k_segment_idx.size(1));
  int const total = batch * seqlen_q;
  constexpr int kThreads = 256;
  int const blocks = (total + kThreads - 1) / kThreads;
  FlashSCABuildDenseResetChunkPositionsKernel
      <<<blocks, kThreads, 0, stream>>>(
          q_segment_idx.data_ptr<int64_t>(),
          k_segment_idx.data_ptr<int64_t>(), q_positions.data_ptr<int>(),
          batch, seqlen_q, seqlen_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (FlashSCATensorCacheable(q_segment_idx) &&
      FlashSCATensorCacheable(k_segment_idx)) {
    cache.q_segment_idx = q_segment_idx;
    cache.k_segment_idx = k_segment_idx;
    cache.q_chunk_positions = q_positions;
    cache.q_version = q_segment_idx._version();
    cache.k_version = k_segment_idx._version();
    cache.device = q_segment_idx.get_device();
    cache.stream = stream;
  }
  return q_positions;
}

}  // namespace

bool FlashSCASM90SegmentMetadataIsDense(
    const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx) {
  TORCH_CHECK(
      q_segment_idx.dim() == 2 && k_segment_idx.dim() == 2 &&
          q_segment_idx.scalar_type() == at::kLong &&
          k_segment_idx.scalar_type() == at::kLong &&
          q_segment_idx.device().type() == torch::kCUDA &&
          k_segment_idx.device() == q_segment_idx.device(),
      "FlashSCA dense segment planning expects same-device CUDA int64 "
      "[B, Lq]/[B, Lk] tensors");
  TORCH_CHECK(
      q_segment_idx.size(0) > 0 && q_segment_idx.size(1) > 0 &&
          k_segment_idx.size(0) == q_segment_idx.size(0) &&
          k_segment_idx.size(1) > 0,
      "FlashSCA dense segment planning expects nonempty metadata with "
      "matching batch sizes");
  TORCH_CHECK(
      q_segment_idx.size(0) <= std::numeric_limits<int>::max() &&
          q_segment_idx.size(1) <= std::numeric_limits<int>::max() &&
          k_segment_idx.size(1) <= std::numeric_limits<int>::max(),
      "FlashSCA dense segment planning dimensions must fit int32");

  at::cuda::OptionalCUDAGuard guard(at::device_of(q_segment_idx));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (FlashSCASegmentDensePlanCacheEntry* entry =
          FlashSCAFindSegmentDensePlanCacheEntry(
              q_segment_idx, k_segment_idx, stream)) {
    return entry->is_dense;
  }

  torch::Tensor q_segment_idx_c = q_segment_idx.contiguous();
  torch::Tensor k_segment_idx_c = k_segment_idx.contiguous();
  torch::Tensor mismatch = torch::zeros(
      {},
      q_segment_idx.options()
          .dtype(at::kInt)
          .memory_format(at::MemoryFormat::Contiguous));
  const int batch = static_cast<int>(q_segment_idx.size(0));
  const int q_length = static_cast<int>(q_segment_idx.size(1));
  const int k_length = static_cast<int>(k_segment_idx.size(1));
  const int64_t total_values =
      int64_t(batch) * q_length + int64_t(batch) * k_length + batch;
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      kMaxBlocks,
      std::max<int64_t>(1, (total_values + kThreads - 1) / kThreads)));
  FlashSCACheckSegmentMetadataDenseKernel<<<blocks, kThreads, 0, stream>>>(
      q_segment_idx_c.data_ptr<int64_t>(),
      k_segment_idx_c.data_ptr<int64_t>(), batch, q_length, k_length,
      total_values, mismatch.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const bool is_dense = mismatch.cpu().item<int>() == 0;
  FlashSCAStoreSegmentDensePlan(
      q_segment_idx, k_segment_idx, stream, is_dense);
  return is_dense;
}

bool FlashSCASM90VarlenMetadataIsDense(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t batch,
    int64_t q_row_length,
    int64_t k_row_length,
    const c10::optional<torch::Tensor>& position_offsets) {
  TORCH_CHECK(
      cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1 &&
          cu_seqlens_q.scalar_type() == at::kInt &&
          cu_seqlens_k.scalar_type() == at::kInt &&
          cu_seqlens_q.device().type() == torch::kCUDA &&
          cu_seqlens_k.device() == cu_seqlens_q.device(),
      "FlashSCA dense varlen planning expects same-device CUDA int32 "
      "cu_seqlens_q/cu_seqlens_k");
  TORCH_CHECK(
      batch > 0 && q_row_length > 0 && k_row_length > 0 &&
          batch <= std::numeric_limits<int>::max() &&
          q_row_length <= std::numeric_limits<int>::max() &&
          k_row_length <= std::numeric_limits<int>::max() &&
          batch <= std::numeric_limits<int>::max() / q_row_length &&
          batch <= std::numeric_limits<int>::max() / k_row_length,
      "FlashSCA dense varlen planning dimensions must be positive and fit "
      "int32");
  if (position_offsets.has_value()) {
    TORCH_CHECK(
        position_offsets.value().device() == cu_seqlens_q.device() &&
            position_offsets.value().scalar_type() == at::kInt &&
            position_offsets.value().dim() == 1 &&
            position_offsets.value().numel() == batch,
        "FlashSCA dense varlen planning position_offsets must be a "
        "same-device CUDA int32 [B] tensor");
  }

  at::cuda::OptionalCUDAGuard guard(at::device_of(cu_seqlens_q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (FlashSCAVarlenDensePlanCacheEntry* entry =
          FlashSCAFindVarlenDensePlanCacheEntry(
              cu_seqlens_q, cu_seqlens_k, batch, q_row_length,
              k_row_length, position_offsets, stream)) {
    return entry->is_dense;
  }

  bool is_dense =
      cu_seqlens_q.numel() == batch + 1 &&
      cu_seqlens_k.numel() == batch + 1;
  if (is_dense) {
    torch::Tensor cu_q_cpu = cu_seqlens_q.contiguous().cpu();
    torch::Tensor cu_k_cpu = cu_seqlens_k.contiguous().cpu();
    const int* cu_q = cu_q_cpu.data_ptr<int>();
    const int* cu_k = cu_k_cpu.data_ptr<int>();
    for (int64_t row = 0; row <= batch; ++row) {
      if (cu_q[row] != row * q_row_length ||
          cu_k[row] != row * k_row_length) {
        is_dense = false;
        break;
      }
    }
  }
  if (is_dense && position_offsets.has_value()) {
    torch::Tensor position_offsets_cpu =
        position_offsets.value().contiguous().cpu();
    const int* offsets = position_offsets_cpu.data_ptr<int>();
    for (int64_t row = 0; row < batch; ++row) {
      if (offsets[row] != 0) {
        is_dense = false;
        break;
      }
    }
  }
  FlashSCAStoreVarlenDensePlan(
      cu_seqlens_q, cu_seqlens_k, batch, q_row_length, k_row_length,
      position_offsets, stream, is_dense);
  return is_dense;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
FlashSCASM90BuildRowAlignedMetadata(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t batch, int64_t q_row_length, int64_t k_row_length) {
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1 &&
                  cu_seqlens_q.scalar_type() == at::kInt &&
                  cu_seqlens_k.scalar_type() == at::kInt &&
                  cu_seqlens_q.device().type() == torch::kCUDA &&
                  cu_seqlens_k.device() == cu_seqlens_q.device(),
              "FlashSCA SM90 row alignment expects same-device CUDA int32 "
              "1D cu_seqlens_q/cu_seqlens_k");
  TORCH_CHECK(batch > 1 && q_row_length > 0 && k_row_length > 0,
              "FlashSCA SM90 row alignment expects B > 1 and positive row lengths");
  TORCH_CHECK(batch <= std::numeric_limits<int>::max() &&
                  q_row_length <= std::numeric_limits<int>::max() &&
                  k_row_length <= std::numeric_limits<int>::max() &&
                  cu_seqlens_q.numel() <= std::numeric_limits<int>::max() &&
                  cu_seqlens_k.numel() <= std::numeric_limits<int>::max(),
              "FlashSCA SM90 row-alignment metadata must fit int32");
  TORCH_CHECK(batch <= std::numeric_limits<int>::max() / q_row_length &&
                  batch <= std::numeric_limits<int>::max() / k_row_length,
              "FlashSCA SM90 flattened row lengths must fit int32");
  TORCH_CHECK(cu_seqlens_q.numel() >= batch + 1 &&
                  cu_seqlens_k.numel() >= cu_seqlens_q.numel(),
              "FlashSCA SM90 row alignment requires at least one Q run per "
              "row and at least as many total K runs as Q runs");

  at::cuda::OptionalCUDAGuard guard(at::device_of(cu_seqlens_q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  struct RowAlignedMetadataCache {
    torch::Tensor cu_q;
    torch::Tensor cu_k;
    torch::Tensor k_run_starts;
    torch::Tensor k_run_lengths;
    torch::Tensor k_prefix_ends;
    torch::Tensor q_row_positions;
    int64_t q_version = -1;
    int64_t k_version = -1;
    int64_t batch = -1;
    int64_t q_row_length = -1;
    int64_t k_row_length = -1;
    int device = -1;
    cudaStream_t stream = nullptr;
  };
  static thread_local RowAlignedMetadataCache cache;
  const bool cache_hit =
      cache.k_run_starts.defined() &&
      cache.cu_q.unsafeGetTensorImpl() == cu_seqlens_q.unsafeGetTensorImpl() &&
      cache.cu_k.unsafeGetTensorImpl() == cu_seqlens_k.unsafeGetTensorImpl() &&
      cache.q_version == cu_seqlens_q._version() &&
      cache.k_version == cu_seqlens_k._version() &&
      cache.batch == batch && cache.q_row_length == q_row_length &&
      cache.k_row_length == k_row_length &&
      cache.device == cu_seqlens_q.get_device() && cache.stream == stream;
  if (cache_hit) {
    return std::make_tuple(
        cache.k_run_starts, cache.k_run_lengths, cache.k_prefix_ends,
        cache.q_row_positions);
  }

  torch::Tensor cu_q_c = cu_seqlens_q.contiguous();
  torch::Tensor cu_k_c = cu_seqlens_k.contiguous();
  const int64_t num_q_runs = cu_q_c.numel() - 1;
  torch::Tensor k_run_starts = torch::empty(
      {num_q_runs}, cu_q_c.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor k_run_lengths = torch::empty_like(k_run_starts);
  torch::Tensor k_prefix_ends = torch::empty(
      {batch}, cu_q_c.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor q_row_positions = torch::empty_like(k_run_starts);

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((num_q_runs + kThreads - 1) / kThreads);
  FlashSCABuildRowAlignedMetadataKernel<<<blocks, kThreads, 0, stream>>>(
      cu_q_c.data_ptr<int>(), static_cast<int>(cu_q_c.numel()),
      cu_k_c.data_ptr<int>(), static_cast<int>(cu_k_c.numel()),
      static_cast<int>(batch), static_cast<int>(q_row_length),
      static_cast<int>(k_row_length), k_run_starts.data_ptr<int>(),
      k_run_lengths.data_ptr<int>(), k_prefix_ends.data_ptr<int>(),
      q_row_positions.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (FlashSCATensorCacheable(cu_seqlens_q) &&
      FlashSCATensorCacheable(cu_seqlens_k)) {
    cache.cu_q = cu_seqlens_q;
    cache.cu_k = cu_seqlens_k;
    cache.k_run_starts = k_run_starts;
    cache.k_run_lengths = k_run_lengths;
    cache.k_prefix_ends = k_prefix_ends;
    cache.q_row_positions = q_row_positions;
    cache.q_version = cu_seqlens_q._version();
    cache.k_version = cu_seqlens_k._version();
    cache.batch = batch;
    cache.q_row_length = q_row_length;
    cache.k_row_length = k_row_length;
    cache.device = cu_seqlens_q.get_device();
    cache.stream = stream;
  }
  return std::make_tuple(
      std::move(k_run_starts), std::move(k_run_lengths),
      std::move(k_prefix_ends), std::move(q_row_positions));
}

std::tuple<torch::Tensor, int64_t, int64_t> FlashSCASM90BosPlan(
    const torch::Tensor& bos) {
  TORCH_CHECK(bos.dim() == 2 && bos.scalar_type() == at::kBool &&
                  bos.device().type() == torch::kCUDA,
              "FlashSCA BOS metadata must be a CUDA bool [B, L] tensor");
  TORCH_CHECK(bos.size(0) > 0 && bos.size(1) > 0,
              "FlashSCA BOS metadata must be nonempty");
  TORCH_CHECK(bos.numel() <= std::numeric_limits<int>::max(),
              "FlashSCA BOS metadata token count must fit int32");

  at::cuda::OptionalCUDAGuard guard(at::device_of(bos));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (FlashSCABosPlanCacheEntry* entry =
          FlashSCAFindBosPlanCacheEntry(bos, stream);
      entry != nullptr && entry->cu_seqlens.defined()) {
    return std::make_tuple(
        entry->cu_seqlens, entry->max_seqlen, entry->num_sequences);
  }

  torch::Tensor bos_contiguous = bos.contiguous();
  auto int_options = bos.options().dtype(at::kInt).memory_format(
      at::MemoryFormat::Contiguous);
  torch::Tensor cu_storage = torch::empty({bos.numel() + 1}, int_options);
  torch::Tensor plan_stats = torch::zeros({2}, int_options);
  cub::CountingInputIterator<int> token_indices(0);
  FlashSCABosStartPredicate is_run_start{
      bos_contiguous.data_ptr<bool>(), static_cast<int>(bos.size(1))};
  cub::TransformInputIterator<
      bool, FlashSCABosStartPredicate, cub::CountingInputIterator<int>>
      run_start_flags(token_indices, is_run_start);
  size_t temp_storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
      nullptr, temp_storage_bytes, token_indices, run_start_flags,
      cu_storage.data_ptr<int>(), plan_stats.data_ptr<int>(),
      static_cast<int>(bos.numel()), stream));
  torch::Tensor temp_storage = torch::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      bos.options().dtype(at::kByte).memory_format(
          at::MemoryFormat::Contiguous));
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
      temp_storage.data_ptr(), temp_storage_bytes, token_indices,
      run_start_flags, cu_storage.data_ptr<int>(), plan_stats.data_ptr<int>(),
      static_cast<int>(bos.numel()), stream));

  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 256;
  const int64_t work_items = std::max<int64_t>(bos.numel(), bos.size(0));
  const int blocks = static_cast<int>(std::min<int64_t>(
      kMaxBlocks, std::max<int64_t>(
                      1, (work_items + kThreads - 1) / kThreads)));
  FlashSCAFinalizeBosToCuSeqlensKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<int>(bos.numel()), cu_storage.data_ptr<int>(),
      plan_stats.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  torch::Tensor plan_stats_cpu = plan_stats.cpu();
  const int* stats = plan_stats_cpu.data_ptr<int>();
  const int num_sequences = stats[0];
  const int max_seqlen = stats[1];
  TORCH_CHECK(num_sequences >= bos.size(0) && max_seqlen > 0,
              "FlashSCA BOS metadata did not produce a valid sequence plan");
  torch::Tensor cu_seqlens =
      cu_storage.narrow(0, 0, int64_t(num_sequences) + 1);

  if (FlashSCABosPlanCacheable(bos)) {
    FlashSCABosPlanCacheEntry& entry =
        FlashSCAGetOrCreateBosPlanCacheEntry(bos, stream);
    entry.cu_seqlens = cu_seqlens;
    entry.max_seqlen = max_seqlen;
    entry.num_sequences = num_sequences;
  }
  return std::make_tuple(
      std::move(cu_seqlens), int64_t(max_seqlen), int64_t(num_sequences));
}

std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t>
FlashSCASM90PairedBosPlan(
    const torch::Tensor& bos, int64_t q_row_length) {
  TORCH_CHECK(bos.dim() == 2 && bos.scalar_type() == at::kBool &&
                  bos.device().type() == torch::kCUDA,
              "FlashSCA BOS metadata must be a CUDA bool [B, L] tensor");
  TORCH_CHECK(bos.size(0) > 0 && bos.size(1) > 0,
              "FlashSCA BOS metadata must be nonempty");
  TORCH_CHECK(q_row_length > 0 && q_row_length <= bos.size(1),
              "FlashSCA paired BOS metadata has an invalid Q row length");
  TORCH_CHECK(bos.numel() <= std::numeric_limits<int>::max() &&
                  bos.size(0) * q_row_length <=
                      std::numeric_limits<int>::max(),
              "FlashSCA BOS metadata token counts must fit int32");

  at::cuda::OptionalCUDAGuard guard(at::device_of(bos));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor bos_contiguous = bos.contiguous();
  auto int_options = bos.options().dtype(at::kInt).memory_format(
      at::MemoryFormat::Contiguous);
  const int q_tokens = static_cast<int>(bos.size(0) * q_row_length);
  const int k_tokens = static_cast<int>(bos.numel());
  const int k_row_length = static_cast<int>(bos.size(1));
  const int q_offset = k_row_length - static_cast<int>(q_row_length);
  torch::Tensor cu_q_storage =
      torch::empty({int64_t(q_tokens) + 1}, int_options);
  torch::Tensor cu_k_storage =
      torch::empty({int64_t(k_tokens) + 1}, int_options);
  torch::Tensor plan_stats = torch::zeros({4}, int_options);

  cub::CountingInputIterator<int> q_indices(0);
  FlashSCABosSuffixStartPredicate q_is_run_start{
      bos_contiguous.data_ptr<bool>(), static_cast<int>(q_row_length),
      k_row_length, q_offset};
  cub::TransformInputIterator<
      bool, FlashSCABosSuffixStartPredicate,
      cub::CountingInputIterator<int>>
      q_run_start_flags(q_indices, q_is_run_start);
  size_t q_temp_storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
      nullptr, q_temp_storage_bytes, q_indices, q_run_start_flags,
      cu_q_storage.data_ptr<int>(), plan_stats.data_ptr<int>(),
      q_tokens, stream));

  cub::CountingInputIterator<int> k_indices(0);
  FlashSCABosStartPredicate k_is_run_start{
      bos_contiguous.data_ptr<bool>(), k_row_length};
  cub::TransformInputIterator<
      bool, FlashSCABosStartPredicate, cub::CountingInputIterator<int>>
      k_run_start_flags(k_indices, k_is_run_start);
  size_t k_temp_storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
      nullptr, k_temp_storage_bytes, k_indices, k_run_start_flags,
      cu_k_storage.data_ptr<int>(), plan_stats.data_ptr<int>() + 2,
      k_tokens, stream));

  const size_t temp_storage_bytes =
      std::max(q_temp_storage_bytes, k_temp_storage_bytes);
  torch::Tensor temp_storage = torch::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      bos.options().dtype(at::kByte).memory_format(
          at::MemoryFormat::Contiguous));
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
      temp_storage.data_ptr(), q_temp_storage_bytes,
      q_indices, q_run_start_flags, cu_q_storage.data_ptr<int>(),
      plan_stats.data_ptr<int>(), q_tokens, stream));
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
      temp_storage.data_ptr(), k_temp_storage_bytes,
      k_indices, k_run_start_flags, cu_k_storage.data_ptr<int>(),
      plan_stats.data_ptr<int>() + 2, k_tokens, stream));

  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 256;
  const int q_blocks = static_cast<int>(std::min<int64_t>(
      kMaxBlocks, std::max<int64_t>(
                      1, (int64_t(q_tokens) + kThreads - 1) / kThreads)));
  const int k_blocks = static_cast<int>(std::min<int64_t>(
      kMaxBlocks, std::max<int64_t>(
                      1, (int64_t(k_tokens) + kThreads - 1) / kThreads)));
  FlashSCAFinalizeBosToCuSeqlensKernel<<<q_blocks, kThreads, 0, stream>>>(
      q_tokens, cu_q_storage.data_ptr<int>(), plan_stats.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  FlashSCAFinalizeBosToCuSeqlensKernel<<<k_blocks, kThreads, 0, stream>>>(
      k_tokens, cu_k_storage.data_ptr<int>(), plan_stats.data_ptr<int>() + 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  torch::Tensor plan_stats_cpu = plan_stats.cpu();
  const int* stats = plan_stats_cpu.data_ptr<int>();
  const int num_q_sequences = stats[0];
  const int max_q_seqlen = stats[1];
  const int num_k_sequences = stats[2];
  const int max_k_seqlen = stats[3];
  TORCH_CHECK(
      num_q_sequences >= bos.size(0) && max_q_seqlen > 0 &&
          num_k_sequences >= bos.size(0) && max_k_seqlen > 0,
      "FlashSCA paired BOS metadata did not produce valid sequence plans");
  torch::Tensor cu_seqlens_q =
      cu_q_storage.narrow(0, 0, int64_t(num_q_sequences) + 1);
  torch::Tensor cu_seqlens_k =
      cu_k_storage.narrow(0, 0, int64_t(num_k_sequences) + 1);
  return std::make_tuple(
      std::move(cu_seqlens_q), std::move(cu_seqlens_k),
      int64_t(max_q_seqlen), int64_t(max_k_seqlen),
      int64_t(num_q_sequences), int64_t(num_k_sequences));
}

std::tuple<torch::Tensor, int64_t> FlashSCASM90BosToCuSeqlens(
    const torch::Tensor& bos) {
  auto [cu_seqlens, max_seqlen, num_sequences] = FlashSCASM90BosPlan(bos);
  static_cast<void>(num_sequences);
  return std::make_tuple(std::move(cu_seqlens), max_seqlen);
}

torch::Tensor FlashSCASM90BosToSegmentIdx(const torch::Tensor& bos) {
  TORCH_CHECK(bos.dim() == 2 && bos.scalar_type() == at::kBool &&
                  bos.device().type() == torch::kCUDA,
              "FlashSCA BOS metadata must be a CUDA bool [B, L] tensor");
  TORCH_CHECK(bos.size(0) > 0 && bos.size(1) > 0,
              "FlashSCA BOS metadata must be nonempty");

  at::cuda::OptionalCUDAGuard guard(at::device_of(bos));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  FlashSCABosPlanCacheEntry* cached_entry =
      FlashSCAFindBosPlanCacheEntry(bos, stream);
  if (cached_entry != nullptr && cached_entry->segment_idx.defined()) {
    return cached_entry->segment_idx;
  }

  torch::Tensor segment_idx = at::cumsum(bos, 1, at::kLong);
  if (FlashSCABosPlanCacheable(bos)) {
    FlashSCABosPlanCacheEntry& entry =
        FlashSCAGetOrCreateBosPlanCacheEntry(bos, stream);
    entry.segment_idx = segment_idx;
  }
  return segment_idx;
}

std::tuple<torch::Tensor, torch::Tensor> FlashSCASM90VarlenFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& position_offsets,
    bool reset_chunk_pos_per_seq,
    const c10::optional<torch::Tensor>& k_run_starts,
    const c10::optional<torch::Tensor>& k_run_lengths,
    bool strict_past,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  const int arch = FlashSCACurrentDeviceArch(q);
  TORCH_CHECK(arch == 90,
              "FlashSCA SM90 FWD requires Hopper sm90, got sm", arch);
  TORCH_CHECK(chunk_size > 0,
              "FlashSCA SM90 FWD expects positive chunk_size");
  FlashSCACheckPackedQKVSM90(q, k, v);
  const FlashSCAVarlenMetadataSM90 metadata =
      FlashSCACheckVarlenMetadataSM90(
          q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k);
  const int num_sequences = metadata.num_sequences;
  TORCH_CHECK(!reset_chunk_pos_per_seq || !position_offsets.has_value(),
              "FlashSCA SM90 reset varlen does not accept position_offsets");
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  const bool cast_compute = q.scalar_type() == at::kFloat;
  const c10::ScalarType compute_dtype =
      q.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kHalf;
  torch::Tensor q_c = FlashSCASM90MaybeCastCompute(
      q, compute_dtype, cast_compute);
  torch::Tensor k_c = FlashSCASM90MaybeCastCompute(
      k, compute_dtype, cast_compute);
  torch::Tensor v_c = FlashSCASM90MaybeCastCompute(
      v, compute_dtype, cast_compute);
  torch::Tensor cu_q_c = cu_seqlens_q.contiguous();
  torch::Tensor cu_k_c = cu_seqlens_k.contiguous();
  TORCH_CHECK(k_run_starts.has_value() == k_run_lengths.has_value(),
              "FlashSCA SM90 k_run_starts/k_run_lengths must be provided together");
  torch::Tensor k_run_starts_c;
  torch::Tensor k_run_lengths_c;
  const int* cu_k_aligned = nullptr;
  const int* seqused_k_data = nullptr;
  if (k_run_starts.has_value()) {
    TORCH_CHECK(k_run_starts.value().device() == q.device() &&
                    k_run_lengths.value().device() == q.device() &&
                    k_run_starts.value().scalar_type() == at::kInt &&
                    k_run_lengths.value().scalar_type() == at::kInt &&
                    k_run_starts.value().dim() == 1 &&
                    k_run_lengths.value().dim() == 1 &&
                    k_run_starts.value().numel() == num_sequences &&
                    k_run_lengths.value().numel() == num_sequences,
                "FlashSCA SM90 row-aligned K metadata must be CUDA int32 "
                "with shape [num_q_runs]");
    k_run_starts_c = k_run_starts.value().contiguous();
    k_run_lengths_c = k_run_lengths.value().contiguous();
    cu_k_aligned = k_run_starts_c.data_ptr<int>();
    seqused_k_data = k_run_lengths_c.data_ptr<int>();
  } else {
    // Global suffix alignment for packed Q/K/V.
    cu_k_aligned =
        cu_k_c.data_ptr<int>() + metadata.k_sequence_offset;
  }
  torch::Tensor position_offsets_c;
  const int* position_offsets_data = nullptr;
  if (!reset_chunk_pos_per_seq) {
    if (position_offsets.has_value()) {
      TORCH_CHECK(position_offsets.value().device().type() == torch::kCUDA &&
                      position_offsets.value().scalar_type() == at::kInt &&
                      position_offsets.value().dim() == 1 &&
                      position_offsets.value().numel() == num_sequences,
                  "FlashSCA SM90 position_offsets must be CUDA int32 "
                  "with shape [num_sequences]");
      position_offsets_c = position_offsets.value().contiguous();
      position_offsets_data = position_offsets_c.data_ptr<int>();
    } else {
      position_offsets_data = cu_q_c.data_ptr<int>();
    }
  }

  const int64_t total_q = q.size(0);
  const int64_t H = q.size(1);
  const int64_t V = v.size(2);
  TORCH_CHECK(!output_fp32 || !output_state.has_value(),
              "FP32-only output and output_state are mutually exclusive");
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    TORCH_CHECK(
        state.device() == q.device() && state.scalar_type() == at::kFloat &&
            state.is_contiguous() && state.dim() == 3 &&
            state.size(0) == total_q && state.size(1) == H &&
            state.size(2) == V,
        "FlashSCA SM90 varlen output_state must be contiguous float32 "
        "[total_q, Hq, V] on the Q device");
  }
  torch::Tensor y_compute = torch::empty(
      {total_q, H, V},
      v_c.options().dtype(output_fp32 ? at::kFloat : v_c.scalar_type())
          .memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor lse = torch::empty(
      {H, total_q},
      q.options().dtype(at::kFloat).memory_format(
          at::MemoryFormat::Contiguous));

  if (q_c.scalar_type() == at::kHalf) {
    FlashSCASM90FwdImpl<cutlass::half_t, cutlass::half_t>(
        q_c, k_c, v_c, chunk_size, static_cast<float>(scale),
        cu_q_c.data_ptr<int>(), cu_k_aligned, seqused_k_data, num_sequences,
        max_seqlen_q, max_seqlen_k, position_offsets_data, strict_past,
        y_compute, lse, output_state);
  } else {
    FlashSCASM90FwdImpl<
        cutlass::bfloat16_t, cutlass::bfloat16_t>(
        q_c, k_c, v_c, chunk_size, static_cast<float>(scale),
        cu_q_c.data_ptr<int>(), cu_k_aligned, seqused_k_data, num_sequences,
        max_seqlen_q, max_seqlen_k, position_offsets_data, strict_past,
        y_compute, lse, output_state);
  }

  torch::Tensor y =
      cast_compute && !output_fp32 ? y_compute.to(q.scalar_type()) : y_compute;
  return std::make_tuple<torch::Tensor, torch::Tensor>(std::move(y),
                                                       std::move(lse));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FlashSCASM90VarlenBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t chunk_size, double scale, bool deterministic,
    const c10::optional<torch::Tensor>& position_offsets,
    bool reset_chunk_pos_per_seq,
    const c10::optional<torch::Tensor>& k_run_starts,
    const c10::optional<torch::Tensor>& k_run_lengths,
    const c10::optional<torch::Tensor>& k_prefix_ends,
    int64_t k_row_length,
    bool strict_past) {
  const int arch = FlashSCACurrentDeviceArch(q);
  TORCH_CHECK(arch == 90,
              "FlashSCA SM90 BWD requires Hopper sm90, got sm", arch);
  TORCH_CHECK(chunk_size > 0,
              "FlashSCA SM90 BWD expects positive chunk_size");
  FlashSCACheckBwdTensorsSM90(y_grad, q, k, v, y, lse);
  const FlashSCAVarlenMetadataSM90 metadata =
      FlashSCACheckVarlenMetadataSM90(
          q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k);
  const int num_sequences = metadata.num_sequences;
  TORCH_CHECK(!reset_chunk_pos_per_seq || !position_offsets.has_value(),
              "FlashSCA SM90 reset varlen does not accept position_offsets");
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  const bool cast_compute = q.scalar_type() == at::kFloat;
  const c10::ScalarType compute_dtype =
      q.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kHalf;
  torch::Tensor q_c = FlashSCASM90MaybeCastCompute(
      q, compute_dtype, cast_compute);
  torch::Tensor k_c = FlashSCASM90MaybeCastCompute(
      k, compute_dtype, cast_compute);
  torch::Tensor v_c = FlashSCASM90MaybeCastCompute(
      v, compute_dtype, cast_compute);
  torch::Tensor dy_c = FlashSCASM90MaybeCastCompute(
      y_grad, compute_dtype, cast_compute);
  torch::Tensor y_c = FlashSCASM90MaybeCastCompute(
      y, compute_dtype, cast_compute);
  torch::Tensor lse_c = lse.contiguous();
  torch::Tensor cu_q_c = cu_seqlens_q.contiguous();
  torch::Tensor cu_k_c = cu_seqlens_k.contiguous();
  TORCH_CHECK(k_run_starts.has_value() == k_run_lengths.has_value(),
              "FlashSCA SM90 k_run_starts/k_run_lengths must be provided together");
  TORCH_CHECK(k_run_starts.has_value() == k_prefix_ends.has_value(),
              "FlashSCA SM90 row-aligned BWD requires K starts, lengths, and prefixes together");
  torch::Tensor k_run_starts_c;
  torch::Tensor k_run_lengths_c;
  torch::Tensor k_prefix_ends_c;
  const int* cu_k_aligned = nullptr;
  const int* seqused_k_data = nullptr;
  if (k_run_starts.has_value()) {
    TORCH_CHECK(k_run_starts.value().device() == q.device() &&
                    k_run_lengths.value().device() == q.device() &&
                    k_prefix_ends.value().device() == q.device() &&
                    k_run_starts.value().scalar_type() == at::kInt &&
                    k_run_lengths.value().scalar_type() == at::kInt &&
                    k_prefix_ends.value().scalar_type() == at::kInt &&
                    k_run_starts.value().dim() == 1 &&
                    k_run_lengths.value().dim() == 1 &&
                    k_prefix_ends.value().dim() == 1 &&
                    k_run_starts.value().numel() == num_sequences &&
                    k_run_lengths.value().numel() == num_sequences &&
                    k_prefix_ends.value().numel() > 1 &&
                    k_row_length > 0 &&
                    k_prefix_ends.value().numel() * k_row_length == k.size(0),
                "FlashSCA SM90 row-aligned BWD metadata has an invalid "
                "dtype, shape, device, or K row length");
    k_run_starts_c = k_run_starts.value().contiguous();
    k_run_lengths_c = k_run_lengths.value().contiguous();
    k_prefix_ends_c = k_prefix_ends.value().contiguous();
    cu_k_aligned = k_run_starts_c.data_ptr<int>();
    seqused_k_data = k_run_lengths_c.data_ptr<int>();
  } else {
    cu_k_aligned =
        cu_k_c.data_ptr<int>() + metadata.k_sequence_offset;
  }
  torch::Tensor position_offsets_c;
  const int* position_offsets_data = nullptr;
  if (!reset_chunk_pos_per_seq) {
    if (position_offsets.has_value()) {
      TORCH_CHECK(position_offsets.value().device().type() == torch::kCUDA &&
                      position_offsets.value().scalar_type() == at::kInt &&
                      position_offsets.value().dim() == 1 &&
                      position_offsets.value().numel() == num_sequences,
                  "FlashSCA SM90 position_offsets must be CUDA int32 "
                  "with shape [num_sequences]");
      position_offsets_c = position_offsets.value().contiguous();
      position_offsets_data = position_offsets_c.data_ptr<int>();
    } else {
      position_offsets_data = cu_q_c.data_ptr<int>();
    }
  }

  const int64_t D = q.size(2);
  const int64_t V = v.size(2);
  int64_t bucket_dim_qk = FlashSCABwdHeadDimBucketSm90(
      static_cast<int>(D));
  int64_t bucket_dim_v = FlashSCABwdHeadDimBucketSm90(
      static_cast<int>(V));
  if (bucket_dim_qk == 192 && bucket_dim_v == 160) {
    bucket_dim_v = 192;
  }
  if ((bucket_dim_qk == 160 || bucket_dim_qk == 192) &&
      bucket_dim_v == 256) {
    bucket_dim_qk = 256;
  }
  // D256/V32 BWD dispatch: V64 layout.
  if (bucket_dim_qk == 256 && bucket_dim_v == 32) {
    bucket_dim_v = 64;
  }
  // Deterministic D=256 varlen V bucket: 128 or 256.
  if (deterministic && bucket_dim_qk == 256) {
    bucket_dim_v = bucket_dim_v <= 128 ? 128 : 256;
  }
  TORCH_CHECK(bucket_dim_qk <= 256 && bucket_dim_v <= 256,
              "FlashSCA SM90 BWD supports max(D,V) <= 256; got D=",
              D, ", V=", V);
  const bool needs_qk_dim_pad = D != bucket_dim_qk;
  const bool needs_v_dim_pad = V != bucket_dim_v;
  if (needs_qk_dim_pad) {
    q_c = FlashSCASM90PadLastDim(q_c, bucket_dim_qk);
    k_c = FlashSCASM90PadLastDim(k_c, bucket_dim_qk);
  }
  if (needs_v_dim_pad) {
    v_c = FlashSCASM90PadLastDim(v_c, bucket_dim_v);
    dy_c = FlashSCASM90PadLastDim(dy_c, bucket_dim_v);
    y_c = FlashSCASM90PadLastDim(y_c, bucket_dim_v);
  }

  torch::Tensor dq_pad = torch::empty_like(q_c);
  // Unmatched leading K/V gradients: zero.
  torch::Tensor dk_total_pad = torch::empty_like(k_c);
  torch::Tensor dv_total_pad = torch::empty_like(v_c);
  // GQA/MQA FP32 workspaces include all K/V prefixes after the final cast.
  // MHA direct-write path explicitly zeros unmatched prefixes.
  if (q_c.size(1) == k_c.size(1)) {
    if (k_run_starts.has_value()) {
      FlashSCAZeroRowAlignedKVPrefixes(
          dk_total_pad, dv_total_pad, k_prefix_ends_c, k_row_length);
    } else if (metadata.k_sequence_offset > 0) {
      FlashSCAZeroRightAlignedKVPrefix(
          dk_total_pad, dv_total_pad, cu_k_aligned);
    }
  }
  // Deterministic dQ layout: one BWD-row guard region per sequence.
  const int64_t bwd_block_m =
      bucket_dim_qk <= 64 && bucket_dim_v <= 128 ? 128 : 64;
  const int64_t seqlen_q_padded = FlashSCARoundUp(
      q.size(0) + int64_t(num_sequences) * bwd_block_m, bwd_block_m);

  if (q_c.scalar_type() == at::kHalf) {
    FlashSCASM90BwdImpl<cutlass::half_t>(
        dy_c, q_c, k_c, v_c, y_c, lse_c, chunk_size,
        static_cast<float>(scale), cu_q_c.data_ptr<int>(),
        cu_k_aligned, seqused_k_data, num_sequences, max_seqlen_q, max_seqlen_k,
        seqlen_q_padded, position_offsets_data, strict_past, dq_pad, dk_total_pad,
        dv_total_pad, deterministic);
  } else {
    FlashSCASM90BwdImpl<cutlass::bfloat16_t>(
        dy_c, q_c, k_c, v_c, y_c, lse_c, chunk_size,
        static_cast<float>(scale), cu_q_c.data_ptr<int>(),
        cu_k_aligned, seqused_k_data, num_sequences, max_seqlen_q, max_seqlen_k,
        seqlen_q_padded, position_offsets_data, strict_past, dq_pad, dk_total_pad,
        dv_total_pad, deterministic);
  }

  torch::Tensor dq_c =
      needs_qk_dim_pad ? dq_pad.narrow(2, 0, D).contiguous() : dq_pad;
  torch::Tensor dk_c = needs_qk_dim_pad
      ? dk_total_pad.narrow(2, 0, D).contiguous()
      : dk_total_pad;
  torch::Tensor dv_c = needs_v_dim_pad
      ? dv_total_pad.narrow(2, 0, V).contiguous()
      : dv_total_pad;

  torch::Tensor dq = cast_compute ? dq_c.to(q.scalar_type()) : dq_c;
  torch::Tensor dk = cast_compute ? dk_c.to(k.scalar_type()) : dk_c;
  torch::Tensor dv = cast_compute ? dv_c.to(v.scalar_type()) : dv_c;
  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
      std::move(dq), std::move(dk), std::move(dv));
}

bool FlashSCASM90Available(const torch::Tensor& q) {
  TORCH_CHECK(q.device().type() == torch::kCUDA,
              "FlashSCA SM90 dispatch expects q to be CUDA");
  const int arch = FlashSCACurrentDeviceArch(q);
  return arch == 90;
}

std::tuple<torch::Tensor, torch::Tensor> FlashSCASM90Fwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool reset_chunk_pos_per_seq, bool strict_past,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  const int arch = FlashSCACurrentDeviceArch(q);
  TORCH_CHECK(arch == 90,
              "FlashSCA SM90 FWD requires Hopper sm90, got sm", arch);
  TORCH_CHECK(chunk_size > 0,
              "FlashSCA SM90 FWD expects positive chunk_size");
  TORCH_CHECK(prev_k.has_value() == prev_v.has_value(),
              "prev_k and prev_v must both be provided or both be None");
  FlashSCACheckDenseQKVSM90(q, k, v);
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  const bool cast_compute = q.scalar_type() == at::kFloat;
  const c10::ScalarType compute_dtype =
      q.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kHalf;
  torch::Tensor q_c = !cast_compute &&
          attention::hopper::CanUseStridedTmaQuery(q)
      ? q : FlashSCASM90MaybeCastCompute(q, compute_dtype, cast_compute);
  torch::Tensor k_c = FlashSCASM90MaybeCastCompute(
      k, compute_dtype, cast_compute);
  torch::Tensor v_c = FlashSCASM90MaybeCastCompute(
      v, compute_dtype, cast_compute);

  const bool has_prev = prev_k.has_value();
  c10::optional<torch::Tensor> prev_k_c = c10::nullopt;
  c10::optional<torch::Tensor> prev_v_c = c10::nullopt;
  if (has_prev) {
    FlashSCACheckPrevChunkSM90(
        q, k, v, prev_k.value(), prev_v.value(), chunk_size);
    prev_k_c = FlashSCASM90MaybeCastCompute(
        prev_k.value(), compute_dtype, cast_compute);
    prev_v_c = FlashSCASM90MaybeCastCompute(
        prev_v.value(), compute_dtype, cast_compute);
    k_c = torch::cat({prev_k_c.value(), k_c}, 1);
    v_c = torch::cat({prev_v_c.value(), v_c}, 1);
  }

  attention::partition::SegmentMetadata segment_metadata =
      attention::partition::NormalizeSegmentMetadata(
          q_segment_idx, k_segment_idx, q.size(0), q.size(1),
          k.size(1) + (has_prev ? chunk_size : int64_t(0)),
          q.device(), "FlashSCA");
  const bool has_segment = segment_metadata.enabled();
  torch::Tensor q_segment_idx_c = segment_metadata.q_segment_idx;
  torch::Tensor k_segment_idx_c = segment_metadata.k_segment_idx;
  const int64_t* q_segment_idx_data = nullptr;
  const int64_t* k_segment_idx_data = nullptr;
  int64_t k_segment_len = 0;
  if (has_segment) {
    q_segment_idx_data = q_segment_idx_c.data_ptr<int64_t>();
    k_segment_idx_data = k_segment_idx_c.data_ptr<int64_t>();
    k_segment_len = k_segment_idx_c.size(1);
  }

  const int64_t B = q.size(0);
  const int64_t L = q.size(1);
  const int64_t H = q.size(2);
  const int64_t V = v.size(3);
  TORCH_CHECK(!output_fp32 || !output_state.has_value(),
              "FP32-only output and output_state are mutually exclusive");
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    TORCH_CHECK(
        state.device() == q.device() && state.scalar_type() == at::kFloat &&
            state.is_contiguous() && state.dim() == 4 &&
            state.size(0) == B && state.size(1) == L &&
            state.size(2) == H && state.size(3) == V,
        "FlashSCA SM90 output_state must be contiguous float32 "
        "[B, L, Hq, V] on the Q device");
  }

  torch::Tensor y_compute = torch::empty(
      {B, L, H, V},
      v_c.options().dtype(output_fp32 ? at::kFloat : v_c.scalar_type())
          .memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor lse = torch::empty(
      {B, H, L},
      q.options().dtype(at::kFloat).memory_format(
          at::MemoryFormat::Contiguous));

  torch::Tensor q_chunk_positions;
  const bool dense_reset_with_segments =
      reset_chunk_pos_per_seq && has_segment;
  const int* q_chunk_positions_data = nullptr;
  if (dense_reset_with_segments) {
    q_chunk_positions = FlashSCAGetDenseResetChunkPositions(
        q_segment_idx_c, k_segment_idx_c);
    q_chunk_positions_data = q_chunk_positions.data_ptr<int>();
  }

  if (q_c.scalar_type() == at::kHalf) {
    FlashSCASM90DenseFwdImpl<cutlass::half_t, cutlass::half_t>(
        q_c, k_c, v_c, chunk_size, static_cast<float>(scale),
        q_segment_idx_data, k_segment_idx_data, k_segment_len,
        q_chunk_positions_data, dense_reset_with_segments, strict_past,
        y_compute, lse, output_state);
  } else {
    FlashSCASM90DenseFwdImpl<
        cutlass::bfloat16_t, cutlass::bfloat16_t>(
        q_c, k_c, v_c, chunk_size, static_cast<float>(scale),
        q_segment_idx_data, k_segment_idx_data, k_segment_len,
        q_chunk_positions_data, dense_reset_with_segments, strict_past,
        y_compute, lse, output_state);
  }

  torch::Tensor y =
      cast_compute && !output_fp32 ? y_compute.to(q.scalar_type()) : y_compute;
  return std::make_tuple<torch::Tensor, torch::Tensor>(std::move(y),
                                                       std::move(lse));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCASM90BwdWithHeadBoundary(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, bool reset_chunk_pos_per_seq,
    bool strict_past, int odd_head_window_right_delta) {
  const int arch = FlashSCACurrentDeviceArch(q);
  TORCH_CHECK(arch == 90,
              "FlashSCA SM90 BWD requires Hopper sm90, got sm", arch);
  TORCH_CHECK(chunk_size > 0,
              "FlashSCA SM90 BWD expects positive chunk_size");
  TORCH_CHECK(
      odd_head_window_right_delta == 0 ||
          odd_head_window_right_delta == -1,
      "FlashSCA SM90 BWD odd-head boundary delta must be 0 or -1");
  TORCH_CHECK(
      odd_head_window_right_delta == 0 ||
          (!strict_past && q.size(2) % 2 == 0),
      "FlashSCA SM90 BWD odd-head boundary mode requires inclusive "
      "base semantics and interleaved head pairs");
  TORCH_CHECK(prev_k.has_value() == prev_v.has_value(),
              "prev_k and prev_v must both be provided or both be None");
  FlashSCACheckDenseBwdTensorsSM90(y_grad, q, k, v, y, lse);
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));

  const bool cast_compute = q.scalar_type() == at::kFloat;
  const c10::ScalarType compute_dtype =
      q.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kHalf;
  torch::Tensor q_c = FlashSCASM90MaybeCastCompute(
      q, compute_dtype, cast_compute);
  torch::Tensor k_c = FlashSCASM90MaybeCastCompute(
      k, compute_dtype, cast_compute);
  torch::Tensor v_c = FlashSCASM90MaybeCastCompute(
      v, compute_dtype, cast_compute);
  torch::Tensor dy_c = FlashSCASM90MaybeCastCompute(
      y_grad, compute_dtype, cast_compute);
  torch::Tensor y_c = FlashSCASM90MaybeCastCompute(
      y, compute_dtype, cast_compute);

  const bool has_prev = prev_k.has_value();
  c10::optional<torch::Tensor> prev_k_c = c10::nullopt;
  c10::optional<torch::Tensor> prev_v_c = c10::nullopt;
  if (has_prev) {
    FlashSCACheckPrevChunkSM90(
        q, k, v, prev_k.value(), prev_v.value(), chunk_size);
    prev_k_c = FlashSCASM90MaybeCastCompute(
        prev_k.value(), compute_dtype, cast_compute);
    prev_v_c = FlashSCASM90MaybeCastCompute(
        prev_v.value(), compute_dtype, cast_compute);
    k_c = torch::cat({prev_k_c.value(), k_c}, 1);
    v_c = torch::cat({prev_v_c.value(), v_c}, 1);
  }

  attention::partition::SegmentMetadata segment_metadata =
      attention::partition::NormalizeSegmentMetadata(
          q_segment_idx, k_segment_idx, q.size(0), q.size(1),
          k.size(1) + (has_prev ? chunk_size : int64_t(0)),
          q.device(), "FlashSCA");
  const bool has_segment = segment_metadata.enabled();
  torch::Tensor q_segment_idx_c = segment_metadata.q_segment_idx;
  torch::Tensor k_segment_idx_c = segment_metadata.k_segment_idx;
  const int64_t* q_segment_idx_data = nullptr;
  const int64_t* k_segment_idx_data = nullptr;
  int64_t k_segment_len = 0;
  if (has_segment) {
    q_segment_idx_data = q_segment_idx_c.data_ptr<int64_t>();
    k_segment_idx_data = k_segment_idx_c.data_ptr<int64_t>();
    k_segment_len = k_segment_idx_c.size(1);
  }

  const int64_t B = q.size(0);
  const int64_t L = q.size(1);
  const int64_t H = q.size(2);
  const int64_t D = q.size(3);
  const int64_t V = v.size(3);

  const bool native_small_chunk_det =
      deterministic && !has_segment && !has_prev &&
      q.size(1) == k.size(1) &&
      odd_head_window_right_delta == 0 &&
      UseNativeSmallChunkDetBwd(D, V, chunk_size);
  const int64_t bucket_dim_qk = FlashSCABwdHeadDimBucketSm90(
      static_cast<int>(D));
  int64_t bucket_dim_v = FlashSCABwdHeadDimBucketSm90(
      static_cast<int>(V));
  if (!native_small_chunk_det && bucket_dim_qk == 192 && bucket_dim_v == 160) {
    bucket_dim_v = 192;
  }
  TORCH_CHECK(bucket_dim_qk <= 256 && bucket_dim_v <= 256,
              "FlashSCA SM90 BWD supports max(D,V) <= 256; got D=",
              D, ", V=", V);
  const bool needs_qk_dim_pad = D != bucket_dim_qk;
  const bool needs_v_dim_pad = V != bucket_dim_v;
  if (needs_qk_dim_pad) {
    q_c = FlashSCASM90PadLastDim(q_c, bucket_dim_qk);
    k_c = FlashSCASM90PadLastDim(k_c, bucket_dim_qk);
  }
  if (needs_v_dim_pad) {
    v_c = FlashSCASM90PadLastDim(v_c, bucket_dim_v);
    dy_c = FlashSCASM90PadLastDim(dy_c, bucket_dim_v);
    y_c = FlashSCASM90PadLastDim(y_c, bucket_dim_v);
  }

  torch::Tensor dq_pad = torch::empty_like(q_c);
  const bool grouped_heads = q.size(2) != k.size(2);
  const bool dense = !has_segment && !has_prev && q.size(1) == k.size(1);
  const bool exact_d192_v128_gqa =
      dense && !deterministic && bucket_dim_qk == 192 &&
      bucket_dim_v == 128 && q.size(2) == 2 * k.size(2) &&
      chunk_size <= kDenseSmallChunk;
  const bool use_d192 =
      dense && !native_small_chunk_det && !exact_d192_v128_gqa &&
      ((bucket_dim_qk == 192 && bucket_dim_v == 128) ||
                (bucket_dim_qk == 160 &&
                 (bucket_dim_v == 128 || bucket_dim_v == 160)) ||
                (deterministic && grouped_heads &&
                 ((bucket_dim_qk == 160 && bucket_dim_v == 192) ||
                  (bucket_dim_qk == 192 && bucket_dim_v == 160))));
  const bool use_d96 =
      dense && !deterministic && D == 64 && V == 96;
  const bool pad_grouped_k_grad =
      grouped_heads &&
      ((use_d192 && bucket_dim_qk != 192) || use_d96);
  const bool pad_grouped_v_grad =
      grouped_heads && use_d192 && bucket_dim_v != 192;
  torch::Tensor dk_total_pad = pad_grouped_k_grad
      ? FlashSCASM90EmptyLastDimLike(k_c, use_d192 ? 192 : 96)
      : torch::empty_like(k_c);
  torch::Tensor dv_total_pad = pad_grouped_v_grad
      ? FlashSCASM90EmptyLastDimLike(v_c, 192)
      : torch::empty_like(v_c);

  torch::Tensor q_chunk_positions;
  const bool dense_reset_with_segments =
      reset_chunk_pos_per_seq && has_segment;
  const int* q_chunk_positions_data = nullptr;
  if (dense_reset_with_segments) {
    q_chunk_positions = FlashSCAGetDenseResetChunkPositions(
        q_segment_idx_c, k_segment_idx_c);
    q_chunk_positions_data = q_chunk_positions.data_ptr<int>();
  }

  if (q_c.scalar_type() == at::kHalf) {
    FlashSCASM90DenseBwdImpl<cutlass::half_t>(
        dy_c, q_c, k_c, v_c, y_c, lse, chunk_size,
        static_cast<float>(scale), q_segment_idx_data, k_segment_idx_data,
        k_segment_len, q_chunk_positions_data, dense_reset_with_segments,
        dq_pad, dk_total_pad, dv_total_pad,
        deterministic, strict_past, odd_head_window_right_delta,
        native_small_chunk_det);
  } else {
    FlashSCASM90DenseBwdImpl<cutlass::bfloat16_t>(
        dy_c, q_c, k_c, v_c, y_c, lse, chunk_size,
        static_cast<float>(scale), q_segment_idx_data, k_segment_idx_data,
        k_segment_len, q_chunk_positions_data, dense_reset_with_segments,
        dq_pad, dk_total_pad, dv_total_pad,
        deterministic, strict_past, odd_head_window_right_delta,
        native_small_chunk_det);
  }

  torch::Tensor dq_c =
      needs_qk_dim_pad ? dq_pad.narrow(3, 0, D).contiguous() : dq_pad;
  torch::Tensor dk_c;
  torch::Tensor dv_c;
  c10::optional<torch::Tensor> prev_dk_c = c10::nullopt;
  c10::optional<torch::Tensor> prev_dv_c = c10::nullopt;
  if (has_prev) {
    torch::Tensor prev_dk_view = dk_total_pad.narrow(1, 0, chunk_size);
    torch::Tensor prev_dv_view = dv_total_pad.narrow(1, 0, chunk_size);
    torch::Tensor dk_view = dk_total_pad.narrow(1, chunk_size, k.size(1));
    torch::Tensor dv_view = dv_total_pad.narrow(1, chunk_size, k.size(1));
    if (needs_qk_dim_pad || pad_grouped_k_grad) {
      prev_dk_view = prev_dk_view.narrow(3, 0, D);
      dk_view = dk_view.narrow(3, 0, D);
    }
    if (needs_v_dim_pad || pad_grouped_v_grad) {
      prev_dv_view = prev_dv_view.narrow(3, 0, V);
      dv_view = dv_view.narrow(3, 0, V);
    }
    prev_dk_c = c10::make_optional(prev_dk_view.contiguous());
    prev_dv_c = c10::make_optional(prev_dv_view.contiguous());
    dk_c = dk_view.contiguous();
    dv_c = dv_view.contiguous();
  } else {
    dk_c = needs_qk_dim_pad || pad_grouped_k_grad
        ? dk_total_pad.narrow(3, 0, D).contiguous()
        : dk_total_pad;
    dv_c = needs_v_dim_pad || pad_grouped_v_grad
        ? dv_total_pad.narrow(3, 0, V).contiguous()
        : dv_total_pad;
  }

  torch::Tensor dq = cast_compute ? dq_c.to(q.scalar_type()) : dq_c;
  torch::Tensor dk = cast_compute ? dk_c.to(k.scalar_type()) : dk_c;
  torch::Tensor dv = cast_compute ? dv_c.to(v.scalar_type()) : dv_c;
  c10::optional<torch::Tensor> prev_dk_out = c10::nullopt;
  c10::optional<torch::Tensor> prev_dv_out = c10::nullopt;
  if (has_prev) {
    prev_dk_out =
        cast_compute
            ? c10::make_optional(prev_dk_c.value().to(prev_k.value().scalar_type()))
            : prev_dk_c;
    prev_dv_out =
        cast_compute
            ? c10::make_optional(prev_dv_c.value().to(prev_v.value().scalar_type()))
            : prev_dv_c;
  }
  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         c10::optional<torch::Tensor>,
                         c10::optional<torch::Tensor>>(
      std::move(dq), std::move(dk), std::move(dv), std::move(prev_dk_out),
      std::move(prev_dv_out));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCASM90Bwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, bool reset_chunk_pos_per_seq,
    bool strict_past) {
  return FlashSCASM90BwdWithHeadBoundary(
      y_grad, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic,
      reset_chunk_pos_per_seq, strict_past,
      0 /*odd_head_window_right_delta*/);
}

}  // namespace ops
}  // namespace xattn

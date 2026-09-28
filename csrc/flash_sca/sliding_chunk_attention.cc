// Author: Shicheng Wen

#include "bindings.h"
#include "flash_sca/sliding_chunk_attention.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <string>

namespace xattn {
namespace ops {

#ifdef XATTN_HAS_SM90
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
FlashSCASM90BuildRowAlignedMetadata(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t batch, int64_t q_row_length, int64_t k_row_length);
std::tuple<torch::Tensor, int64_t> FlashSCASM90BosToCuSeqlens(
    const torch::Tensor& bos);
std::tuple<torch::Tensor, int64_t, int64_t> FlashSCASM90BosPlan(
    const torch::Tensor& bos);
std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t>
FlashSCASM90PairedBosPlan(
    const torch::Tensor& bos, int64_t q_row_length);
torch::Tensor FlashSCASM90BosToSegmentIdx(const torch::Tensor& bos);
bool FlashSCASM90SegmentMetadataIsDense(
    const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx);
bool FlashSCASM90VarlenMetadataIsDense(
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t batch,
    int64_t q_row_length,
    int64_t k_row_length,
    const c10::optional<torch::Tensor>& position_offsets);

std::tuple<torch::Tensor, torch::Tensor>
FlashSCASM90VarlenFwd(
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
    const c10::optional<torch::Tensor>& output_state = c10::nullopt,
    bool output_fp32 = false);

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
    bool strict_past);
#endif

namespace {

void CheckOptionalCUDA(const c10::optional<torch::Tensor>& tensor,
                       const char* name) {
  if (tensor.has_value()) {
    TORCH_CHECK(tensor.value().device().type() == torch::kCUDA,
                name, " must be a CUDA tensor");
  }
}

std::string NormalizeFlashSCABackend(std::string backend) {
  std::transform(
      backend.begin(), backend.end(), backend.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return backend;
}

void CheckFlashSCASM90OnlyBackend(const std::string& backend) {
  const std::string normalized_backend = NormalizeFlashSCABackend(backend);
  TORCH_CHECK(normalized_backend.empty() || normalized_backend == "auto" ||
                  normalized_backend == "sm90",
              "This xattn build includes only the FlashSCA SM90 backend; "
              "build with XATTN_TARGET_SM=all or use backend='auto'/'sm90'");
}

void CheckFlashSCASM90OnlyAvailable(const torch::Tensor& q) {
#ifdef XATTN_HAS_SM90
  TORCH_CHECK(FlashSCASM90Available(q),
              "This xattn build requires the FlashSCA SM90 backend, "
              "but SM90 is unavailable or disabled");
#else
  TORCH_CHECK(false,
              "This xattn build was not compiled with the FlashSCA SM90 "
              "backend; set XATTN_TARGET_SM=90 or XATTN_TARGET_SM=all");
#endif
}

#ifdef XATTN_HAS_SM90
using FlashSCAWeakTensorImpl = c10::weak_intrusive_ptr<
    at::TensorImpl, at::UndefinedTensorImpl>;

struct FlashSCATensorLayout {
  std::array<int64_t, 4> sizes{};
  c10::ScalarType dtype = c10::ScalarType::Undefined;
  int device = -1;
};

struct FlashSCAInferenceBosFwdPlanCacheEntry {
  c10::optional<FlashSCAWeakTensorImpl> bos;
  FlashSCATensorLayout q;
  FlashSCATensorLayout k;
  FlashSCATensorLayout v;
  c10::optional<FlashSCATensorLayout> prev_k;
  c10::optional<FlashSCATensorLayout> prev_v;
  torch::Tensor q_segment_idx;
  torch::Tensor k_segment_idx;
  torch::Tensor cu_seqlens_q;
  torch::Tensor cu_seqlens_k;
  torch::Tensor position_offsets;
  torch::Tensor k_run_starts;
  torch::Tensor k_run_lengths;
  int64_t bos_size_0 = -1;
  int64_t bos_size_1 = -1;
  int64_t bos_stride_0 = -1;
  int64_t bos_stride_1 = -1;
  int64_t chunk_size = -1;
  int64_t max_seqlen_q = -1;
  int64_t max_seqlen_k = -1;
  int64_t route = -1;
  int device = -1;
  bool reset_chunk_pos_per_seq = false;
  cudaStream_t stream = nullptr;
  bool initialized = false;
};

constexpr int kFlashSCAInferenceBosFwdPlanCacheCapacity = 8;
thread_local std::array<
    FlashSCAInferenceBosFwdPlanCacheEntry,
    kFlashSCAInferenceBosFwdPlanCacheCapacity>
    g_flash_sca_inference_bos_fwd_plan_cache;
thread_local int g_flash_sca_inference_bos_fwd_plan_cache_next = 0;
thread_local int g_flash_sca_inference_bos_fwd_plan_cache_last = -1;

FlashSCATensorLayout FlashSCACaptureTensorLayout(
    const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 4, "FlashSCA Q/K/V tensors must be 4D");
  return {
      {tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)},
      tensor.scalar_type(),
      tensor.get_device(),
  };
}

bool FlashSCATensorLayoutMatches(
    const FlashSCATensorLayout& layout,
    const torch::Tensor& tensor) {
  return tensor.dim() == 4 && tensor.size(0) == layout.sizes[0] &&
      tensor.size(1) == layout.sizes[1] &&
      tensor.size(2) == layout.sizes[2] &&
      tensor.size(3) == layout.sizes[3] &&
      tensor.scalar_type() == layout.dtype &&
      tensor.get_device() == layout.device;
}

bool FlashSCAOptionalTensorLayoutMatches(
    const c10::optional<FlashSCATensorLayout>& layout,
    const c10::optional<torch::Tensor>& tensor) {
  if (layout.has_value() != tensor.has_value()) {
    return false;
  }
  return !layout.has_value() ||
      FlashSCATensorLayoutMatches(layout.value(), tensor.value());
}

bool FlashSCAInferenceBosFwdPlanCacheMatches(
    const FlashSCAInferenceBosFwdPlanCacheEntry& entry,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const torch::Tensor& bos,
    bool reset_chunk_pos_per_seq,
    cudaStream_t stream) {
  if (!entry.initialized || !entry.bos.has_value()) {
    return false;
  }
  auto bos_impl = entry.bos.value().lock();
  return bos_impl &&
      bos_impl.get() == bos.unsafeGetTensorImpl() &&
      bos.dim() == 2 &&
      bos.size(0) == entry.bos_size_0 &&
      bos.size(1) == entry.bos_size_1 &&
      bos.stride(0) == entry.bos_stride_0 &&
      bos.stride(1) == entry.bos_stride_1 &&
      FlashSCATensorLayoutMatches(entry.q, q) &&
      FlashSCATensorLayoutMatches(entry.k, k) &&
      FlashSCATensorLayoutMatches(entry.v, v) &&
      FlashSCAOptionalTensorLayoutMatches(entry.prev_k, prev_k) &&
      FlashSCAOptionalTensorLayoutMatches(entry.prev_v, prev_v) &&
      entry.chunk_size == chunk_size &&
      entry.device == bos.get_device() &&
      entry.reset_chunk_pos_per_seq == reset_chunk_pos_per_seq &&
      entry.stream == stream;
}

int FlashSCAFindInferenceBosFwdPlanCacheEntry(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const torch::Tensor& bos,
    bool reset_chunk_pos_per_seq,
    cudaStream_t stream) {
  if (g_flash_sca_inference_bos_fwd_plan_cache_last >= 0) {
    const int index = g_flash_sca_inference_bos_fwd_plan_cache_last;
    if (FlashSCAInferenceBosFwdPlanCacheMatches(
            g_flash_sca_inference_bos_fwd_plan_cache[index],
            q, k, v, chunk_size, prev_k, prev_v, bos,
            reset_chunk_pos_per_seq, stream)) {
      return index;
    }
  }
  for (int index = 0;
       index < kFlashSCAInferenceBosFwdPlanCacheCapacity;
       ++index) {
    if (index == g_flash_sca_inference_bos_fwd_plan_cache_last) {
      continue;
    }
    if (FlashSCAInferenceBosFwdPlanCacheMatches(
            g_flash_sca_inference_bos_fwd_plan_cache[index],
            q, k, v, chunk_size, prev_k, prev_v, bos,
            reset_chunk_pos_per_seq, stream)) {
      return index;
    }
  }
  return -1;
}

std::tuple<torch::Tensor, torch::Tensor>
FlashSCAExecuteInferenceBosFwdPlan(
    const FlashSCAInferenceBosFwdPlanCacheEntry& entry,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size,
    double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const std::string& backend,
    bool reset_chunk_pos_per_seq) {
  if (entry.route != 2) {
    const c10::optional<torch::Tensor> q_segment_idx =
        entry.q_segment_idx.defined()
        ? c10::optional<torch::Tensor>(entry.q_segment_idx)
        : c10::nullopt;
    const c10::optional<torch::Tensor> k_segment_idx =
        entry.k_segment_idx.defined()
        ? c10::optional<torch::Tensor>(entry.k_segment_idx)
        : c10::nullopt;
    return FlashSCAFwd(
        q, k, v, chunk_size, scale, prev_k, prev_v,
        q_segment_idx, k_segment_idx, backend,
        reset_chunk_pos_per_seq);
  }

  torch::Tensor q_packed = q.flatten(0, 1);
  torch::Tensor k_combined;
  torch::Tensor v_combined;
  if (prev_k.has_value()) {
    k_combined =
        torch::cat({prev_k.value(), k}, 1).flatten(0, 1);
    v_combined =
        torch::cat({prev_v.value(), v}, 1).flatten(0, 1);
  } else {
    k_combined = k.flatten(0, 1);
    v_combined = v.flatten(0, 1);
  }
  const c10::optional<torch::Tensor> position_offsets =
      entry.position_offsets.defined()
      ? c10::optional<torch::Tensor>(entry.position_offsets)
      : c10::nullopt;
  const c10::optional<torch::Tensor> k_run_starts =
      entry.k_run_starts.defined()
      ? c10::optional<torch::Tensor>(entry.k_run_starts)
      : c10::nullopt;
  const c10::optional<torch::Tensor> k_run_lengths =
      entry.k_run_lengths.defined()
      ? c10::optional<torch::Tensor>(entry.k_run_lengths)
      : c10::nullopt;
  auto [y_packed, lse_packed] = FlashSCASM90VarlenFwd(
      q_packed, k_combined, v_combined,
      entry.cu_seqlens_q, entry.cu_seqlens_k,
      entry.max_seqlen_q, entry.max_seqlen_k,
      chunk_size, scale, position_offsets,
      reset_chunk_pos_per_seq, k_run_starts, k_run_lengths, false);
  torch::Tensor y = y_packed.reshape(
      {q.size(0), q.size(1), q.size(2), v.size(3)});
  torch::Tensor lse = lse_packed
      .reshape({q.size(2), q.size(0), q.size(1)})
      .permute({1, 0, 2})
      .contiguous();
  return std::make_tuple(std::move(y), std::move(lse));
}

c10::optional<std::tuple<torch::Tensor, torch::Tensor>>
FlashSCATryInferenceBosFwdPlan(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size,
    double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const torch::Tensor& bos,
    const std::string& backend,
    bool reset_chunk_pos_per_seq) {
  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int index = FlashSCAFindInferenceBosFwdPlanCacheEntry(
      q, k, v, chunk_size, prev_k, prev_v, bos,
      reset_chunk_pos_per_seq, stream);
  if (index < 0) {
    return c10::nullopt;
  }
  g_flash_sca_inference_bos_fwd_plan_cache_last = index;
  return FlashSCAExecuteInferenceBosFwdPlan(
      g_flash_sca_inference_bos_fwd_plan_cache[index],
      q, k, v, chunk_size, scale, prev_k, prev_v, backend,
      reset_chunk_pos_per_seq);
}

int FlashSCAStoreInferenceBosFwdPlanEntry(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const torch::Tensor& bos,
    bool reset_chunk_pos_per_seq,
    int64_t route,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const c10::optional<torch::Tensor>& cu_seqlens_q,
    const c10::optional<torch::Tensor>& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    const c10::optional<torch::Tensor>& position_offsets,
    const c10::optional<torch::Tensor>& k_run_starts,
    const c10::optional<torch::Tensor>& k_run_lengths) {
  TORCH_CHECK(
      bos.dim() == 2 && bos.scalar_type() == at::kBool &&
          bos.device().type() == torch::kCUDA,
      "FlashSCA inference BOS plan requires a CUDA bool [B, L] tensor");
  TORCH_CHECK(
      !bos.unsafeGetTensorImpl()->version_counter().enabled(),
      "FlashSCA inference BOS plan requires an inference tensor");
  TORCH_CHECK(route >= 0 && route <= 2,
              "FlashSCA inference BOS plan route is invalid");
  TORCH_CHECK(
      route != 1 ||
          (q_segment_idx.has_value() && k_segment_idx.has_value()),
      "FlashSCA segment plan requires q/k segment metadata");
  TORCH_CHECK(
      route != 2 ||
          (cu_seqlens_q.has_value() && cu_seqlens_k.has_value() &&
           max_seqlen_q > 0 && max_seqlen_k > 0),
      "FlashSCA varlen plan requires sequence metadata");

  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int index = -1;
  for (int candidate = 0;
       candidate < kFlashSCAInferenceBosFwdPlanCacheCapacity;
       ++candidate) {
    const auto& entry =
        g_flash_sca_inference_bos_fwd_plan_cache[candidate];
    if (!entry.initialized ||
        !entry.bos.has_value() ||
        entry.bos.value().expired()) {
      index = candidate;
      break;
    }
  }
  if (index < 0) {
    index = g_flash_sca_inference_bos_fwd_plan_cache_next;
  }
  g_flash_sca_inference_bos_fwd_plan_cache_next =
      (index + 1) % kFlashSCAInferenceBosFwdPlanCacheCapacity;

  FlashSCAInferenceBosFwdPlanCacheEntry entry;
  entry.bos = FlashSCAWeakTensorImpl(bos.getIntrusivePtr());
  entry.q = FlashSCACaptureTensorLayout(q);
  entry.k = FlashSCACaptureTensorLayout(k);
  entry.v = FlashSCACaptureTensorLayout(v);
  if (prev_k.has_value()) {
    entry.prev_k = FlashSCACaptureTensorLayout(prev_k.value());
  }
  if (prev_v.has_value()) {
    entry.prev_v = FlashSCACaptureTensorLayout(prev_v.value());
  }
  entry.q_segment_idx =
      q_segment_idx.has_value() ? q_segment_idx.value() : torch::Tensor();
  entry.k_segment_idx =
      k_segment_idx.has_value() ? k_segment_idx.value() : torch::Tensor();
  entry.cu_seqlens_q =
      cu_seqlens_q.has_value() ? cu_seqlens_q.value() : torch::Tensor();
  entry.cu_seqlens_k =
      cu_seqlens_k.has_value() ? cu_seqlens_k.value() : torch::Tensor();
  entry.position_offsets =
      position_offsets.has_value()
      ? position_offsets.value()
      : torch::Tensor();
  entry.k_run_starts =
      k_run_starts.has_value() ? k_run_starts.value() : torch::Tensor();
  entry.k_run_lengths =
      k_run_lengths.has_value() ? k_run_lengths.value() : torch::Tensor();
  entry.bos_size_0 = bos.size(0);
  entry.bos_size_1 = bos.size(1);
  entry.bos_stride_0 = bos.stride(0);
  entry.bos_stride_1 = bos.stride(1);
  entry.chunk_size = chunk_size;
  entry.max_seqlen_q = max_seqlen_q;
  entry.max_seqlen_k = max_seqlen_k;
  entry.route = route;
  entry.device = bos.get_device();
  entry.reset_chunk_pos_per_seq = reset_chunk_pos_per_seq;
  entry.stream = stream;
  entry.initialized = true;
  g_flash_sca_inference_bos_fwd_plan_cache[index] = std::move(entry);
  g_flash_sca_inference_bos_fwd_plan_cache_last = index;
  return index;
}

void FlashSCAStoreInferenceBosFwdPlan(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const torch::Tensor& bos,
    bool reset_chunk_pos_per_seq,
    int64_t route,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const c10::optional<torch::Tensor>& cu_seqlens_q,
    const c10::optional<torch::Tensor>& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    const c10::optional<torch::Tensor>& position_offsets,
    const c10::optional<torch::Tensor>& k_run_starts,
    const c10::optional<torch::Tensor>& k_run_lengths) {
  static_cast<void>(FlashSCAStoreInferenceBosFwdPlanEntry(
      q, k, v, chunk_size, prev_k, prev_v, bos,
      reset_chunk_pos_per_seq, route, q_segment_idx, k_segment_idx,
      cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
      position_offsets, k_run_starts, k_run_lengths));
}

std::tuple<torch::Tensor, torch::Tensor>
FlashSCAStoreAndExecuteInferenceBosFwdPlan(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size,
    double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const torch::Tensor& bos,
    const std::string& backend,
    bool reset_chunk_pos_per_seq,
    int64_t route,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const c10::optional<torch::Tensor>& cu_seqlens_q,
    const c10::optional<torch::Tensor>& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    const c10::optional<torch::Tensor>& position_offsets,
    const c10::optional<torch::Tensor>& k_run_starts,
    const c10::optional<torch::Tensor>& k_run_lengths) {
  const int index = FlashSCAStoreInferenceBosFwdPlanEntry(
      q, k, v, chunk_size, prev_k, prev_v, bos,
      reset_chunk_pos_per_seq, route, q_segment_idx, k_segment_idx,
      cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
      position_offsets, k_run_starts, k_run_lengths);
  return FlashSCAExecuteInferenceBosFwdPlan(
      g_flash_sca_inference_bos_fwd_plan_cache[index],
      q, k, v, chunk_size, scale, prev_k, prev_v, backend,
      reset_chunk_pos_per_seq);
}

void FlashSCAClearInferenceBosFwdPlanCache() {
  for (auto& entry : g_flash_sca_inference_bos_fwd_plan_cache) {
    entry = FlashSCAInferenceBosFwdPlanCacheEntry{};
  }
  g_flash_sca_inference_bos_fwd_plan_cache_next = 0;
  g_flash_sca_inference_bos_fwd_plan_cache_last = -1;
}
#endif

}  // namespace

bool FlashSCASM90BackendAvailable(const torch::Tensor& q) {
#ifdef XATTN_HAS_SM90
  return FlashSCASM90Available(q);
#else
  return false;
#endif
}

std::tuple<torch::Tensor, torch::Tensor> FlashSCAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    bool reset_chunk_pos_per_seq,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  TORCH_CHECK(q.device().type() == torch::kCUDA);
  TORCH_CHECK(k.device().type() == torch::kCUDA);
  TORCH_CHECK(v.device().type() == torch::kCUDA);
  CheckOptionalCUDA(prev_k, "prev_k");
  CheckOptionalCUDA(prev_v, "prev_v");
  CheckOptionalCUDA(q_segment_idx, "q_segment_idx");
  CheckOptionalCUDA(k_segment_idx, "k_segment_idx");
  CheckOptionalCUDA(output_state, "output_state");
#ifdef XATTN_SM90_ONLY
  CheckFlashSCASM90OnlyBackend(backend);
  CheckFlashSCASM90OnlyAvailable(q);
  return FlashSCASM90Fwd(
      q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx,
      k_segment_idx, reset_chunk_pos_per_seq, false, output_state, output_fp32);
#else
  return FlashSCACUDAFwd(
      q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx,
      k_segment_idx, backend, reset_chunk_pos_per_seq, output_state, output_fp32);
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCABwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend,
    bool reset_chunk_pos_per_seq) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(q.device().type() == torch::kCUDA);
  TORCH_CHECK(k.device().type() == torch::kCUDA);
  TORCH_CHECK(v.device().type() == torch::kCUDA);
  TORCH_CHECK(y.device().type() == torch::kCUDA);
  TORCH_CHECK(lse.device().type() == torch::kCUDA);
  CheckOptionalCUDA(prev_k, "prev_k");
  CheckOptionalCUDA(prev_v, "prev_v");
  CheckOptionalCUDA(q_segment_idx, "q_segment_idx");
  CheckOptionalCUDA(k_segment_idx, "k_segment_idx");
#ifdef XATTN_SM90_ONLY
  CheckFlashSCASM90OnlyBackend(backend);
  CheckFlashSCASM90OnlyAvailable(q);
  return FlashSCASM90Bwd(
      y_grad, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic,
      reset_chunk_pos_per_seq);
#else
  return FlashSCACUDABwd(
      y_grad, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic, backend,
      reset_chunk_pos_per_seq);
#endif
}

std::tuple<torch::Tensor, torch::Tensor> FlashSCAStrictPastFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const std::string& backend,
    bool reset_chunk_pos_per_seq,
    const c10::optional<torch::Tensor>& output_state, bool output_fp32) {
  TORCH_CHECK(q.device().type() == torch::kCUDA);
  TORCH_CHECK(k.device().type() == torch::kCUDA);
  TORCH_CHECK(v.device().type() == torch::kCUDA);
  CheckOptionalCUDA(prev_k, "prev_k");
  CheckOptionalCUDA(prev_v, "prev_v");
  CheckOptionalCUDA(q_segment_idx, "q_segment_idx");
  CheckOptionalCUDA(k_segment_idx, "k_segment_idx");
  CheckOptionalCUDA(output_state, "output_state");
  CheckFlashSCASM90OnlyBackend(backend);
  CheckFlashSCASM90OnlyAvailable(q);
#ifdef XATTN_HAS_SM90
  return FlashSCASM90Fwd(
      q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx,
      k_segment_idx, reset_chunk_pos_per_seq, true, output_state, output_fp32);
#else
  TORCH_CHECK(false, "FlashSCA SM90 support is not built");
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSCAStrictPastBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    const torch::Tensor& lse, int64_t chunk_size, double scale,
    const c10::optional<torch::Tensor>& prev_k,
    const c10::optional<torch::Tensor>& prev_v,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    bool deterministic, const std::string& backend,
    bool reset_chunk_pos_per_seq) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(q.device().type() == torch::kCUDA);
  TORCH_CHECK(k.device().type() == torch::kCUDA);
  TORCH_CHECK(v.device().type() == torch::kCUDA);
  TORCH_CHECK(y.device().type() == torch::kCUDA);
  TORCH_CHECK(lse.device().type() == torch::kCUDA);
  CheckOptionalCUDA(prev_k, "prev_k");
  CheckOptionalCUDA(prev_v, "prev_v");
  CheckOptionalCUDA(q_segment_idx, "q_segment_idx");
  CheckOptionalCUDA(k_segment_idx, "k_segment_idx");
  CheckFlashSCASM90OnlyBackend(backend);
  CheckFlashSCASM90OnlyAvailable(q);
#ifdef XATTN_HAS_SM90
  return FlashSCASM90Bwd(
      y_grad, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,
      q_segment_idx, k_segment_idx, deterministic,
      reset_chunk_pos_per_seq, true);
#else
  TORCH_CHECK(false, "FlashSCA SM90 support is not built");
#endif
}

void DefineFlashSCAOps(py::module& m) {
  m.def("flash_sca_sm90_available",
        &FlashSCASM90BackendAvailable,
        "Return whether the FlashSCA SM90 backend is available for q",
        py::arg("q"))
#ifdef XATTN_HAS_SM90
      .def("_flash_sca_sm90_bos_to_cu_seqlens",
           &FlashSCASM90BosToCuSeqlens,
           "Build flattened cu_seqlens and max run length from a BOS mask",
           py::arg("bos"))
      .def("_flash_sca_sm90_bos_plan",
           &FlashSCASM90BosPlan,
           "Build cu_seqlens and run statistics from a BOS mask",
           py::arg("bos"))
      .def("_flash_sca_sm90_paired_bos_plan",
           &FlashSCASM90PairedBosPlan,
           "Build paired Q/K plans from a right-aligned BOS mask",
           py::arg("bos"),
           py::arg("q_row_length"))
      .def("_flash_sca_sm90_bos_to_segment_idx",
           &FlashSCASM90BosToSegmentIdx,
           "Build dense segment ids from a BOS mask",
           py::arg("bos"))
      .def("_flash_sca_sm90_segment_metadata_is_dense",
           &FlashSCASM90SegmentMetadataIsDense,
           "Return whether segment metadata has one matching run per row",
           py::arg("q_segment_idx"),
           py::arg("k_segment_idx"))
      .def("_flash_sca_sm90_varlen_metadata_is_dense",
           &FlashSCASM90VarlenMetadataIsDense,
           "Return whether 4D varlen metadata contains only row boundaries",
           py::arg("cu_seqlens_q"),
           py::arg("cu_seqlens_k"),
           py::arg("batch"),
           py::arg("q_row_length"),
           py::arg("k_row_length"),
           py::arg("position_offsets") = py::none())
      .def("_flash_sca_sm90_build_row_aligned_metadata",
           &FlashSCASM90BuildRowAlignedMetadata,
           "Build per-row right-aligned FlashSCA metadata",
           py::arg("cu_seqlens_q"),
           py::arg("cu_seqlens_k"),
           py::arg("batch"),
           py::arg("q_row_length"),
           py::arg("k_row_length"))
      .def(
          "_flash_sca_sm90_try_inference_bos_fwd_plan",
          [](const torch::Tensor& q,
             const torch::Tensor& k,
             const torch::Tensor& v,
             int64_t chunk_size,
             double scale,
             const c10::optional<torch::Tensor>& prev_k,
             const c10::optional<torch::Tensor>& prev_v,
             const torch::Tensor& bos,
             const std::string& backend,
             bool reset_chunk_pos_per_seq) -> py::object {
            auto result = FlashSCATryInferenceBosFwdPlan(
                q, k, v, chunk_size, scale, prev_k, prev_v, bos,
                backend, reset_chunk_pos_per_seq);
            if (!result.has_value()) {
              return py::none();
            }
            return py::cast(std::move(result.value()));
          },
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("chunk_size"),
          py::arg("scale"),
          py::arg("prev_k"),
          py::arg("prev_v"),
          py::arg("bos"),
          py::arg("backend") = "auto",
          py::arg("reset_chunk_pos_per_seq") = false)
      .def("_flash_sca_sm90_store_inference_bos_fwd_plan",
           &FlashSCAStoreInferenceBosFwdPlan,
           py::arg("q"),
           py::arg("k"),
           py::arg("v"),
           py::arg("chunk_size"),
           py::arg("prev_k"),
           py::arg("prev_v"),
           py::arg("bos"),
           py::arg("reset_chunk_pos_per_seq"),
           py::arg("route"),
           py::arg("q_segment_idx") = py::none(),
           py::arg("k_segment_idx") = py::none(),
           py::arg("cu_seqlens_q") = py::none(),
           py::arg("cu_seqlens_k") = py::none(),
           py::arg("max_seqlen_q") = -1,
           py::arg("max_seqlen_k") = -1,
           py::arg("position_offsets") = py::none(),
           py::arg("k_run_starts") = py::none(),
           py::arg("k_run_lengths") = py::none())
      .def("_flash_sca_sm90_store_and_execute_inference_bos_fwd_plan",
           &FlashSCAStoreAndExecuteInferenceBosFwdPlan,
           py::arg("q"),
           py::arg("k"),
           py::arg("v"),
           py::arg("chunk_size"),
           py::arg("scale"),
           py::arg("prev_k"),
           py::arg("prev_v"),
           py::arg("bos"),
           py::arg("backend"),
           py::arg("reset_chunk_pos_per_seq"),
           py::arg("route"),
           py::arg("q_segment_idx") = py::none(),
           py::arg("k_segment_idx") = py::none(),
           py::arg("cu_seqlens_q") = py::none(),
           py::arg("cu_seqlens_k") = py::none(),
           py::arg("max_seqlen_q") = -1,
           py::arg("max_seqlen_k") = -1,
           py::arg("position_offsets") = py::none(),
           py::arg("k_run_starts") = py::none(),
           py::arg("k_run_lengths") = py::none())
      .def("_flash_sca_sm90_clear_inference_bos_fwd_plan_cache",
           &FlashSCAClearInferenceBosFwdPlanCache)
      .def("_flash_sca_sm90_varlen_fwd",
           &FlashSCASM90VarlenFwd,
           "Run prepared-varlen FlashSCA SM90 forward",
           py::arg("q"),
           py::arg("k"),
           py::arg("v"),
           py::arg("cu_seqlens_q"),
           py::arg("cu_seqlens_k"),
           py::arg("max_seqlen_q"),
           py::arg("max_seqlen_k"),
           py::arg("chunk_size"),
           py::arg("scale"),
           py::arg("position_offsets") = py::none(),
           py::arg("reset_chunk_pos_per_seq") = true,
           py::arg("k_run_starts") = py::none(),
           py::arg("k_run_lengths") = py::none(),
           py::arg("strict_past") = false,
           py::arg("output_state") = py::none(),
           py::arg("output_fp32") = false)
      .def("_flash_sca_sm90_varlen_bwd",
           &FlashSCASM90VarlenBwd,
           "Run prepared-varlen FlashSCA SM90 backward",
           py::arg("y_grad"),
           py::arg("q"),
           py::arg("k"),
           py::arg("v"),
           py::arg("y"),
           py::arg("lse"),
           py::arg("cu_seqlens_q"),
           py::arg("cu_seqlens_k"),
           py::arg("max_seqlen_q"),
           py::arg("max_seqlen_k"),
           py::arg("chunk_size"),
           py::arg("scale"),
           py::arg("deterministic") = false,
           py::arg("position_offsets") = py::none(),
           py::arg("reset_chunk_pos_per_seq") = true,
           py::arg("k_run_starts") = py::none(),
           py::arg("k_run_lengths") = py::none(),
           py::arg("k_prefix_ends") = py::none(),
           py::arg("k_row_length") = 0,
           py::arg("strict_past") = false)
#endif
      .def("_flash_sca_sm90_strict_past_fwd",
           &FlashSCAStrictPastFwd,
           "Strict-past FlashSCA SM90 FWD",
           py::arg("q"),
           py::arg("k"),
           py::arg("v"),
           py::arg("chunk_size"),
           py::arg("scale"),
           py::arg("prev_k") = py::none(),
           py::arg("prev_v") = py::none(),
           py::arg("q_segment_idx") = py::none(),
           py::arg("k_segment_idx") = py::none(),
           py::arg("backend") = "auto",
           py::arg("reset_chunk_pos_per_seq") = false,
           py::arg("output_state") = py::none(),
           py::arg("output_fp32") = false)
      .def("_flash_sca_sm90_strict_past_bwd",
           &FlashSCAStrictPastBwd,
           "Strict-past FlashSCA SM90 BWD",
           py::arg("y_grad"),
           py::arg("q"),
           py::arg("k"),
           py::arg("v"),
           py::arg("y"),
           py::arg("lse"),
           py::arg("chunk_size"),
           py::arg("scale"),
           py::arg("prev_k") = py::none(),
           py::arg("prev_v") = py::none(),
           py::arg("q_segment_idx") = py::none(),
           py::arg("k_segment_idx") = py::none(),
           py::arg("deterministic") = false,
           py::arg("backend") = "auto",
           py::arg("reset_chunk_pos_per_seq") = false)
      .def("flash_sca_fwd",
           &FlashSCAFwd, "FlashSCAFwd",
           py::arg("q"),
           py::arg("k"),
           py::arg("v"),
           py::arg("chunk_size"),
           py::arg("scale"),
           py::arg("prev_k"),
           py::arg("prev_v"),
           py::arg("q_segment_idx"),
           py::arg("k_segment_idx"),
           py::arg("backend") = "auto",
           py::arg("reset_chunk_pos_per_seq") = false,
           py::arg("output_state") = py::none(),
           py::arg("output_fp32") = false)
      .def("flash_sca_bwd",
           &FlashSCABwd, "FlashSCABwd",
           py::arg("y_grad"),
           py::arg("q"),
           py::arg("k"),
           py::arg("v"),
           py::arg("y"),
           py::arg("lse"),
           py::arg("chunk_size"),
           py::arg("scale"),
           py::arg("prev_k"),
           py::arg("prev_v"),
           py::arg("q_segment_idx"),
           py::arg("k_segment_idx"),
           py::arg("deterministic") = false,
           py::arg("backend") = "auto",
           py::arg("reset_chunk_pos_per_seq") = false);
}

}  // namespace ops
}  // namespace xattn

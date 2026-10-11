// Author: Shicheng Wen

#include "softdelta/gradient_merge.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace xattn {
namespace ops {
namespace {

void CheckPair(
    const torch::Tensor& first,
    const torch::Tensor& second,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(first.sizes() == second.sizes(), name, " shapes must match");
  TORCH_CHECK(first.device() == reference.device() &&
                  second.device() == reference.device(),
              name, " tensors must share a device");
  TORCH_CHECK(first.scalar_type() == reference.scalar_type() &&
                  second.scalar_type() == reference.scalar_type(),
              name, " tensors must share a dtype");
  TORCH_CHECK(first.is_contiguous() && second.is_contiguous(),
              name, " tensors must be contiguous");
}

void CheckOptionalPair(
    const c10::optional<torch::Tensor>& first,
    const c10::optional<torch::Tensor>& second,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(first.has_value() == second.has_value(),
              name, " tensors must both be present or absent");
  if (first.has_value()) {
    CheckPair(*first, *second, reference, name);
  }
}

// Two packed elements summed in FP32, as in the scalar loop below.
template <typename scalar_t>
__device__ __forceinline__ uint32_t AddPair(uint32_t first, uint32_t second) {
  scalar_t a[2], b[2], sum[2];
  memcpy(a, &first, sizeof(first));
  memcpy(b, &second, sizeof(second));
  for (int i = 0; i < 2; ++i) {
    sum[i] = static_cast<scalar_t>(
        static_cast<float>(a[i]) + static_cast<float>(b[i]));
  }
  uint32_t result;
  memcpy(&result, sum, sizeof(result));
  return result;
}

// Adds 16-byte chunk `offset` of one gradient pair, or moves `offset` past
// this pair's `chunks`.
template <typename scalar_t>
__device__ __forceinline__ bool AddChunk(
    const scalar_t* first, const scalar_t* second, scalar_t* output,
    int64_t chunks, int64_t& offset) {
  if (offset >= chunks) {
    offset -= chunks;
    return false;
  }
  const uint4 a = reinterpret_cast<const uint4*>(first)[offset];
  const uint4 b = reinterpret_cast<const uint4*>(second)[offset];
  reinterpret_cast<uint4*>(output)[offset] = make_uint4(
      AddPair<scalar_t>(a.x, b.x), AddPair<scalar_t>(a.y, b.y),
      AddPair<scalar_t>(a.z, b.z), AddPair<scalar_t>(a.w, b.w));
  return true;
}

template <typename scalar_t>
__global__ void FlashSoftDeltaMergeGradientsKernel(
    const scalar_t* read_q,
    const scalar_t* correction_q,
    const scalar_t* read_k,
    const scalar_t* correction_k,
    const scalar_t* read_v,
    const scalar_t* correction_v,
    const scalar_t* read_prev_k,
    const scalar_t* correction_prev_k,
    const scalar_t* read_prev_v,
    const scalar_t* correction_prev_v,
    scalar_t* q,
    scalar_t* k,
    scalar_t* v,
    scalar_t* prev_k,
    scalar_t* prev_v,
    int64_t q_prefixes,
    int64_t q_heads,
    int64_t q_head_dim,
    bool vectorized_q,
    bool vectorized_sums,
    int64_t k_elements,
    int64_t v_elements,
    int64_t prev_k_elements,
    int64_t prev_v_elements,
    int64_t total_elements) {
  if (blockIdx.x < q_prefixes) {
    const int64_t prefix = blockIdx.x;
    constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
    if (vectorized_q) {
      const int vectors_per_head = q_head_dim / kScalarsPerVector;
      const int vectors_per_prefix = q_heads * vectors_per_head;
      const uint4* read_vectors = reinterpret_cast<const uint4*>(read_q);
      const uint4* correction_vectors =
          reinterpret_cast<const uint4*>(correction_q);
      uint4* output_vectors = reinterpret_cast<uint4*>(q);
      for (int source_offset = threadIdx.x;
           source_offset < vectors_per_prefix;
           source_offset += blockDim.x) {
        const int head = source_offset / vectors_per_head;
        const int column = source_offset - head * vectors_per_head;
        const int64_t source = prefix * vectors_per_prefix + source_offset;
        const int64_t destination =
            (prefix * 2 * q_heads + 2 * head) * vectors_per_head + column;
        output_vectors[destination] = read_vectors[source];
        output_vectors[destination + vectors_per_head] =
            correction_vectors[source];
      }
    } else {
      const int elements_per_prefix = q_heads * q_head_dim;
      for (int source_offset = threadIdx.x;
           source_offset < elements_per_prefix;
           source_offset += blockDim.x) {
        const int head = source_offset / q_head_dim;
        const int column = source_offset - head * q_head_dim;
        const int64_t source = prefix * elements_per_prefix + source_offset;
        const int64_t destination =
            (prefix * 2 * q_heads + 2 * head) * q_head_dim + column;
        q[destination] = read_q[source];
        q[destination + q_head_dim] = correction_q[source];
      }
    }
    return;
  }
  const int64_t add_block = int64_t(blockIdx.x) - q_prefixes;
  if (vectorized_sums) {
    constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
    for (int64_t index = add_block * blockDim.x + threadIdx.x;
         index < total_elements / kScalarsPerVector;
         index += int64_t(blockDim.x) * (gridDim.x - q_prefixes)) {
      int64_t offset = index;
      if (AddChunk(read_k, correction_k, k,
                   k_elements / kScalarsPerVector, offset) ||
          AddChunk(read_v, correction_v, v,
                   v_elements / kScalarsPerVector, offset) ||
          AddChunk(read_prev_k, correction_prev_k, prev_k,
                   prev_k_elements / kScalarsPerVector, offset)) {
        continue;
      }
      AddChunk(read_prev_v, correction_prev_v, prev_v,
               prev_v_elements / kScalarsPerVector, offset);
    }
    return;
  }
  for (int64_t index = add_block * blockDim.x + threadIdx.x;
       index < total_elements;
       index += int64_t(blockDim.x) * (gridDim.x - q_prefixes)) {
    int64_t offset = index;
    if (offset < k_elements) {
      k[offset] = static_cast<scalar_t>(
          static_cast<float>(read_k[offset]) +
          static_cast<float>(correction_k[offset]));
      continue;
    }
    offset -= k_elements;
    if (offset < v_elements) {
      v[offset] = static_cast<scalar_t>(
          static_cast<float>(read_v[offset]) +
          static_cast<float>(correction_v[offset]));
      continue;
    }
    offset -= v_elements;
    if (offset < prev_k_elements) {
      prev_k[offset] = static_cast<scalar_t>(
          static_cast<float>(read_prev_k[offset]) +
          static_cast<float>(correction_prev_k[offset]));
      continue;
    }
    offset -= prev_k_elements;
    if (offset < prev_v_elements) {
      prev_v[offset] = static_cast<scalar_t>(
          static_cast<float>(read_prev_v[offset]) +
          static_cast<float>(correction_prev_v[offset]));
    }
  }
}

template <typename scalar_t>
void LaunchMergeGradients(
    const torch::Tensor& read_q_grad,
    const torch::Tensor& correction_q_grad,
    const torch::Tensor& read_k_grad,
    const torch::Tensor& correction_k_grad,
    const torch::Tensor& read_v_grad,
    const torch::Tensor& correction_v_grad,
    const c10::optional<torch::Tensor>& read_prev_k_grad,
    const c10::optional<torch::Tensor>& correction_prev_k_grad,
    const c10::optional<torch::Tensor>& read_prev_v_grad,
    const c10::optional<torch::Tensor>& correction_prev_v_grad,
    torch::Tensor& q_grad,
    torch::Tensor& k_grad,
    torch::Tensor& v_grad,
    c10::optional<torch::Tensor>& prev_k_grad,
    c10::optional<torch::Tensor>& prev_v_grad) {
  const int64_t k_elements = k_grad.numel();
  const int64_t v_elements = v_grad.numel();
  const int64_t prev_k_elements =
      prev_k_grad.has_value() ? prev_k_grad->numel() : 0;
  const int64_t prev_v_elements =
      prev_v_grad.has_value() ? prev_v_grad->numel() : 0;
  const int64_t total_elements = k_elements + v_elements +
      prev_k_elements + prev_v_elements;
  constexpr int kThreads = 256;
  constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
  auto aligned = [](const c10::optional<torch::Tensor>& tensor) {
    return !tensor.has_value() ||
        reinterpret_cast<std::uintptr_t>(tensor->data_ptr()) %
                sizeof(uint4) == 0;
  };
  const bool vectorized_sums =
      k_elements % kScalarsPerVector == 0 &&
      v_elements % kScalarsPerVector == 0 &&
      prev_k_elements % kScalarsPerVector == 0 &&
      prev_v_elements % kScalarsPerVector == 0 &&
      aligned(read_k_grad) && aligned(correction_k_grad) && aligned(k_grad) &&
      aligned(read_v_grad) && aligned(correction_v_grad) && aligned(v_grad) &&
      aligned(read_prev_k_grad) && aligned(correction_prev_k_grad) &&
      aligned(prev_k_grad) && aligned(read_prev_v_grad) &&
      aligned(correction_prev_v_grad) && aligned(prev_v_grad);
  const int64_t add_items =
      vectorized_sums ? total_elements / kScalarsPerVector : total_elements;
  const int add_blocks = static_cast<int>(
      (add_items + kThreads - 1) / kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t q_head_dim = read_q_grad.size(3);
  const int64_t q_prefixes =
      read_q_grad.numel() / (read_q_grad.size(2) * q_head_dim);
  FlashSoftDeltaMergeGradientsKernel<scalar_t>
      <<<q_prefixes + add_blocks, kThreads, 0, stream>>>(
          read_q_grad.data_ptr<scalar_t>(),
          correction_q_grad.data_ptr<scalar_t>(),
          read_k_grad.data_ptr<scalar_t>(),
          correction_k_grad.data_ptr<scalar_t>(),
          read_v_grad.data_ptr<scalar_t>(),
          correction_v_grad.data_ptr<scalar_t>(),
          read_prev_k_grad.has_value()
              ? read_prev_k_grad->data_ptr<scalar_t>()
              : nullptr,
          correction_prev_k_grad.has_value()
              ? correction_prev_k_grad->data_ptr<scalar_t>()
              : nullptr,
          read_prev_v_grad.has_value()
              ? read_prev_v_grad->data_ptr<scalar_t>()
              : nullptr,
          correction_prev_v_grad.has_value()
              ? correction_prev_v_grad->data_ptr<scalar_t>()
              : nullptr,
          q_grad.data_ptr<scalar_t>(), k_grad.data_ptr<scalar_t>(),
          v_grad.data_ptr<scalar_t>(),
          prev_k_grad.has_value() ? prev_k_grad->data_ptr<scalar_t>() : nullptr,
          prev_v_grad.has_value() ? prev_v_grad->data_ptr<scalar_t>() : nullptr,
          q_prefixes, read_q_grad.size(2), q_head_dim,
          q_head_dim % kScalarsPerVector == 0, vectorized_sums,
          k_elements, v_elements, prev_k_elements, prev_v_elements,
          total_elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FlashSoftDeltaMergeGradients(
    const torch::Tensor& read_q_grad,
    const torch::Tensor& correction_q_grad,
    const torch::Tensor& read_k_grad,
    const torch::Tensor& correction_k_grad,
    const torch::Tensor& read_v_grad,
    const torch::Tensor& correction_v_grad,
    const c10::optional<torch::Tensor>& read_prev_k_grad,
    const c10::optional<torch::Tensor>& correction_prev_k_grad,
    const c10::optional<torch::Tensor>& read_prev_v_grad,
    const c10::optional<torch::Tensor>& correction_prev_v_grad) {
  TORCH_CHECK(read_q_grad.is_cuda(),
              "Flash SoftDelta gradient merge requires CUDA tensors");
  TORCH_CHECK(read_q_grad.dim() == 4,
              "Flash SoftDelta Q gradients must be 4D");
  TORCH_CHECK(read_q_grad.scalar_type() == at::kHalf ||
                  read_q_grad.scalar_type() == at::kBFloat16,
              "Flash SoftDelta gradient merge supports only fp16 and bf16");
  CheckPair(read_q_grad, correction_q_grad, read_q_grad, "Q gradient");
  CheckPair(read_k_grad, correction_k_grad, read_q_grad, "K gradient");
  CheckPair(read_v_grad, correction_v_grad, read_q_grad, "V gradient");
  CheckOptionalPair(
      read_prev_k_grad, correction_prev_k_grad, read_q_grad,
      "previous K gradient");
  CheckOptionalPair(
      read_prev_v_grad, correction_prev_v_grad, read_q_grad,
      "previous V gradient");
  c10::cuda::CUDAGuard guard(read_q_grad.device());

  torch::Tensor q_grad = torch::empty(
      {read_q_grad.size(0), read_q_grad.size(1),
       2 * read_q_grad.size(2), read_q_grad.size(3)},
      read_q_grad.options());
  torch::Tensor k_grad = torch::empty_like(read_k_grad);
  torch::Tensor v_grad = torch::empty_like(read_v_grad);
  c10::optional<torch::Tensor> prev_k_grad = c10::nullopt;
  c10::optional<torch::Tensor> prev_v_grad = c10::nullopt;
  if (read_prev_k_grad.has_value()) {
    prev_k_grad = torch::empty_like(*read_prev_k_grad);
  }
  if (read_prev_v_grad.has_value()) {
    prev_v_grad = torch::empty_like(*read_prev_v_grad);
  }

  if (read_q_grad.scalar_type() == at::kHalf) {
    LaunchMergeGradients<at::Half>(
        read_q_grad, correction_q_grad, read_k_grad, correction_k_grad,
        read_v_grad, correction_v_grad, read_prev_k_grad,
        correction_prev_k_grad, read_prev_v_grad, correction_prev_v_grad,
        q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad);
  } else {
    LaunchMergeGradients<at::BFloat16>(
        read_q_grad, correction_q_grad, read_k_grad, correction_k_grad,
        read_v_grad, correction_v_grad, read_prev_k_grad,
        correction_prev_k_grad, read_prev_v_grad, correction_prev_v_grad,
        q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad);
  }
  return {q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad};
}

}  // namespace ops
}  // namespace xattn

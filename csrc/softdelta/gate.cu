// Author: Shicheng Wen

#include "softdelta/gate.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace xattn {
namespace ops {
namespace {

void CheckGateTensor(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(tensor.device() == reference.device(),
              name, " must share a device with read");
  TORCH_CHECK(tensor.scalar_type() == reference.scalar_type(),
              name, " must share a dtype with read");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckGateInputs(
    const torch::Tensor& read,
    const torch::Tensor& correction,
    const torch::Tensor& gate) {
  TORCH_CHECK(read.is_cuda(), "Flash SoftDelta gate requires CUDA tensors");
  TORCH_CHECK(read.dim() == 5 && correction.dim() == 5 && gate.dim() == 5,
              "Flash SoftDelta gate expects 5D tensors");
  TORCH_CHECK(read.scalar_type() == at::kHalf ||
                  read.scalar_type() == at::kBFloat16,
              "Flash SoftDelta gate supports only fp16 and bf16");
  TORCH_CHECK(read.is_contiguous(), "read must be contiguous");
  CheckGateTensor(correction, read, "correction");
  CheckGateTensor(gate, read, "gate");
  TORCH_CHECK(read.sizes() == correction.sizes(),
              "read and correction shapes must match");
  TORCH_CHECK(gate.size(0) == read.size(0) &&
                  gate.size(1) == read.size(1) &&
                  gate.size(2) == read.size(2) &&
                  gate.size(3) == read.size(3) && gate.size(4) == 1,
              "gate shape must match read except for a unit final dimension");
  TORCH_CHECK(read.numel() > 0, "Flash SoftDelta gate tensors must be nonempty");
}

void CheckPairGateInputs(
    const torch::Tensor& pair_output,
    const torch::Tensor& gate) {
  TORCH_CHECK(
      pair_output.is_cuda(),
      "Flash SoftDelta pair gate requires CUDA tensors");
  TORCH_CHECK(
      pair_output.dim() == 4 && gate.dim() == 5,
      "Flash SoftDelta pair gate expects 4D reads and a 5D gate");
  TORCH_CHECK(
      pair_output.scalar_type() == at::kHalf ||
          pair_output.scalar_type() == at::kBFloat16,
      "Flash SoftDelta pair gate supports only fp16 and bf16");
  TORCH_CHECK(
      pair_output.device() == gate.device() &&
          pair_output.scalar_type() == gate.scalar_type(),
      "Flash SoftDelta pair reads and gate must share device and dtype");
  TORCH_CHECK(
      pair_output.is_contiguous() && gate.is_contiguous(),
      "Flash SoftDelta pair reads and gate must be contiguous");
  TORCH_CHECK(
      pair_output.size(2) == 2 * gate.size(2) &&
          pair_output.size(0) == gate.size(0) &&
          pair_output.size(1) == gate.size(1) && gate.size(4) == 1,
      "Flash SoftDelta pair reads and gate shapes are incompatible");
  TORCH_CHECK(
      gate.size(3) > 0 &&
          pair_output.size(3) % gate.size(3) == 0,
      "Flash SoftDelta pair value dim must be divisible by gate groups");
  TORCH_CHECK(
      pair_output.numel() > 0,
      "Flash SoftDelta pair gate tensors must be nonempty");
}

void CheckPairGateBackwardInputs(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_gate_state,
    const torch::Tensor& gate) {
  TORCH_CHECK(
      output_grad.is_cuda() && pair_gate_state.is_cuda() && gate.is_cuda(),
      "Flash SoftDelta pair gate BWD requires CUDA tensors");
  TORCH_CHECK(
      output_grad.dim() == 5 && pair_gate_state.dim() == 4 &&
          gate.dim() == 5,
      "Flash SoftDelta pair gate BWD expects a 5D output_grad, 4D pair "
      "state, and 5D gate");
  TORCH_CHECK(
      output_grad.scalar_type() == at::kHalf ||
          output_grad.scalar_type() == at::kBFloat16,
      "Flash SoftDelta pair gate BWD supports only fp16 and bf16 "
      "output gradients");
  TORCH_CHECK(
      output_grad.device() == pair_gate_state.device() &&
          output_grad.device() == gate.device() &&
          output_grad.scalar_type() == gate.scalar_type(),
      "Flash SoftDelta pair gate BWD tensors must share a device, and "
      "output_grad and gate must share a dtype");
  TORCH_CHECK(
      pair_gate_state.scalar_type() == output_grad.scalar_type() ||
          pair_gate_state.scalar_type() == at::kFloat,
      "Flash SoftDelta pair gate BWD state must use the output dtype or "
      "float32");
  TORCH_CHECK(
      output_grad.is_contiguous() && pair_gate_state.is_contiguous() &&
          gate.is_contiguous(),
      "Flash SoftDelta pair gate BWD tensors must be contiguous");
  TORCH_CHECK(
      pair_gate_state.size(2) == 2 * gate.size(2) &&
          pair_gate_state.size(0) == gate.size(0) &&
          pair_gate_state.size(1) == gate.size(1) &&
          gate.size(4) == 1 && gate.size(3) > 0 &&
          output_grad.size(0) == gate.size(0) &&
          output_grad.size(1) == gate.size(1) &&
          output_grad.size(2) == gate.size(2) &&
          output_grad.size(3) == gate.size(3) &&
          output_grad.size(4) > 0 &&
          pair_gate_state.size(3) ==
              output_grad.size(3) * output_grad.size(4),
      "Flash SoftDelta pair gate BWD shapes are incompatible");
  TORCH_CHECK(
      pair_gate_state.numel() > 0,
      "Flash SoftDelta pair gate BWD tensors must be nonempty");
}

template <typename scalar_t>
__device__ __forceinline__ float LoadFloat(const scalar_t* pointer) {
  return static_cast<float>(*pointer);
}

template <typename scalar_t>
union alignas(16) GateVector {
  uint4 packed;
  scalar_t values[sizeof(uint4) / sizeof(scalar_t)];
};

union alignas(16) FloatGateVector {
  uint4 packed;
  float values[sizeof(uint4) / sizeof(float)];
};

template <typename scalar_t>
__device__ __forceinline__ float RoundStateToScalar(float value) {
  return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float Sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaGateFwdKernel(
    const scalar_t* read,
    const scalar_t* correction,
    const scalar_t* gate,
    scalar_t* output,
    int64_t rows,
    int64_t group_dim) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }
  float gate_value =
      sublane == 0 ? Sigmoid(LoadFloat(gate + row)) : 0.0f;
  gate_value = __shfl_sync(
      active_mask, gate_value, 0, kSubwarpSize);
  for (int64_t column = sublane; column < group_dim;
       column += kSubwarpSize) {
    const int64_t index = row * group_dim + column;
    const float value =
        LoadFloat(read + index) - gate_value * LoadFloat(correction + index);
    output[index] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaGateBwdKernel(
    const scalar_t* output_grad,
    const scalar_t* correction,
    const scalar_t* gate,
    scalar_t* correction_grad,
    scalar_t* gate_grad,
    int64_t rows,
    int64_t group_dim) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }

  const float gate_value = Sigmoid(LoadFloat(gate + row));
  float gate_sum = 0.0f;
  for (int64_t column = sublane; column < group_dim;
       column += kSubwarpSize) {
    const int64_t index = row * group_dim + column;
    const float grad = LoadFloat(output_grad + index);
    const float correction_value = LoadFloat(correction + index);
    correction_grad[index] = static_cast<scalar_t>(-gate_value * grad);
    gate_sum -= correction_value * grad;
  }
  for (int offset = kSubwarpSize / 2; offset > 0; offset /= 2) {
    gate_sum += __shfl_down_sync(
        active_mask, gate_sum, offset, kSubwarpSize);
  }
  if (sublane == 0) {
    gate_grad[row] = static_cast<scalar_t>(
        gate_sum * gate_value * (1.0f - gate_value));
  }
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaPairGateBwdKernel(
    const scalar_t* output_grad,
    const scalar_t* pair_output,
    const scalar_t* gate,
    scalar_t* pair_grad,
    scalar_t* gate_grad,
    int64_t rows,
    int groups,
    int group_dim) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }

  const float gate_value = Sigmoid(LoadFloat(gate + row));
  const int group = static_cast<int>(row % groups);
  const int64_t logical_pair = row / groups;
  const int64_t value_dim = int64_t(groups) * group_dim;
  const int64_t pair_base = logical_pair * 2 * value_dim;
  const int64_t read_base = pair_base + int64_t(group) * group_dim;
  const int64_t correction_base = read_base + value_dim;
  const int64_t output_grad_base = row * group_dim;
  float gate_sum = 0.0f;
  for (int column = sublane; column < group_dim;
       column += kSubwarpSize) {
    const float grad = LoadFloat(output_grad + output_grad_base + column);
    pair_grad[read_base + column] = static_cast<scalar_t>(grad);
    pair_grad[correction_base + column] =
        static_cast<scalar_t>(-gate_value * grad);
    gate_sum -=
        LoadFloat(pair_output + correction_base + column) * grad;
  }
  for (int offset = kSubwarpSize / 2; offset > 0; offset /= 2) {
    gate_sum += __shfl_down_sync(
        active_mask, gate_sum, offset, kSubwarpSize);
  }
  if (sublane == 0) {
    gate_grad[row] = static_cast<scalar_t>(
        gate_sum * gate_value * (1.0f - gate_value));
  }
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaPairGateBwdStateKernel(
    const scalar_t* output_grad,
    const float* pair_gate_state,
    const scalar_t* gate,
    scalar_t* pair_grad,
    scalar_t* gate_grad,
    int64_t rows,
    int groups,
    int group_dim) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }

  const float gate_value = Sigmoid(LoadFloat(gate + row));
  const int group = static_cast<int>(row % groups);
  const int64_t logical_pair = row / groups;
  const int64_t value_dim = int64_t(groups) * group_dim;
  const int64_t pair_base = logical_pair * 2 * value_dim;
  const int64_t read_base = pair_base + int64_t(group) * group_dim;
  const int64_t correction_base = read_base + value_dim;
  const int64_t output_grad_base = row * group_dim;
  float gate_sum = 0.0f;
  for (int column = sublane; column < group_dim;
       column += kSubwarpSize) {
    const float grad = LoadFloat(output_grad + output_grad_base + column);
    pair_grad[read_base + column] = static_cast<scalar_t>(grad);
    pair_grad[correction_base + column] =
        static_cast<scalar_t>(-gate_value * grad);
    gate_sum -=
        RoundStateToScalar<scalar_t>(
            pair_gate_state[correction_base + column]) * grad;
  }
  for (int offset = kSubwarpSize / 2; offset > 0; offset /= 2) {
    gate_sum += __shfl_down_sync(
        active_mask, gate_sum, offset, kSubwarpSize);
  }
  if (sublane == 0) {
    gate_grad[row] = static_cast<scalar_t>(
        gate_sum * gate_value * (1.0f - gate_value));
  }
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaGateFwdVectorKernel(
    const scalar_t* read,
    const scalar_t* correction,
    const scalar_t* gate,
    scalar_t* output,
    int64_t rows,
    int vectors_per_row) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }
  float gate_value =
      sublane == 0 ? Sigmoid(LoadFloat(gate + row)) : 0.0f;
  gate_value = __shfl_sync(
      active_mask, gate_value, 0, kSubwarpSize);
  const uint4* read_vectors = reinterpret_cast<const uint4*>(read);
  const uint4* correction_vectors =
      reinterpret_cast<const uint4*>(correction);
  uint4* output_vectors = reinterpret_cast<uint4*>(output);
  for (int vector_column = sublane; vector_column < vectors_per_row;
       vector_column += kSubwarpSize) {
    const int64_t vector_index = row * vectors_per_row + vector_column;
    GateVector<scalar_t> read_vector;
    GateVector<scalar_t> correction_vector;
    GateVector<scalar_t> output_vector;
    read_vector.packed = read_vectors[vector_index];
    correction_vector.packed = correction_vectors[vector_index];
#pragma unroll
    for (int element = 0; element < kScalarsPerVector; ++element) {
      const float value =
          static_cast<float>(read_vector.values[element]) -
          gate_value * static_cast<float>(correction_vector.values[element]);
      output_vector.values[element] = static_cast<scalar_t>(value);
    }
    output_vectors[vector_index] = output_vector.packed;
  }
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaPairGateFwdKernel(
    const scalar_t* pair_output,
    const scalar_t* gate,
    scalar_t* output,
    int64_t rows,
    int groups,
    int group_dim) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }
  float gate_value =
      sublane == 0 ? Sigmoid(LoadFloat(gate + row)) : 0.0f;
  gate_value = __shfl_sync(active_mask, gate_value, 0, kSubwarpSize);
  const int group = static_cast<int>(row % groups);
  const int64_t pair = row / groups;
  const int64_t pair_base = pair * 2 * groups * group_dim;
  const int64_t read_base = pair_base + group * group_dim;
  const int64_t correction_base =
      pair_base + (groups + group) * group_dim;
  const int64_t output_base = row * group_dim;
  for (int column = sublane; column < group_dim;
       column += kSubwarpSize) {
    const float value =
        LoadFloat(pair_output + read_base + column) -
        gate_value * LoadFloat(pair_output + correction_base + column);
    output[output_base + column] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaPairGateFwdVectorKernel(
    const scalar_t* pair_output,
    const scalar_t* gate,
    scalar_t* output,
    int64_t rows,
    int groups,
    int vectors_per_group) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }
  float gate_value =
      sublane == 0 ? Sigmoid(LoadFloat(gate + row)) : 0.0f;
  gate_value = __shfl_sync(active_mask, gate_value, 0, kSubwarpSize);
  const int group = static_cast<int>(row % groups);
  const int64_t pair = row / groups;
  const int64_t pair_base = pair * 2 * groups * vectors_per_group;
  const int64_t read_base = pair_base + group * vectors_per_group;
  const int64_t correction_base =
      pair_base + (groups + group) * vectors_per_group;
  const int64_t output_base = row * vectors_per_group;
  const uint4* pair_vectors =
      reinterpret_cast<const uint4*>(pair_output);
  uint4* output_vectors = reinterpret_cast<uint4*>(output);
  for (int vector_column = sublane;
       vector_column < vectors_per_group;
       vector_column += kSubwarpSize) {
    GateVector<scalar_t> read_vector;
    GateVector<scalar_t> correction_vector;
    GateVector<scalar_t> output_vector;
    read_vector.packed = pair_vectors[read_base + vector_column];
    correction_vector.packed =
        pair_vectors[correction_base + vector_column];
#pragma unroll
    for (int element = 0; element < kScalarsPerVector; ++element) {
      const float value =
          static_cast<float>(read_vector.values[element]) -
          gate_value *
              static_cast<float>(correction_vector.values[element]);
      output_vector.values[element] = static_cast<scalar_t>(value);
    }
    output_vectors[output_base + vector_column] = output_vector.packed;
  }
}

template <typename scalar_t, int kVectorsPerGroup>
__global__ void FlashSoftDeltaPairGateFwdFourGroupVectorKernel(
    const scalar_t* pair_output,
    const scalar_t* gate,
    scalar_t* output,
    int64_t pairs) {
  constexpr int kWarpSize = 32;
  constexpr int kGroups = 4;
  constexpr int kLanesPerPair = kGroups * kVectorsPerGroup;
  constexpr int kPairsPerWarp = kWarpSize / kLanesPerPair;
  constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
  static_assert(kWarpSize % kLanesPerPair == 0);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int pair_lane = lane % kLanesPerPair;
  const int pair_in_warp = lane / kLanesPerPair;
  const int group = pair_lane / kVectorsPerGroup;
  const int vector_column = pair_lane % kVectorsPerGroup;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t pair =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kPairsPerWarp +
      pair_in_warp;
  const bool active = pair < pairs;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }
  float gate_value = vector_column == 0
      ? Sigmoid(LoadFloat(gate + pair * kGroups + group))
      : 0.0f;
  gate_value = __shfl_sync(
      active_mask, gate_value, 0, kVectorsPerGroup);
  constexpr int kReadVectors = kGroups * kVectorsPerGroup;
  const int64_t read_index =
      pair * 2 * kReadVectors + group * kVectorsPerGroup + vector_column;
  const int64_t correction_index = read_index + kReadVectors;
  const int64_t output_index =
      pair * kReadVectors + group * kVectorsPerGroup + vector_column;
  const uint4* pair_vectors =
      reinterpret_cast<const uint4*>(pair_output);
  uint4* output_vectors = reinterpret_cast<uint4*>(output);
  GateVector<scalar_t> read_vector;
  GateVector<scalar_t> correction_vector;
  GateVector<scalar_t> output_vector;
  read_vector.packed = pair_vectors[read_index];
  correction_vector.packed = pair_vectors[correction_index];
#pragma unroll
  for (int element = 0; element < kScalarsPerVector; ++element) {
    const float value =
        static_cast<float>(read_vector.values[element]) -
        gate_value * static_cast<float>(correction_vector.values[element]);
    output_vector.values[element] = static_cast<scalar_t>(value);
  }
  output_vectors[output_index] = output_vector.packed;
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaGateBwdVectorKernel(
    const scalar_t* output_grad,
    const scalar_t* correction,
    const scalar_t* gate,
    scalar_t* correction_grad,
    scalar_t* gate_grad,
    int64_t rows,
    int vectors_per_row) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }

  const float gate_value = Sigmoid(LoadFloat(gate + row));
  const uint4* output_grad_vectors =
      reinterpret_cast<const uint4*>(output_grad);
  const uint4* correction_vectors =
      reinterpret_cast<const uint4*>(correction);
  uint4* correction_grad_vectors =
      reinterpret_cast<uint4*>(correction_grad);
  float gate_sum = 0.0f;
  for (int vector_column = sublane; vector_column < vectors_per_row;
       vector_column += kSubwarpSize) {
    const int64_t vector_index = row * vectors_per_row + vector_column;
    GateVector<scalar_t> output_grad_vector;
    GateVector<scalar_t> correction_vector;
    GateVector<scalar_t> correction_grad_vector;
    output_grad_vector.packed = output_grad_vectors[vector_index];
    correction_vector.packed = correction_vectors[vector_index];
#pragma unroll
    for (int element = 0; element < kScalarsPerVector; ++element) {
      const float grad =
          static_cast<float>(output_grad_vector.values[element]);
      correction_grad_vector.values[element] =
          static_cast<scalar_t>(-gate_value * grad);
      gate_sum -= static_cast<float>(correction_vector.values[element]) * grad;
    }
    correction_grad_vectors[vector_index] = correction_grad_vector.packed;
  }
  for (int offset = kSubwarpSize / 2; offset > 0; offset /= 2) {
    gate_sum += __shfl_down_sync(
        active_mask, gate_sum, offset, kSubwarpSize);
  }
  if (sublane == 0) {
    gate_grad[row] = static_cast<scalar_t>(
        gate_sum * gate_value * (1.0f - gate_value));
  }
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaPairGateBwdVectorKernel(
    const scalar_t* output_grad,
    const scalar_t* pair_output,
    const scalar_t* gate,
    scalar_t* pair_grad,
    scalar_t* gate_grad,
    int64_t rows,
    int groups,
    int vectors_per_group) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }

  const float gate_value = Sigmoid(LoadFloat(gate + row));
  const int group = static_cast<int>(row % groups);
  const int64_t logical_pair = row / groups;
  const int64_t vectors_per_value = int64_t(groups) * vectors_per_group;
  const int64_t pair_base = logical_pair * 2 * vectors_per_value;
  const int64_t read_base = pair_base + int64_t(group) * vectors_per_group;
  const int64_t correction_base = read_base + vectors_per_value;
  const int64_t output_grad_base = row * vectors_per_group;
  const uint4* output_grad_vectors =
      reinterpret_cast<const uint4*>(output_grad);
  const uint4* pair_output_vectors =
      reinterpret_cast<const uint4*>(pair_output);
  uint4* pair_grad_vectors = reinterpret_cast<uint4*>(pair_grad);
  float gate_sum = 0.0f;
  for (int vector_column = sublane;
       vector_column < vectors_per_group;
       vector_column += kSubwarpSize) {
    GateVector<scalar_t> output_grad_vector;
    GateVector<scalar_t> correction_vector;
    GateVector<scalar_t> read_grad_vector;
    GateVector<scalar_t> correction_grad_vector;
    output_grad_vector.packed =
        output_grad_vectors[output_grad_base + vector_column];
    correction_vector.packed =
        pair_output_vectors[correction_base + vector_column];
#pragma unroll
    for (int element = 0; element < kScalarsPerVector; ++element) {
      const float grad =
          static_cast<float>(output_grad_vector.values[element]);
      read_grad_vector.values[element] = static_cast<scalar_t>(grad);
      correction_grad_vector.values[element] =
          static_cast<scalar_t>(-gate_value * grad);
      gate_sum -=
          static_cast<float>(correction_vector.values[element]) * grad;
    }
    pair_grad_vectors[read_base + vector_column] =
        read_grad_vector.packed;
    pair_grad_vectors[correction_base + vector_column] =
        correction_grad_vector.packed;
  }
  for (int offset = kSubwarpSize / 2; offset > 0; offset /= 2) {
    gate_sum += __shfl_down_sync(
        active_mask, gate_sum, offset, kSubwarpSize);
  }
  if (sublane == 0) {
    gate_grad[row] = static_cast<scalar_t>(
        gate_sum * gate_value * (1.0f - gate_value));
  }
}

template <typename scalar_t, int kRowsPerWarp>
__global__ void FlashSoftDeltaPairGateBwdStateVectorKernel(
    const scalar_t* output_grad,
    const float* pair_gate_state,
    const scalar_t* gate,
    scalar_t* pair_grad,
    scalar_t* gate_grad,
    int64_t rows,
    int groups,
    int vectors_per_group) {
  constexpr int kWarpSize = 32;
  constexpr int kSubwarpSize = kWarpSize / kRowsPerWarp;
  constexpr int kScalarsPerVector = sizeof(uint4) / sizeof(scalar_t);
  constexpr int kFloatsPerVector = sizeof(uint4) / sizeof(float);
  constexpr int kStateVectorsPerGradVector =
      kScalarsPerVector / kFloatsPerVector;
  static_assert(
      kScalarsPerVector % kFloatsPerVector == 0,
      "gate state vector widths must divide evenly");
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int subwarp = lane / kSubwarpSize;
  const int sublane = lane % kSubwarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row =
      (int64_t(blockIdx.x) * warps_per_block + warp) * kRowsPerWarp +
      subwarp;
  const bool active = row < rows;
  const unsigned active_mask = __ballot_sync(0xffffffff, active);
  if (!active) {
    return;
  }

  const float gate_value = Sigmoid(LoadFloat(gate + row));
  const int group = static_cast<int>(row % groups);
  const int64_t logical_pair = row / groups;
  const int64_t vectors_per_value = int64_t(groups) * vectors_per_group;
  const int64_t pair_base = logical_pair * 2 * vectors_per_value;
  const int64_t read_base = pair_base + int64_t(group) * vectors_per_group;
  const int64_t correction_base = read_base + vectors_per_value;
  const int64_t output_grad_base = row * vectors_per_group;
  const int64_t state_scalars_per_value =
      vectors_per_value * kScalarsPerVector;
  const int64_t state_pair_base =
      logical_pair * 2 * state_scalars_per_value;
  const int64_t state_correction_base =
      state_pair_base + int64_t(group) * vectors_per_group *
          kScalarsPerVector + state_scalars_per_value;
  const int64_t state_vector_base =
      state_correction_base / kFloatsPerVector;
  const uint4* output_grad_vectors =
      reinterpret_cast<const uint4*>(output_grad);
  const uint4* pair_gate_state_vectors =
      reinterpret_cast<const uint4*>(pair_gate_state);
  uint4* pair_grad_vectors = reinterpret_cast<uint4*>(pair_grad);
  float gate_sum = 0.0f;
  for (int vector_column = sublane;
       vector_column < vectors_per_group;
       vector_column += kSubwarpSize) {
    GateVector<scalar_t> output_grad_vector;
    GateVector<scalar_t> read_grad_vector;
    GateVector<scalar_t> correction_grad_vector;
    FloatGateVector correction_state_vectors[
        kStateVectorsPerGradVector];
    output_grad_vector.packed =
        output_grad_vectors[output_grad_base + vector_column];
#pragma unroll
    for (int state_vector = 0;
         state_vector < kStateVectorsPerGradVector;
         ++state_vector) {
      correction_state_vectors[state_vector].packed =
          pair_gate_state_vectors[
              state_vector_base +
              int64_t(vector_column) * kStateVectorsPerGradVector +
              state_vector];
    }
#pragma unroll
    for (int element = 0; element < kScalarsPerVector; ++element) {
      const float grad =
          static_cast<float>(output_grad_vector.values[element]);
      const float correction_value = RoundStateToScalar<scalar_t>(
          correction_state_vectors[element / kFloatsPerVector]
              .values[element % kFloatsPerVector]);
      read_grad_vector.values[element] = static_cast<scalar_t>(grad);
      correction_grad_vector.values[element] =
          static_cast<scalar_t>(-gate_value * grad);
      gate_sum -= correction_value * grad;
    }
    pair_grad_vectors[read_base + vector_column] =
        read_grad_vector.packed;
    pair_grad_vectors[correction_base + vector_column] =
        correction_grad_vector.packed;
  }
  for (int offset = kSubwarpSize / 2; offset > 0; offset /= 2) {
    gate_sum += __shfl_down_sync(
        active_mask, gate_sum, offset, kSubwarpSize);
  }
  if (sublane == 0) {
    gate_grad[row] = static_cast<scalar_t>(
        gate_sum * gate_value * (1.0f - gate_value));
  }
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchGateFwdRows(
    const torch::Tensor& read,
    const torch::Tensor& correction,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  const int64_t rows = gate.numel();
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  FlashSoftDeltaGateFwdKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, stream>>>(
      read.data_ptr<scalar_t>(), correction.data_ptr<scalar_t>(),
      gate.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), rows,
      read.size(4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
constexpr int GateScalarsPerVector() {
  return sizeof(uint4) / sizeof(scalar_t);
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchGateFwdVectorRows(
    const torch::Tensor& read,
    const torch::Tensor& correction,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int64_t rows = gate.numel();
  const int vectors_per_row =
      read.size(4) / GateScalarsPerVector<scalar_t>();
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  FlashSoftDeltaGateFwdVectorKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          read.data_ptr<scalar_t>(), correction.data_ptr<scalar_t>(),
          gate.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), rows,
          vectors_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void LaunchGateFwd(
    const torch::Tensor& read,
    const torch::Tensor& correction,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  if (read.size(4) % GateScalarsPerVector<scalar_t>() == 0) {
    const int vectors_per_row =
        read.size(4) / GateScalarsPerVector<scalar_t>();
    if (vectors_per_row <= 2) {
      LaunchGateFwdVectorRows<scalar_t, 16>(
          read, correction, gate, output);
    } else if (vectors_per_row <= 4) {
      LaunchGateFwdVectorRows<scalar_t, 8>(
          read, correction, gate, output);
    } else {
      LaunchGateFwdVectorRows<scalar_t, 4>(
          read, correction, gate, output);
    }
    return;
  }
  if (read.size(4) <= 8) {
    LaunchGateFwdRows<scalar_t, 4>(read, correction, gate, output);
  } else if (read.size(4) <= 16) {
    LaunchGateFwdRows<scalar_t, 2>(read, correction, gate, output);
  } else {
    LaunchGateFwdRows<scalar_t, 1>(read, correction, gate, output);
  }
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchPairGateFwdRows(
    const torch::Tensor& pair_output,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int64_t rows = gate.numel();
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  FlashSoftDeltaPairGateFwdKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          pair_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(), rows,
          static_cast<int>(gate.size(3)),
          static_cast<int>(output.size(4)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchPairGateFwdVectorRows(
    const torch::Tensor& pair_output,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int64_t rows = gate.numel();
  const int vectors_per_group =
      output.size(4) / GateScalarsPerVector<scalar_t>();
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  FlashSoftDeltaPairGateFwdVectorKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          pair_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(), rows,
          static_cast<int>(gate.size(3)), vectors_per_group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int kVectorsPerGroup>
void LaunchPairGateFwdFourGroupVectorRows(
    const torch::Tensor& pair_output,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kPairsPerWarp = 32 / (4 * kVectorsPerGroup);
  constexpr int kPairsPerBlock = kWarpsPerBlock * kPairsPerWarp;
  const int64_t pairs = gate.numel() / 4;
  const int blocks = static_cast<int>(
      (pairs + kPairsPerBlock - 1) / kPairsPerBlock);
  FlashSoftDeltaPairGateFwdFourGroupVectorKernel<
      scalar_t, kVectorsPerGroup>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          pair_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(), pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void LaunchPairGateFwd(
    const torch::Tensor& pair_output,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  const int group_dim = static_cast<int>(output.size(4));
  if (group_dim % GateScalarsPerVector<scalar_t>() == 0) {
    const int vectors_per_group =
        group_dim / GateScalarsPerVector<scalar_t>();
    if (gate.size(3) == 4 && vectors_per_group == 4) {
      LaunchPairGateFwdFourGroupVectorRows<scalar_t, 4>(
          pair_output, gate, output);
      return;
    }
    if (gate.size(3) == 4 && vectors_per_group == 8) {
      LaunchPairGateFwdFourGroupVectorRows<scalar_t, 8>(
          pair_output, gate, output);
      return;
    }
    if (vectors_per_group <= 2) {
      LaunchPairGateFwdVectorRows<scalar_t, 16>(
          pair_output, gate, output);
    } else if (vectors_per_group <= 4) {
      LaunchPairGateFwdVectorRows<scalar_t, 8>(
          pair_output, gate, output);
    } else {
      LaunchPairGateFwdVectorRows<scalar_t, 4>(
          pair_output, gate, output);
    }
    return;
  }
  if (group_dim <= 8) {
    LaunchPairGateFwdRows<scalar_t, 4>(pair_output, gate, output);
  } else if (group_dim <= 16) {
    LaunchPairGateFwdRows<scalar_t, 2>(pair_output, gate, output);
  } else {
    LaunchPairGateFwdRows<scalar_t, 1>(pair_output, gate, output);
  }
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchGateBwdRows(
    const torch::Tensor& output_grad,
    const torch::Tensor& correction,
    const torch::Tensor& gate,
    torch::Tensor& correction_grad,
    torch::Tensor& gate_grad) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  const int64_t rows = gate.numel();
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  FlashSoftDeltaGateBwdKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, stream>>>(
      output_grad.data_ptr<scalar_t>(), correction.data_ptr<scalar_t>(),
      gate.data_ptr<scalar_t>(), correction_grad.data_ptr<scalar_t>(),
      gate_grad.data_ptr<scalar_t>(), rows, correction.size(4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchGateBwdVectorRows(
    const torch::Tensor& output_grad,
    const torch::Tensor& correction,
    const torch::Tensor& gate,
    torch::Tensor& correction_grad,
    torch::Tensor& gate_grad) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int64_t rows = gate.numel();
  const int vectors_per_row =
      correction.size(4) / GateScalarsPerVector<scalar_t>();
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  FlashSoftDeltaGateBwdVectorKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          output_grad.data_ptr<scalar_t>(), correction.data_ptr<scalar_t>(),
          gate.data_ptr<scalar_t>(), correction_grad.data_ptr<scalar_t>(),
          gate_grad.data_ptr<scalar_t>(), rows, vectors_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void LaunchGateBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& correction,
    const torch::Tensor& gate,
    torch::Tensor& correction_grad,
    torch::Tensor& gate_grad) {
  if (correction.size(4) % GateScalarsPerVector<scalar_t>() == 0) {
    const int vectors_per_row =
        correction.size(4) / GateScalarsPerVector<scalar_t>();
    if (vectors_per_row <= 2) {
      LaunchGateBwdVectorRows<scalar_t, 16>(
          output_grad, correction, gate, correction_grad, gate_grad);
    } else if (vectors_per_row <= 4) {
      LaunchGateBwdVectorRows<scalar_t, 8>(
          output_grad, correction, gate, correction_grad, gate_grad);
    } else {
      LaunchGateBwdVectorRows<scalar_t, 4>(
          output_grad, correction, gate, correction_grad, gate_grad);
    }
    return;
  }
  if (correction.size(4) <= 8) {
    LaunchGateBwdRows<scalar_t, 4>(
        output_grad, correction, gate, correction_grad, gate_grad);
  } else if (correction.size(4) <= 16) {
    LaunchGateBwdRows<scalar_t, 2>(
        output_grad, correction, gate, correction_grad, gate_grad);
  } else {
    LaunchGateBwdRows<scalar_t, 1>(
        output_grad, correction, gate, correction_grad, gate_grad);
  }
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchPairGateBwdRows(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_output,
    const torch::Tensor& gate,
    torch::Tensor& pair_grad,
    torch::Tensor& gate_grad) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int64_t rows = gate.numel();
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  FlashSoftDeltaPairGateBwdKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          output_grad.data_ptr<scalar_t>(),
          pair_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
          pair_grad.data_ptr<scalar_t>(), gate_grad.data_ptr<scalar_t>(),
          rows, static_cast<int>(gate.size(3)),
          static_cast<int>(output_grad.size(4)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchPairGateBwdVectorRows(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_output,
    const torch::Tensor& gate,
    torch::Tensor& pair_grad,
    torch::Tensor& gate_grad) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int64_t rows = gate.numel();
  const int vectors_per_group =
      output_grad.size(4) / GateScalarsPerVector<scalar_t>();
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  FlashSoftDeltaPairGateBwdVectorKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          output_grad.data_ptr<scalar_t>(),
          pair_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
          pair_grad.data_ptr<scalar_t>(), gate_grad.data_ptr<scalar_t>(),
          rows, static_cast<int>(gate.size(3)), vectors_per_group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void LaunchPairGateBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_output,
    const torch::Tensor& gate,
    torch::Tensor& pair_grad,
    torch::Tensor& gate_grad) {
  const int group_dim = static_cast<int>(output_grad.size(4));
  if (group_dim % GateScalarsPerVector<scalar_t>() == 0) {
    const int vectors_per_group =
        group_dim / GateScalarsPerVector<scalar_t>();
    if (vectors_per_group <= 2) {
      LaunchPairGateBwdVectorRows<scalar_t, 16>(
          output_grad, pair_output, gate, pair_grad, gate_grad);
    } else if (vectors_per_group <= 4) {
      LaunchPairGateBwdVectorRows<scalar_t, 8>(
          output_grad, pair_output, gate, pair_grad, gate_grad);
    } else {
      LaunchPairGateBwdVectorRows<scalar_t, 4>(
          output_grad, pair_output, gate, pair_grad, gate_grad);
    }
    return;
  }
  if (group_dim <= 8) {
    LaunchPairGateBwdRows<scalar_t, 4>(
        output_grad, pair_output, gate, pair_grad, gate_grad);
  } else if (group_dim <= 16) {
    LaunchPairGateBwdRows<scalar_t, 2>(
        output_grad, pair_output, gate, pair_grad, gate_grad);
  } else {
    LaunchPairGateBwdRows<scalar_t, 1>(
        output_grad, pair_output, gate, pair_grad, gate_grad);
  }
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchPairGateBwdStateRows(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_gate_state,
    const torch::Tensor& gate,
    torch::Tensor& pair_grad,
    torch::Tensor& gate_grad) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int64_t rows = gate.numel();
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  FlashSoftDeltaPairGateBwdStateKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          output_grad.data_ptr<scalar_t>(),
          pair_gate_state.data_ptr<float>(), gate.data_ptr<scalar_t>(),
          pair_grad.data_ptr<scalar_t>(), gate_grad.data_ptr<scalar_t>(),
          rows, static_cast<int>(gate.size(3)),
          static_cast<int>(output_grad.size(4)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int kRowsPerWarp>
void LaunchPairGateBwdStateVectorRows(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_gate_state,
    const torch::Tensor& gate,
    torch::Tensor& pair_grad,
    torch::Tensor& gate_grad) {
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  const int64_t rows = gate.numel();
  const int vectors_per_group =
      output_grad.size(4) / GateScalarsPerVector<scalar_t>();
  const int blocks = static_cast<int>(
      (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  FlashSoftDeltaPairGateBwdStateVectorKernel<scalar_t, kRowsPerWarp>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          output_grad.data_ptr<scalar_t>(),
          pair_gate_state.data_ptr<float>(), gate.data_ptr<scalar_t>(),
          pair_grad.data_ptr<scalar_t>(), gate_grad.data_ptr<scalar_t>(),
          rows, static_cast<int>(gate.size(3)), vectors_per_group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void LaunchPairGateBwdState(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_gate_state,
    const torch::Tensor& gate,
    torch::Tensor& pair_grad,
    torch::Tensor& gate_grad) {
  const int group_dim = static_cast<int>(output_grad.size(4));
  if (group_dim % GateScalarsPerVector<scalar_t>() == 0) {
    const int vectors_per_group =
        group_dim / GateScalarsPerVector<scalar_t>();
    if (vectors_per_group <= 2) {
      LaunchPairGateBwdStateVectorRows<scalar_t, 16>(
          output_grad, pair_gate_state, gate, pair_grad, gate_grad);
    } else if (vectors_per_group <= 4) {
      LaunchPairGateBwdStateVectorRows<scalar_t, 8>(
          output_grad, pair_gate_state, gate, pair_grad, gate_grad);
    } else {
      LaunchPairGateBwdStateVectorRows<scalar_t, 4>(
          output_grad, pair_gate_state, gate, pair_grad, gate_grad);
    }
    return;
  }
  if (group_dim <= 8) {
    LaunchPairGateBwdStateRows<scalar_t, 4>(
        output_grad, pair_gate_state, gate, pair_grad, gate_grad);
  } else if (group_dim <= 16) {
    LaunchPairGateBwdStateRows<scalar_t, 2>(
        output_grad, pair_gate_state, gate, pair_grad, gate_grad);
  } else {
    LaunchPairGateBwdStateRows<scalar_t, 1>(
        output_grad, pair_gate_state, gate, pair_grad, gate_grad);
  }
}

}  // namespace

torch::Tensor FlashSoftDeltaGateFwd(
    const torch::Tensor& read,
    const torch::Tensor& correction,
    const torch::Tensor& gate) {
  CheckGateInputs(read, correction, gate);
  c10::cuda::CUDAGuard guard(read.device());
  torch::Tensor output = torch::empty_like(read);
  if (read.scalar_type() == at::kHalf) {
    LaunchGateFwd<at::Half>(read, correction, gate, output);
  } else {
    LaunchGateFwd<at::BFloat16>(read, correction, gate, output);
  }
  return output;
}

torch::Tensor FlashSoftDeltaPairGateFwd(
    const torch::Tensor& pair_output,
    const torch::Tensor& gate) {
  CheckPairGateInputs(pair_output, gate);
  c10::cuda::CUDAGuard guard(pair_output.device());
  const int64_t group_dim = pair_output.size(3) / gate.size(3);
  torch::Tensor output = torch::empty(
      {pair_output.size(0), pair_output.size(1), gate.size(2),
       gate.size(3), group_dim},
      pair_output.options().memory_format(at::MemoryFormat::Contiguous));
  if (pair_output.scalar_type() == at::kHalf) {
    LaunchPairGateFwd<at::Half>(pair_output, gate, output);
  } else {
    LaunchPairGateFwd<at::BFloat16>(pair_output, gate, output);
  }
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FlashSoftDeltaGateBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& correction,
    const torch::Tensor& gate) {
  CheckGateInputs(output_grad, correction, gate);
  c10::cuda::CUDAGuard guard(output_grad.device());
  torch::Tensor read_grad = output_grad;
  torch::Tensor correction_grad = torch::empty_like(correction);
  torch::Tensor gate_grad = torch::empty_like(gate);
  if (output_grad.scalar_type() == at::kHalf) {
    LaunchGateBwd<at::Half>(
        output_grad, correction, gate, correction_grad, gate_grad);
  } else {
    LaunchGateBwd<at::BFloat16>(
        output_grad, correction, gate, correction_grad, gate_grad);
  }
  return {read_grad, correction_grad, gate_grad};
}

std::tuple<torch::Tensor, torch::Tensor>
FlashSoftDeltaPairGateBwd(
    const torch::Tensor& output_grad,
    const torch::Tensor& pair_gate_state,
    const torch::Tensor& gate) {
  CheckPairGateBackwardInputs(output_grad, pair_gate_state, gate);
  c10::cuda::CUDAGuard guard(output_grad.device());
  const bool use_fp32_state =
      pair_gate_state.scalar_type() == at::kFloat;
  torch::Tensor pair_grad = use_fp32_state
      ? torch::empty(pair_gate_state.sizes(), output_grad.options())
      : torch::empty_like(pair_gate_state);
  torch::Tensor gate_grad = torch::empty_like(gate);
  if (output_grad.scalar_type() == at::kHalf) {
    if (use_fp32_state) {
      LaunchPairGateBwdState<at::Half>(
          output_grad, pair_gate_state, gate, pair_grad, gate_grad);
    } else {
      LaunchPairGateBwd<at::Half>(
          output_grad, pair_gate_state, gate, pair_grad, gate_grad);
    }
  } else {
    if (use_fp32_state) {
      LaunchPairGateBwdState<at::BFloat16>(
          output_grad, pair_gate_state, gate, pair_grad, gate_grad);
    } else {
      LaunchPairGateBwd<at::BFloat16>(
          output_grad, pair_gate_state, gate, pair_grad, gate_grad);
    }
  }
  return {std::move(pair_grad), std::move(gate_grad)};
}


}  // namespace ops
}  // namespace xattn

#include "attention/precision/preprocess.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <cmath>
namespace xattn::ops::attention::precision {
namespace {
constexpr int kCenterRows = 32;     // rows per CTA in the centering passes
constexpr int kCenterThreads = 128; // each thread owns 8-feature groups
template <typename E> struct Pack8 { E v[8]; };

// (batch, row chunk) of a centering CTA. The grid is (chunks, B, groups), or
// (chunks * B, 1, groups) once B exceeds grid.y's limit; both decode alike
// when B == 1.
__device__ __forceinline__ int2 CenterBlock(int n) {
  int b = blockIdx.y, chunk = blockIdx.x;
  if (gridDim.y == 1) { const int chunks = (n + kCenterRows - 1) / kCenterRows; b = chunk / chunks; chunk %= chunks; }
  return make_int2(b, chunk);
}

// Pass 1: a feature keeps its first-row shift only if every residual
// E(x - shift) reconstructs x exactly in FP64; otherwise it is flagged.
// Rows are split into chunks; each thread checks 8 contiguous features with
// one 16-byte load per row.
template <typename E>
__global__ void __launch_bounds__(kCenterThreads) CheckCenterRows(const E* x, int* flags, int n, int width) {
  const int2 block = CenterBlock(n);
  const int b = block.x, begin = block.y * kCenterRows, end = min(n, begin + kCenterRows);
  const E* base = x + int64_t(b) * n * width;
  // grid.z spreads feature groups so wide heads do not serialize per thread.
  const int group = blockIdx.z * kCenterThreads + threadIdx.x;
  if (group * 8 < width) {
    const int f = group * 8;
    const Pack8<E> first = *reinterpret_cast<const Pack8<E>*>(base + f);
    float shift[8];
    unsigned bad = 0;
    #pragma unroll
    for (int c = 0; c < 8; ++c) { shift[c] = float(first.v[c]); if (!isfinite(shift[c])) bad |= 1u << c; }
    #pragma unroll 8
    for (int i = begin; i < end; ++i) {
      const Pack8<E> row = *reinterpret_cast<const Pack8<E>*>(base + int64_t(i) * width + f);
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float original = float(row.v[c]);
        const E residual = E(original - shift[c]);
        // Compare in FP64, so adding a large shift cannot hide a lost small value.
        if (double(float(residual)) + double(shift[c]) != double(original)) bad |= 1u << c;
      }
    }
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      // Skip the atomic when another row chunk already set the flag.
      int* flag = flags + int64_t(b) * width + f + c;
      if ((bad & (1u << c)) && !*reinterpret_cast<volatile int*>(flag)) atomicExch(flag, 1);
    }
  }
}

// Pass 2: apply the selected shift (0 for flagged features) and record it.
template <typename E>
__global__ void __launch_bounds__(kCenterThreads) ApplyCenterRows(const E* x, const int* flags, E* y, float* shifts, int n, int width) {
  const int2 block = CenterBlock(n);
  const int b = block.x, begin = block.y * kCenterRows, end = min(n, begin + kCenterRows);
  const E* in = x + int64_t(b) * n * width;
  E* out = y + int64_t(b) * n * width;
  const int group = blockIdx.z * kCenterThreads + threadIdx.x;
  if (group * 8 < width) {
    const int f = group * 8;
    const Pack8<E> first = *reinterpret_cast<const Pack8<E>*>(in + f);
    float shift[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) shift[c] = flags[int64_t(b) * width + f + c] ? 0.f : float(first.v[c]);
    if (block.y == 0) {
      #pragma unroll
      for (int c = 0; c < 8; ++c) shifts[int64_t(b) * width + f + c] = shift[c];
    }
    #pragma unroll 8
    for (int i = begin; i < end; ++i) {
      const Pack8<E> row = *reinterpret_cast<const Pack8<E>*>(in + int64_t(i) * width + f);
      Pack8<E> result;
      #pragma unroll
      for (int c = 0; c < 8; ++c) result.v[c] = shift[c] == 0.f ? row.v[c] : E(float(row.v[c]) - shift[c]);
      *reinterpret_cast<Pack8<E>*>(out + int64_t(i) * width + f) = result;
    }
  }
}

// One warp per dO row (8 rows per CTA): divide by the even power of two of the
// row maximum when every component round-trips; multiplying by the exact
// reciprocal power of two equals the division bitwise.
template <typename E>
__global__ void ScaleKernel(const E* x, E* y, float* scales, int64_t rows, int n, int h, int d) {
  const int64_t row = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= rows) return;
  const int64_t offset = row * d;
  float maximum = 0.f;
  for (int c = lane; c < d; c += 32) maximum = fmaxf(maximum, fabsf(float(x[offset + c])));
  for (int delta = 16; delta; delta /= 2) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
  int exponent = maximum >= 1.f && isfinite(maximum) ? ilogbf(maximum) : 0;
  exponent = min(120, max(0, exponent)); exponent &= ~1;
  float scale = ldexpf(1.f, exponent);
  const float inverse = ldexpf(1.f, -exponent);
  bool invalid = false;
  for (int c = lane; c < d; c += 32) {
    const float original = float(x[offset + c]);
    E low = E(original * inverse);
    invalid |= double(float(low)) * double(scale) != double(original);
  }
  invalid = __any_sync(0xffffffff, invalid);
  if (invalid) scale = 1.f;
  for (int c = lane; c < d; c += 32) y[offset + c] = invalid ? x[offset + c] : E(float(x[offset + c]) * inverse);
  if (lane == 0) {
    const int b = row / (int64_t(n) * h), i = (row / h) % n, head = row % h;
    scales[(int64_t(b) * h + head) * n + i] = scale;
  }
}
void Validate(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim()==4 && x.is_contiguous(),"precision preprocessing requires contiguous CUDA [B,N,H,D]");
  TORCH_CHECK(x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16,"precision preprocessing requires FP16/BF16");
  TORCH_CHECK(x.size(2)>0 && x.size(3)>0,"head count and dimension must be positive");
}
}
std::tuple<torch::Tensor,torch::Tensor> Center(const torch::Tensor& input) {
  Validate(input);c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(input.size(3)%8==0,"precision centering requires D a multiple of 8");
  auto output=torch::empty_like(input);auto shifts=torch::empty({input.size(0),input.size(2),input.size(3)},input.options().dtype(at::kFloat));
  if (!input.numel()) {shifts.zero_();return {output,shifts};}
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr())%16==0,"precision centering requires 16-byte aligned input");
  auto stream=at::cuda::getCurrentCUDAStream(input.get_device());
  const int n=input.size(1),width=input.size(2)*input.size(3);
  auto flags=torch::zeros({input.size(0),width},input.options().dtype(at::kInt));
  // B rides grid.y up to its 65535 limit; larger batches fold into x.
  const int64_t chunks=(n+kCenterRows-1)/kCenterRows,batch=input.size(0);const bool fold=batch>65535;
  TORCH_CHECK(!fold || chunks*batch<=INT_MAX,"precision centering grid exceeds CUDA limits");
  dim3 grid(unsigned(fold ? chunks*batch : chunks),unsigned(fold ? 1 : batch),(width/8+kCenterThreads-1)/kCenterThreads);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(),"precision_center",[&]{
    CheckCenterRows<scalar_t><<<grid,kCenterThreads,0,stream>>>(input.data_ptr<scalar_t>(),flags.data_ptr<int>(),n,width);
    ApplyCenterRows<scalar_t><<<grid,kCenterThreads,0,stream>>>(input.data_ptr<scalar_t>(),flags.data_ptr<int>(),output.data_ptr<scalar_t>(),shifts.data_ptr<float>(),n,width);
  });C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,shifts};
}
std::tuple<torch::Tensor,torch::Tensor> ScaleDO(const torch::Tensor& input) {
  Validate(input);c10::cuda::CUDAGuard guard(input.device());
  auto output=torch::empty_like(input);auto scales=torch::empty({input.size(0),input.size(2),input.size(1)},input.options().dtype(at::kFloat));
  if (!input.numel()) return {output,scales};
  auto stream=at::cuda::getCurrentCUDAStream(input.get_device());
  const int64_t rows=input.numel()/input.size(3);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(),"precision_scale_do",[&]{
    ScaleKernel<scalar_t><<<(rows+7)/8,256,0,stream>>>(input.data_ptr<scalar_t>(),output.data_ptr<scalar_t>(),scales.data_ptr<float>(),rows,input.size(1),input.size(2),input.size(3));
  });C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,scales};
}
}  // namespace xattn::ops::attention::precision

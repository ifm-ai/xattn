#include "attention/precision/anchor_repair.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace xattn::ops::attention::precision {
namespace {
template <typename Element>
__global__ void AnchorRepairKernel(
    const Element* __restrict__ q, const Element* __restrict__ k,
    const double* __restrict__ rho, const int32_t* __restrict__ anchor,
    float* __restrict__ dq, float* __restrict__ dk,
    int64_t count, int nq, int nk, int hq, int hk, int d, float scale) {
  const int64_t x = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= count) return;
  const int c = x % d;
  const int h = (x / d) % hq;
  const int i = (x / (int64_t(d) * hq)) % nq;
  const int b = x / (int64_t(d) * hq * nq);
  const int64_t row = (int64_t(b) * hq + h) * nq + i;
  const int a = anchor[row];
  // Empty rows must remain exactly zero; never gather K[-1]. A producer bug
  // with another invalid index cannot cause an out-of-bounds memory access.
  if (a < 0 || a >= nk) return;
  const int kh = h / (hq / hk);
  const int64_t key = ((int64_t(b) * nk + a) * hk + kh) * d + c;
  const float coefficient = scale * static_cast<float>(rho[row]);
  dq[x] -= coefficient * static_cast<float>(k[key]);
  const float correction = coefficient * static_cast<float>(q[x]);
  if (correction != 0.f) atomicAdd(dk + key, -correction);
}
void CheckDense(const torch::Tensor& t, const torch::Tensor& q,
                at::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == q.device(), name, " must be on Q's CUDA device");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
}  // namespace
void AnchorRepair(const torch::Tensor& q, const torch::Tensor& k,
                  const torch::Tensor& rho, const torch::Tensor& anchor,
                  torch::Tensor dq, torch::Tensor dk, double scale) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && k.dim() == 4, "Q/K must be rank-four CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf,
              "Q/K must be BF16 or FP16");
  TORCH_CHECK(q.size(2) > 0 && k.size(2) > 0 && q.size(2) % k.size(2) == 0,
              "Q heads must be divisible by KV heads");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3), "Q/K batch and D must match");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  CheckDense(q, q, q.scalar_type(), "q"); CheckDense(k, q, q.scalar_type(), "k");
  CheckDense(dq, q, at::kFloat, "dq"); CheckDense(dk, q, at::kFloat, "dk");
  CheckDense(rho, q, at::kDouble, "rho"); CheckDense(anchor, q, at::kInt, "anchor");
  TORCH_CHECK(dq.sizes() == q.sizes() && dk.sizes() == k.sizes(), "accumulator shapes must match Q/K");
  TORCH_CHECK(rho.dim() == 3 && rho.size(0) == q.size(0) && rho.size(1) == q.size(2) && rho.size(2) == q.size(1), "rho must have shape [B,Hq,Nq]");
  TORCH_CHECK(anchor.sizes() == rho.sizes(), "anchor and rho shapes must match");
  if (!q.numel()) return;
  c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  constexpr int threads = 256;
  const int blocks = (q.numel() + threads - 1) / threads;
  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "attention_anchor_repair", [&] {
    AnchorRepairKernel<scalar_t><<<blocks, threads, 0, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), rho.data_ptr<double>(),
        anchor.data_ptr<int32_t>(), dq.data_ptr<float>(), dk.data_ptr<float>(),
        q.numel(), q.size(1), k.size(1), q.size(2), k.size(2), q.size(3), float(scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace xattn::ops::attention::precision

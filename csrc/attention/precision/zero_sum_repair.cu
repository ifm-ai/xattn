#include "attention/precision/zero_sum_repair.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <cmath>
#include <type_traits>

namespace xattn::ops::attention::precision {
namespace {
constexpr int kRows = 64;     // output rows (dQ) or keys (dK) per CTA
constexpr int kChunk = 128;   // staged rows per shared-memory chunk
constexpr int kThreads = 256;
struct MaxOp { __host__ __device__ __forceinline__ int64_t operator()(int64_t a, int64_t b) const { return a > b ? a : b; } };
template <class T> __device__ __forceinline__ float F(T x) { return static_cast<float>(x); }
template <class E> __device__ __forceinline__ float4 Load4(const E* p) {
  // D is a multiple of 8, so 4-element groups are 8-byte aligned.
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const E* e = reinterpret_cast<const E*>(&raw);
  return make_float4(F(e[0]), F(e[1]), F(e[2]), F(e[3]));
}

__global__ void SegmentChangeKernel(const int64_t* segment, int* change, int nk, int64_t total) {
  const int64_t x = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= total) return;
  const int j = x % nk;
  change[x] = j > 0 && segment[x] != segment[x - 1];
}

// Candidate windows [lo, last], encoded so one flat max-scan never crosses a
// batch row: value = b * stride + (bound + 1), stride > nk.
__global__ void EncodeKernel(const float* first_plane, const float* last_plane, const int* changes,
                             int nq, int nk, int64_t head_stride_b, int window, int64_t stride, int64_t total,
                             int* lo_out, int* last_out, int64_t* enc_lo, int64_t* enc_last) {
  const int64_t x = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= total) return;
  const int b = x / nq, i = x % nq;
  const int f = int(first_plane[b * head_stride_b + i]), l = int(last_plane[b * head_stride_b + i]);
  const int lo = max(f, l - window + 1);
  bool cand = l >= f && l < nk && f >= 0;
  // The last admitted key shares the row's segment; the window is admitted
  // exactly when no segment boundary lies in (lo, last].
  if (cand && changes != nullptr) cand = changes[int64_t(b) * nk + l] == changes[int64_t(b) * nk + lo];
  lo_out[x] = lo; last_out[x] = l;
  enc_lo[x] = b * stride + (cand ? lo + 1 : 0);
  enc_last[x] = b * stride + (cand ? l + 1 : 0);
}

// Inclusive candidate maxima are monotone; a candidate is eligible when its
// bounds are >= the maxima of all earlier candidates, so for eligible rows the
// maxima equal their own bounds and binary searches on them return exactly the
// eligible rows whose windows hold a key.
__global__ void CoefficientKernel(const int* lo_in, const int* last_in, const int* changes,
                                  const int64_t* max_lo, const int64_t* max_last,
                                  const double* rho, const float* log_z,
                                  const double* variance, const float* delta,
                                  int* mono_lo, int* mono_last, double* rho_anchor, float* coefficient,
                                  int nq, int nk, int hq, int64_t stride, int64_t total) {
  const int64_t x = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= total) return;
  const int b = x / nq, i = x % nq;
  const int lo = lo_in[x], l = last_in[x];
  bool cand = l >= lo && l < nk && lo >= 0;
  if (cand && changes != nullptr) cand = changes[int64_t(b) * nk + l] == changes[int64_t(b) * nk + lo];
  const int plo = i ? int(max_lo[x - 1] - b * stride) - 1 : -1;
  const int pl = i ? int(max_last[x - 1] - b * stride) - 1 : -1;
  const bool elig = cand && lo >= plo && l >= pl;
  mono_lo[x] = int(max_lo[x] - b * stride) - 1;
  mono_last[x] = int(max_last[x] - b * stride) - 1;
  for (int h = 0; h < hq; ++h) {
    const int64_t row = (int64_t(b) * hq + h) * nq + i;
    const double r = rho[row];
    double u = 0.0;
    float c = 0.f;
    if (elig && r != 0.0 && isfinite(r)) {
      // Anchor share: dS_a^2 / sum dS^2 (rounding variance) when the operand
      // statistics are available, else the anchor probability.
      double share = fmin(fmax(exp(-double(log_z[row])), 0.0), 1.0);
      if (variance != nullptr) {
        const double v = variance[row], da = share * double(delta[row]);
        if (v > 0.0 && isfinite(v) && isfinite(da)) share = fmin(da * da / v, 1.0);
      }
      const double n = double(l - lo + 1);
      c = float((r - r * share) / n);
      u = double(c) * n;  // the anchor share absorbs the FP32 rounding of c
    }
    rho_anchor[row] = r - u;
    coefficient[row] = c;
  }
}

// smem[(r + 1) * d + f] becomes the inclusive prefix of staged rows 0..r.
__device__ __forceinline__ void PrefixRows(float* smem, int count, int d) {
  const int t = threadIdx.x;
  const int parts = d <= kThreads / 2 ? 2 : 1;
  const int f = t % d, part = t / d;
  const int half = (count + 1) / 2;
  const bool active = t < d * parts;
  const int begin = parts == 2 ? part * half : 0, end = parts == 2 ? min(count, begin + half) : count;
  float run = 0.f;
  if (active) {
    for (int r = begin; r < end; ++r) { run += smem[(r + 1) * d + f]; smem[(r + 1) * d + f] = run; }
  }
  __syncthreads();
  if (parts == 2 && active && part == 1) {
    const float carry = smem[half * d + f];
    for (int r = begin; r < end; ++r) smem[(r + 1) * d + f] += carry;
  }
  __syncthreads();
}

// dQ: uniform share over each row's window plus the anchor share; also the
// anchor share of dK (aggregated over consecutive rows of a warp that share
// the anchor, e.g. attention sinks). CTA = (row block, KV head, batch);
// warp w owns rows w, w+8, ...; lanes own 4-feature vectors.
template <class E, int VPL>
__global__ void __launch_bounds__(kThreads) DqKernel(const E* q, const E* k, const double* rho_anchor,
    const float* anchor, const float* coefficient, const int* lo_in, const int* last_in,
    float* dq, float* dk, int nq, int nk, int hq, int hk, int d, float scale, bool anchor_scatter) {
  extern __shared__ float smem[];   // [(kChunk + 1) * d]
  __shared__ int s_lo[kRows], s_last[kRows], s_kmin, s_kmax;
  const int b = blockIdx.z, kh = blockIdx.y, i0 = blockIdx.x * kRows, t = threadIdx.x;
  const int g = hq / hk, width = hk * d, lane = t % 32, warp = t / 32, vecs = d / 4;
  if (t < kRows) {
    const int i = i0 + t;
    s_lo[t] = i < nq ? lo_in[int64_t(b) * nq + i] : 0;
    s_last[t] = i < nq ? last_in[int64_t(b) * nq + i] : -1;
  }
  if (t == 0) { s_kmin = INT_MAX; s_kmax = -1; }
  __syncthreads();
  if (t < kRows && i0 + t < nq && s_last[t] >= s_lo[t]) {
    bool any = false;
    for (int h = 0; h < g; ++h) any |= coefficient[(int64_t(b) * hq + kh * g + h) * nq + i0 + t] != 0.f;
    if (any) { atomicMin(&s_kmin, s_lo[t]); atomicMax(&s_kmax, s_last[t]); }
  }
  __syncthreads();
  float4 sum[kRows / 8][VPL];
  #pragma unroll
  for (int r = 0; r < kRows / 8; ++r) for (int s = 0; s < VPL; ++s) sum[r][s] = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int base = s_kmin; base <= s_kmax; base += kChunk) {
    const int count = min(kChunk, s_kmax - base + 1);
    #pragma unroll 4
    for (int x = t; x < count * vecs; x += kThreads) {
      const int r = x / vecs, v = x % vecs;
      *reinterpret_cast<float4*>(smem + (r + 1) * d + v * 4) =
          Load4(k + (int64_t(b) * nk + base + r) * width + kh * d + v * 4);
    }
    if (t < d) smem[t] = 0.f;
    __syncthreads();
    PrefixRows(smem, count, d);
    #pragma unroll
    for (int rr = 0; rr < kRows / 8; ++rr) {
      const int r = warp + rr * 8;
      const int lo = max(s_lo[r], base), hi = min(s_last[r], base + count - 1);
      if (hi >= lo) {
        #pragma unroll
        for (int s = 0; s < VPL; ++s) {
          const int v = lane + s * 32;
          if (v < vecs) {
            const float4 a = *reinterpret_cast<const float4*>(smem + (hi - base + 1) * d + v * 4);
            const float4 c = *reinterpret_cast<const float4*>(smem + (lo - base) * d + v * 4);
            sum[rr][s].x += a.x - c.x; sum[rr][s].y += a.y - c.y; sum[rr][s].z += a.z - c.z; sum[rr][s].w += a.w - c.w;
          }
        }
      }
    }
    __syncthreads();
  }
  for (int h = 0; h < g; ++h) {
    const int head = kh * g + h;
    int pending = -1;  // anchor key whose dK share is being aggregated
    float4 acc[VPL];
    #pragma unroll
    for (int s = 0; s < VPL; ++s) acc[s] = make_float4(0.f, 0.f, 0.f, 0.f);
    auto flush = [&]() {
      if (pending < 0) return;
      #pragma unroll
      for (int s = 0; s < VPL; ++s) {
        const int v = lane + s * 32;
        if (v < vecs) atomicAdd(reinterpret_cast<float4*>(dk + (int64_t(b) * nk + pending) * width + kh * d + v * 4), acc[s]);
        acc[s] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      pending = -1;
    };
    #pragma unroll
    for (int rr = 0; rr < kRows / 8; ++rr) {
      const int i = i0 + warp + rr * 8;
      if (i >= nq) continue;
      const int64_t row = (int64_t(b) * hq + head) * nq + i;
      const float cf = scale * coefficient[row];
      const int a = int(anchor[row]);
      const double ra = rho_anchor[row];
      // Without the scatter the whole anchor share (dQ and dK) is left to the
      // caller's ordered repair through rho_anchor.
      const bool use_anchor = anchor_scatter && a >= 0 && a < nk && ra != 0.0;
      const float ca = scale * float(ra);
      if (use_anchor && a != pending) flush();
      #pragma unroll
      for (int s = 0; s < VPL; ++s) {
        const int v = lane + s * 32;
        if (v >= vecs) continue;
        const int64_t x = ((int64_t(b) * nq + i) * hq + head) * d + v * 4;
        float4 out = *reinterpret_cast<float4*>(dq + x);
        float4 upd = make_float4(cf * sum[rr][s].x, cf * sum[rr][s].y, cf * sum[rr][s].z, cf * sum[rr][s].w);
        if (use_anchor) {
          const float4 ka = Load4(k + (int64_t(b) * nk + a) * width + kh * d + v * 4);
          upd.x += ca * ka.x; upd.y += ca * ka.y; upd.z += ca * ka.z; upd.w += ca * ka.w;
          const float4 qv = Load4(q + x);
          acc[s].x -= ca * qv.x; acc[s].y -= ca * qv.y; acc[s].z -= ca * qv.z; acc[s].w -= ca * qv.w;
        }
        out.x -= upd.x; out.y -= upd.y; out.z -= upd.z; out.w -= upd.w;
        *reinterpret_cast<float4*>(dq + x) = out;
      }
      if (use_anchor) pending = a;
    }
    flush();
  }
}

__device__ __forceinline__ int FirstAtLeast(const int* a, int n, int v) {
  int lo = 0, hi = n;
  while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] >= v) hi = mid; else lo = mid + 1; }
  return lo;
}
__device__ __forceinline__ int FirstGreater(const int* a, int n, int v) {
  int lo = 0, hi = n;
  while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] > v) hi = mid; else lo = mid + 1; }
  return lo;
}

// Rows whose windows can hold key j: [first i with mono_last >= j,
// first i with mono_lo > j).
__global__ void KeyRowsKernel(const int* mono_lo, const int* mono_last, int2* key_rows, int nq, int nk, int64_t total) {
  const int64_t x = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= total) return;
  const int b = x / nk, j = x % nk;
  key_rows[x] = make_int2(FirstAtLeast(mono_last + int64_t(b) * nq, nq, j), FirstGreater(mono_lo + int64_t(b) * nq, nq, j));
}

// dK uniform share: key j receives sum over eligible rows whose window holds
// j of c_i Q_i (summed over the KV head's Q heads). CTA = (key block, KV head,
// batch); rows are staged as c-weighted Q in shared-memory chunks.
template <class E, int VPL>
__global__ void __launch_bounds__(kThreads) DkKernel(const E* q, const float* coefficient, const int2* key_rows,
    float* dk, int nq, int nk, int hq, int hk, int d, float scale) {
  extern __shared__ float smem[];
  __shared__ int s_begin[kRows], s_end[kRows];
  const int b = blockIdx.z, kh = blockIdx.y, j0 = blockIdx.x * kRows, t = threadIdx.x;
  const int g = hq / hk, width = hk * d, lane = t % 32, warp = t / 32, vecs = d / 4;
  if (t < kRows) {
    const int j = j0 + t;
    const int2 range = j < nk ? key_rows[int64_t(b) * nk + j] : make_int2(nq, 0);
    s_begin[t] = range.x; s_end[t] = range.y;
  }
  __syncthreads();
  const int rmin = s_begin[0];
  const int rmax = s_end[min(kRows, nk - j0) - 1] - 1;
  float4 sum[kRows / 8][VPL];
  #pragma unroll
  for (int r = 0; r < kRows / 8; ++r) for (int s = 0; s < VPL; ++s) sum[r][s] = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int base = rmin; base <= rmax; base += kChunk) {
    const int count = min(kChunk, rmax - base + 1);
    #pragma unroll 4
    for (int x = t; x < count * vecs; x += kThreads) {
      const int r = x / vecs, v = x % vecs;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int h = 0; h < g; ++h) {
        const int head = kh * g + h;
        const float c = coefficient[(int64_t(b) * hq + head) * nq + base + r];
        if (c != 0.f) {
          const float4 qv = Load4(q + ((int64_t(b) * nq + base + r) * hq + head) * d + v * 4);
          acc.x += c * qv.x; acc.y += c * qv.y; acc.z += c * qv.z; acc.w += c * qv.w;
        }
      }
      *reinterpret_cast<float4*>(smem + (r + 1) * d + v * 4) = acc;
    }
    if (t < d) smem[t] = 0.f;
    __syncthreads();
    PrefixRows(smem, count, d);
    #pragma unroll
    for (int rr = 0; rr < kRows / 8; ++rr) {
      const int r = warp + rr * 8;
      const int lo = max(s_begin[r], base), hi = min(s_end[r] - 1, base + count - 1);
      if (hi >= lo) {
        #pragma unroll
        for (int s = 0; s < VPL; ++s) {
          const int v = lane + s * 32;
          if (v < vecs) {
            const float4 a = *reinterpret_cast<const float4*>(smem + (hi - base + 1) * d + v * 4);
            const float4 c = *reinterpret_cast<const float4*>(smem + (lo - base) * d + v * 4);
            sum[rr][s].x += a.x - c.x; sum[rr][s].y += a.y - c.y; sum[rr][s].z += a.z - c.z; sum[rr][s].w += a.w - c.w;
          }
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int rr = 0; rr < kRows / 8; ++rr) {
    const int j = j0 + warp + rr * 8;
    if (j >= nk) continue;
    #pragma unroll
    for (int s = 0; s < VPL; ++s) {
      const int v = lane + s * 32;
      if (v >= vecs) continue;
      const float4 u = sum[rr][s];
      if (u.x == 0.f && u.y == 0.f && u.z == 0.f && u.w == 0.f) continue;
      float4* p = reinterpret_cast<float4*>(dk + (int64_t(b) * nk + j) * width + kh * d + v * 4);
      float4 o = *p;
      o.x -= scale * u.x; o.y -= scale * u.y; o.z -= scale * u.z; o.w -= scale * u.w;
      *p = o;
    }
  }
}
}  // namespace

void ZeroSumRepair(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& rho,
                   const torch::Tensor& meta, const torch::Tensor& log_z,
                   const c10::optional<torch::Tensor>& k_segment, int64_t window, bool anchor_scatter,
                   torch::Tensor rho_anchor, torch::Tensor dq, torch::Tensor dk, double scale,
                   const c10::optional<torch::Tensor>& variance, const c10::optional<torch::Tensor>& delta) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && k.dim() == 4, "zero-sum repair requires CUDA [B,N,H,D] Q/K");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "zero-sum repair requires FP16/BF16 Q/K");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && q.is_contiguous() && k.is_contiguous(), "Q/K must be contiguous with one dtype");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3) && k.size(2) > 0 && q.size(2) % k.size(2) == 0, "invalid Q/K head mapping");
  TORCH_CHECK(q.size(3) % 8 == 0 && q.size(3) <= 256, "zero-sum repair needs D a multiple of 8 up to 256");
  TORCH_CHECK(window >= 1, "zero-sum window must be positive");
  const int64_t rows = q.size(0) * q.size(2) * q.size(1);
  TORCH_CHECK(rho.scalar_type() == at::kDouble && rho.is_contiguous() && rho.numel() == rows, "rho must be contiguous FP64 [B,Hq,Nq]");
  TORCH_CHECK(rho_anchor.scalar_type() == at::kDouble && rho_anchor.is_contiguous() && rho_anchor.numel() == rows, "rho_anchor must be contiguous FP64 [B,Hq,Nq]");
  TORCH_CHECK(log_z.scalar_type() == at::kFloat && log_z.is_contiguous() && log_z.numel() == rows, "log_z must be contiguous FP32 [B,Hq,Nq]");
  TORCH_CHECK(variance.has_value() == delta.has_value(), "variance and delta are required together");
  if (variance) {
    TORCH_CHECK(variance->scalar_type() == at::kDouble && variance->is_contiguous() && variance->numel() == rows && variance->is_cuda() && variance->device() == q.device(), "variance must be contiguous FP64 [B,Hq,Nq]");
    TORCH_CHECK(delta->scalar_type() == at::kFloat && delta->is_contiguous() && delta->numel() == rows && delta->is_cuda() && delta->device() == q.device(), "delta must be contiguous FP32 [B,Hq,Nq]");
  }
  TORCH_CHECK(meta.scalar_type() == at::kFloat && meta.is_contiguous() && meta.numel() == 6 * rows, "invalid precision metadata");
  TORCH_CHECK(dq.scalar_type() == at::kFloat && dk.scalar_type() == at::kFloat && dq.is_contiguous() && dk.is_contiguous(), "dQ/dK must be contiguous FP32");
  TORCH_CHECK(dq.sizes() == q.sizes() && dk.sizes() == k.sizes(), "accumulator shapes must match Q/K");
  for (const auto& t : {rho, rho_anchor, log_z, meta, dq, dk}) TORCH_CHECK(t.is_cuda() && t.device() == q.device(), "repair tensors must share Q's device");
  if (k_segment) TORCH_CHECK(k_segment->is_cuda() && k_segment->device() == q.device() && k_segment->scalar_type() == at::kLong && k_segment->is_contiguous() && k_segment->numel() == k.size(0) * k.size(1), "invalid K segment ids");
  c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  const int b = q.size(0), nq = q.size(1), hq = q.size(2), d = q.size(3);
  const int nk = k.size(1), hk = k.size(2);
  if (!rows) return;
  if (!nk) { rho_anchor.copy_(rho); return; }
  TORCH_CHECK(nk < (1 << 24) && int64_t(b) * (int64_t(nk) + 2) < (int64_t(1) << 62), "zero-sum repair sequence too long");
  const int64_t total = int64_t(b) * nq, stride = int64_t(nk) + 2;
  auto iopt = q.options().dtype(at::kInt), lopt = q.options().dtype(at::kLong);
  auto lo = torch::empty({b, nq}, iopt), last = torch::empty({b, nq}, iopt);
  auto mono_lo = torch::empty({b, nq}, iopt), mono_last = torch::empty({b, nq}, iopt);
  auto enc = torch::empty({2, b, nq}, lopt), maxima = torch::empty({2, b, nq}, lopt);
  torch::Tensor changes;
  size_t temp_bytes = 0, scan_bytes = 0;
  int64_t* enc_lo = enc.data_ptr<int64_t>(); int64_t* enc_last = enc_lo + total;
  int64_t* max_lo = maxima.data_ptr<int64_t>(); int64_t* max_last = max_lo + total;
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, enc_lo, max_lo, MaxOp(), total, stream));
  if (k_segment) {
    const int64_t keys = int64_t(b) * nk;
    auto flags = torch::empty({b, nk}, iopt);
    changes = torch::empty({b, nk}, iopt);
    SegmentChangeKernel<<<(keys + 255) / 256, 256, 0, stream>>>(k_segment->data_ptr<int64_t>(), flags.data_ptr<int>(), nk, keys);
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, flags.data_ptr<int>(), changes.data_ptr<int>(), keys, stream));
    auto temp = torch::empty({int64_t(scan_bytes)}, q.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp.data_ptr(), scan_bytes, flags.data_ptr<int>(), changes.data_ptr<int>(), keys, stream));
  }
  const int* change_ptr = changes.defined() ? changes.data_ptr<int>() : nullptr;
  const float* first_plane = meta.data_ptr<float>() + 4 * rows;
  const float* last_plane = meta.data_ptr<float>() + 5 * rows;
  EncodeKernel<<<(total + 255) / 256, 256, 0, stream>>>(first_plane, last_plane, change_ptr, nq, nk, int64_t(hq) * nq,
      int(std::min<int64_t>(window, nk)), stride, total, lo.data_ptr<int>(), last.data_ptr<int>(), enc_lo, enc_last);
  auto temp = torch::empty({int64_t(temp_bytes)}, q.options().dtype(at::kByte));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveScan(temp.data_ptr(), temp_bytes, enc_lo, max_lo, MaxOp(), total, stream));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveScan(temp.data_ptr(), temp_bytes, enc_last, max_last, MaxOp(), total, stream));
  auto coefficient = torch::empty({b, hq, nq}, q.options().dtype(at::kFloat));
  CoefficientKernel<<<(total + 255) / 256, 256, 0, stream>>>(lo.data_ptr<int>(), last.data_ptr<int>(), change_ptr,
      max_lo, max_last, rho.data_ptr<double>(), log_z.data_ptr<float>(),
      variance ? variance->data_ptr<double>() : nullptr, delta ? delta->data_ptr<float>() : nullptr,
      mono_lo.data_ptr<int>(), mono_last.data_ptr<int>(),
      rho_anchor.data_ptr<double>(), coefficient.data_ptr<float>(), nq, nk, hq, stride, total);
  const int smem = (kChunk + 1) * d * int(sizeof(float));
  auto key_rows = torch::empty({b, nk, 2}, iopt);
  KeyRowsKernel<<<(int64_t(b) * nk + 255) / 256, 256, 0, stream>>>(mono_lo.data_ptr<int>(), mono_last.data_ptr<int>(),
      reinterpret_cast<int2*>(key_rows.data_ptr<int>()), nq, nk, int64_t(b) * nk);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "attention_zero_sum_repair", [&] {
    using E = std::conditional_t<std::is_same_v<scalar_t, c10::Half>, __half, __nv_bfloat16>;
    const E* qp = reinterpret_cast<const E*>(q.data_ptr());
    const E* kp = reinterpret_cast<const E*>(k.data_ptr());
    auto launch = [&](auto vpl) {
      constexpr int VPL = decltype(vpl)::value;
      C10_CUDA_CHECK(cudaFuncSetAttribute(DkKernel<E, VPL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      C10_CUDA_CHECK(cudaFuncSetAttribute(DqKernel<E, VPL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      DkKernel<E, VPL><<<dim3((nk + kRows - 1) / kRows, hk, b), kThreads, smem, stream>>>(qp, coefficient.data_ptr<float>(),
          reinterpret_cast<const int2*>(key_rows.data_ptr<int>()), dk.data_ptr<float>(), nq, nk, hq, hk, d, float(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      DqKernel<E, VPL><<<dim3((nq + kRows - 1) / kRows, hk, b), kThreads, smem, stream>>>(qp, kp, rho_anchor.data_ptr<double>(),
          meta.data_ptr<float>() + 3 * rows, coefficient.data_ptr<float>(), lo.data_ptr<int>(), last.data_ptr<int>(),
          dq.data_ptr<float>(), dk.data_ptr<float>(), nq, nk, hq, hk, d, float(scale), anchor_scatter);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (d <= 128) launch(std::integral_constant<int, 1>{}); else launch(std::integral_constant<int, 2>{});
  });
}
}  // namespace xattn::ops::attention::precision

// Author: Shicheng Wen

#include "softdelta/bwd.h"
#include "softdelta/buckets.h"
#include "softdelta/bwd_launch.h"
#include "softdelta/preprocess.h"
#include "softdelta/visibility.h"
#include "attention/semantics/visibility.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace xattn {
namespace ops {
namespace {

void SetStrides(const torch::Tensor& t, int64_t& batch, int64_t& row, int64_t& head) {
  batch = t.stride(0);
  row = t.stride(1);
  head = t.stride(2);
}

// The backward instance of (D, V): its bucket pair, causal dense rows remapped
// onto square instances (SoftDeltaBwdDenseSquareDim).
std::pair<int, int> BwdBucket(int d, int dv, bool causal_dense) {
  const int d_bucket = SoftDeltaBucketDim(d), v_bucket = SoftDeltaBucketDim(dv);
  const int square = causal_dense ? SoftDeltaBwdDenseSquareDim(d_bucket, v_bucket) : 0;
  return square ? std::pair<int, int>{square, square} : std::pair<int, int>{d_bucket, v_bucket};
}

// Reader groups (one work tile per key tile and KV head over its R readers,
// dK / dV written once) drop the FP32 dK / dV accumulation, its zeroing and
// the cast, but their tiles are R times coarser. Used for R <= 3 (MHA, groups
// of two) with enough work tiles W = B Hkv ceil(L / 128): 256 on dense rows;
// segment rows, where long documents make long tiles in worklist order, 1024,
// or L / 2 (at least 256) below L = 2048. Dense rows on 128-row tiles (D <= 64,
// V <= 128 instances) keep per-head launches.
// Coarser reader-group tiles can reduce parallelism for larger reader counts
// or small workloads.
bool UseReaderGroups(int64_t batch, int64_t length, int64_t heads, int64_t kv_heads, bool has_segment,
                     int d_bucket, int v_bucket) {
  const int64_t readers = heads / kv_heads + 1;
  const int64_t tiles = batch * kv_heads * ((length + 127) / 128);
  if (readers > 3 || (!has_segment && d_bucket <= 64 && v_bucket <= 128)) {
    return false;
  }
  return tiles >= (has_segment ? std::min<int64_t>(1024, std::max<int64_t>(256, length / 2)) : 256);
}

// The instance of the call's visibility (fwd.cu DispatchFwdVisibility).
template <typename Element, int kHeadDim, int kHeadDimV, bool kReaderGroups>
void DispatchBwdVisibility(AttentionBwdParams& params, int64_t visibility, bool has_segment,
                           cudaStream_t stream) {
  using namespace attention::semantics;
  if (visibility == kSoftDeltaChunk) {
    if (has_segment) {
      RunSoftDeltaBwdSm90<Element, kHeadDim, kHeadDimV, SlidingChunkVisibility, true, kReaderGroups>(
          params, stream);
    } else {
      RunSoftDeltaBwdSm90<Element, kHeadDim, kHeadDimV, SlidingChunkVisibility, false, kReaderGroups>(
          params, stream);
    }
  } else if (has_segment) {
    RunSoftDeltaBwdSm90<Element, kHeadDim, kHeadDimV, CausalSlidingWindowVisibility, true, kReaderGroups>(
        params, stream);
  } else if (visibility == kSoftDeltaWindow) {
    RunSoftDeltaBwdSm90<Element, kHeadDim, kHeadDimV, CausalSlidingWindowVisibility, false, kReaderGroups>(
        params, stream);
  } else {
    RunSoftDeltaBwdSm90<Element, kHeadDim, kHeadDimV, CausalFullVisibility, false, kReaderGroups>(
        params, stream);
  }
}

template <typename Element>
void DispatchBwd(AttentionBwdParams& params, int64_t visibility, bool has_segment, bool reader_groups,
                 int d_bucket, int v_bucket, cudaStream_t stream) {
#define XATTN_SOFTDELTA_BWD_CASE(D, V)                                                  \
  if (d_bucket == D && v_bucket == V) {                                                       \
    if (reader_groups) {                                                                      \
      DispatchBwdVisibility<Element, D, V, true>(params, visibility, has_segment, stream);    \
    } else {                                                                                  \
      DispatchBwdVisibility<Element, D, V, false>(params, visibility, has_segment, stream);   \
    }                                                                                         \
    return;                                                                                   \
  }
  XATTN_SOFTDELTA_FOR_EACH_BUCKET(XATTN_SOFTDELTA_BWD_CASE)
#undef XATTN_SOFTDELTA_BWD_CASE
  TORCH_CHECK(false, "softdelta fused BWD has no instance for D=", params.d, ", V=", params.dv);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> SoftDeltaBwd(
    const torch::Tensor& dy_in, const torch::Tensor& gate_in, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& neg_q,
    const torch::Tensor& y, const torch::Tensor& neg_read, const torch::Tensor& lse, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx, bool fold_neg_q,
    int64_t reader_groups, int64_t visibility, int64_t span, bool reset) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && neg_q.dim() == 4 &&
                  dy_in.dim() == 5 && gate_in.dim() == 5 && y.dim() == 4 && neg_read.dim() == 4 && lse.dim() == 3,
              "softdelta BWD expects q / k / v / neg_q / y / neg_read 4D, dy / gate 5D and lse 3D");
  const int64_t C = gate_in.size(3);
  const torch::Tensor dy = dy_in.contiguous().view({dy_in.size(0), dy_in.size(1), dy_in.size(2), -1});
  const torch::Tensor gate = gate_in.contiguous().view({gate_in.size(0), gate_in.size(1), gate_in.size(2), C});
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "softdelta BWD supports only fp16 and bf16");
  for (const torch::Tensor* t : {&k, &v, &neg_q, &dy, &gate}) {
    TORCH_CHECK(t->scalar_type() == q.scalar_type() && t->device() == q.device() && t->stride(-1) == 1,
                "softdelta BWD tensors must share q's dtype and device with a contiguous last dimension");
  }
  TORCH_CHECK(y.scalar_type() == q.scalar_type() && neg_read.scalar_type() == q.scalar_type() &&
                  y.device() == q.device() && neg_read.device() == q.device() &&
                  lse.scalar_type() == at::kFloat && lse.device() == q.device(),
              "softdelta BWD forward state must be in q's dtype with an FP32 LSE");
  const int64_t B = q.size(0), Lq = q.size(1), H = q.size(2), D = q.size(3);
  const int64_t Lk = k.size(1), Hkv = k.size(2), V = v.size(3);
  TORCH_CHECK(Lq == Lk, "softdelta fused BWD needs equal query and key lengths");
  TORCH_CHECK(dy.sizes() == torch::IntArrayRef({B, Lq, H, V}) &&
                  gate_in.sizes() == torch::IntArrayRef({B, Lq, H, C, 1}) &&
                  neg_q.sizes() == torch::IntArrayRef({B, Lq, Hkv, D}) &&
                  neg_read.sizes() == torch::IntArrayRef({B, Lq, Hkv, V}) &&
                  y.sizes() == torch::IntArrayRef({B, Lq, H, V}) &&
                  lse.sizes() == torch::IntArrayRef({B, Hkv + H, Lq}),
              "softdelta BWD shapes are inconsistent");
  const torch::Tensor neg_read_lse = lse.narrow(1, 0, Hkv);
  const torch::Tensor read_lse = lse.narrow(1, Hkv, H);
  TORCH_CHECK(q_segment_idx.has_value() == k_segment_idx.has_value(),
              "softdelta BWD needs both or neither segment index");
  const c10::cuda::CUDAGuard guard(q.device());
  // Contiguous gradients: the dK / dV cast writes the grouped accumulators flat.
  auto dq = torch::empty(q.sizes(), q.options());
  auto dk = torch::empty(k.sizes(), k.options());
  auto dv = torch::empty(v.sizes(), v.options());
  // Folding adds the negative-query gradient to dk (neg_q is the aligned keys, Lq == Lk).
  auto dneg_q = torch::empty({fold_neg_q ? 0 : B, Lq, Hkv, D}, q.options());
  auto neg_dy = torch::empty({B, Lq, Hkv, V}, q.options());
  auto gate_grad = torch::empty(gate_in.sizes(), gate_in.options());
  const int64_t padded = (Lq + 127) / 128 * 128;
  const bool has_segment = q_segment_idx.has_value();
  // Accumulator rows hold the instance's (bucket) head dimensions.
  const auto [d_bucket, v_bucket] =
      BwdBucket(static_cast<int>(D), static_cast<int>(V), !has_segment && visibility == kSoftDeltaCausal);
  TORCH_CHECK(d_bucket > 0 && v_bucket > 0, "softdelta fused BWD supports D, V <= 256");
  TORCH_CHECK(reader_groups >= -1 && reader_groups <= 1, "softdelta BWD reader_groups must be -1, 0 or 1");
  const bool groups = reader_groups < 0
      ? UseReaderGroups(B, Lk, H, Hkv, has_segment, d_bucket, v_bucket)
      : reader_groups == 1;
  // One FP32 workspace: dPsum, LSE_log2 and dQaccum [B, H + Hkv, ...], then the
  // grouped dK | dV accumulators; the preprocess zeroes dQaccum and dK | dV.
  // Reader groups write dV directly and keep only the folded dK rows (written
  // whole by the main kernel).
  const int64_t stat_numel = B * (H + Hkv) * padded;
  const int64_t dq_numel = stat_numel * d_bucket;
  const int64_t dk_numel = groups && !fold_neg_q ? 0 : B * Lk * Hkv * d_bucket;
  const int64_t dv_numel = groups ? 0 : B * Lk * Hkv * v_bucket;
  auto workspace = torch::empty({2 * stat_numel + dq_numel + dk_numel + dv_numel}, q.options().dtype(at::kFloat));
  auto dpsum = workspace.narrow(0, 0, stat_numel).view({B, H + Hkv, padded});
  auto lse_log2 = workspace.narrow(0, stat_numel, stat_numel).view({B, H + Hkv, padded});
  auto dq_accum = workspace.narrow(0, 2 * stat_numel, dq_numel).view({B, H + Hkv, padded * d_bucket});
  float* const dkv_accum = workspace.data_ptr<float>() + 2 * stat_numel + dq_numel;
  auto gate_grad_rows = gate_grad.view(gate.sizes());
  SoftDeltaBwdPreprocess(dy, gate, neg_read, y, neg_read_lse, read_lse, neg_dy, gate_grad_rows, dpsum, lse_log2,
                         dq_accum, static_cast<int>(padded),
                         groups ? DkvAccum{} : DkvAccum{dkv_accum, d_bucket, dkv_accum + dk_numel, v_bucket});

  AttentionBwdParams params;
  std::memset(&params, 0, sizeof(params));
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.dy_ptr = dy.data_ptr();
  params.y_ptr = y.data_ptr();
  params.lse_ptr = read_lse.data_ptr<float>();
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  SetStrides(q, params.q_batch_stride, params.q_row_stride, params.q_head_stride);
  SetStrides(k, params.k_batch_stride, params.k_row_stride, params.k_head_stride);
  SetStrides(v, params.v_batch_stride, params.v_row_stride, params.v_head_stride);
  SetStrides(dy, params.dy_batch_stride, params.dy_row_stride, params.dy_head_stride);
  SetStrides(y, params.y_batch_stride, params.y_row_stride, params.y_head_stride);
  SetStrides(dq, params.dq_batch_stride, params.dq_row_stride, params.dq_head_stride);
  SetStrides(dk, params.dk_batch_stride, params.dk_row_stride, params.dk_head_stride);
  SetStrides(dv, params.dv_batch_stride, params.dv_row_stride, params.dv_head_stride);
  params.neg_q_ptr = neg_q.data_ptr();
  SetStrides(neg_q, params.neg_q_batch_stride, params.neg_q_row_stride, params.neg_q_head_stride);
  params.neg_dy_ptr = neg_dy.data_ptr();
  SetStrides(neg_dy, params.neg_dy_batch_stride, params.neg_dy_row_stride, params.neg_dy_head_stride);
  params.fold_neg_q = fold_neg_q;
  if (!fold_neg_q) {
    params.dneg_q_ptr = dneg_q.data_ptr();
    SetStrides(dneg_q, params.dneg_q_batch_stride, params.dneg_q_row_stride, params.dneg_q_head_stride);
  }

  params.b = static_cast<int>(B);
  params.num_sequences = params.b;
  params.seqlen_q = static_cast<int>(Lq);
  params.seqlen_k = static_cast<int>(Lk);
  params.max_seqlen_q = params.seqlen_q;
  params.max_seqlen_k = params.seqlen_k;
  params.seqlen_q_padded = static_cast<int>(padded);
  params.dpsum_ptr = dpsum.data_ptr<float>();
  params.lse_log2_ptr = lse_log2.data_ptr<float>();
  params.dq_accum_ptr = dq_accum.data_ptr<float>();
  params.dkv_accum_ptr = dk_numel > 0 ? dkv_accum : nullptr;
  params.d = static_cast<int>(D);
  params.dv = static_cast<int>(V);
  params.h = static_cast<int>(H + Hkv);
  params.h_k = static_cast<int>(Hkv);
  params.scale_softmax = static_cast<float>(scale);
  if (has_segment) {
    TORCH_CHECK(q_segment_idx->scalar_type() == at::kLong && k_segment_idx->scalar_type() == at::kLong &&
                    q_segment_idx->is_contiguous() && k_segment_idx->is_contiguous(),
                "softdelta segment ids must be contiguous int64");
    params.q_segment_idx = q_segment_idx->data_ptr<int64_t>();
    params.k_segment_idx = k_segment_idx->data_ptr<int64_t>();
    params.k_segment_len = static_cast<int>(Lk);
  }
  const torch::Tensor chunk_positions = visibility == kSoftDeltaChunk && reset && has_segment
      ? SoftDeltaChunkPositions(*q_segment_idx, *k_segment_idx)
      : torch::Tensor();
  SetSoftDeltaVisibility(params, visibility, span, Lk,
                              chunk_positions.defined() ? chunk_positions.data_ptr<int>() : nullptr);
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kBFloat16) {
    DispatchBwd<cutlass::bfloat16_t>(params, visibility, has_segment, groups, d_bucket, v_bucket, stream);
  } else {
    DispatchBwd<cutlass::half_t>(params, visibility, has_segment, groups, d_bucket, v_bucket, stream);
  }
  return {dq, dk, dv, dneg_q, gate_grad};
}

}  // namespace ops
}  // namespace xattn

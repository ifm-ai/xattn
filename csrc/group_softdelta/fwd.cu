// Author: Shicheng Wen

#include "group_softdelta/fwd.h"
#include "group_softdelta/buckets.h"
#include "group_softdelta/fwd_launch.h"
#include "attention/hopper/launch.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>

#include <cstring>

namespace xattn {
namespace ops {
namespace {

void CheckFwdInputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                    const torch::Tensor& neg_q) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && neg_q.is_cuda(),
              "group_softdelta FWD requires CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && neg_q.dim() == 4,
              "group_softdelta FWD expects 4D q [B, Lq, H, D], k [B, Lk, Hkv, D], "
              "v [B, Lk, Hkv, V] and neg_q [B, Lq, Hkv, D]");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "group_softdelta FWD supports only fp16 and bf16");
  for (const torch::Tensor* t : {&k, &v, &neg_q}) {
    TORCH_CHECK(t->scalar_type() == q.scalar_type() && t->device() == q.device(),
                "group_softdelta FWD tensors must share a dtype and device");
  }
  for (const torch::Tensor* t : {&q, &k, &v, &neg_q}) {
    TORCH_CHECK(t->stride(-1) == 1, "group_softdelta FWD tensors need a contiguous last dimension");
  }
  const auto B = q.size(0), Lq = q.size(1), H = q.size(2), D = q.size(3);
  const auto Lk = k.size(1), Hkv = k.size(2);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == Lk && v.size(2) == Hkv &&
                  k.size(3) == D && Hkv > 0 && H % Hkv == 0 && Lq <= Lk,
              "group_softdelta FWD shapes are inconsistent");
  TORCH_CHECK(neg_q.size(0) == B && neg_q.size(1) == Lq && neg_q.size(2) == Hkv &&
                  neg_q.size(3) == D,
              "group_softdelta neg_q must be [B, Lq, Hkv, D]");
}

template <typename Element>
void DispatchFwd(AttentionFwdParams& params, bool has_segment, cudaStream_t stream) {
  const int d_bucket = GroupSoftDeltaBucketDim(params.d), v_bucket = GroupSoftDeltaBucketDim(params.dv);
#define XATTN_GROUP_SOFTDELTA_FWD_CASE(D, V)                                    \
  if (d_bucket == D && v_bucket == V) {                                         \
    if (has_segment) {                                                          \
      RunGroupSoftDeltaFwdSm90<Element, D, V, true>(params, stream);            \
    } else {                                                                    \
      RunGroupSoftDeltaFwdSm90<Element, D, V, false>(params, stream);           \
    }                                                                           \
    return;                                                                     \
  }
  XATTN_GROUP_SOFTDELTA_FOR_EACH_BUCKET(XATTN_GROUP_SOFTDELTA_FWD_CASE)
#undef XATTN_GROUP_SOFTDELTA_FWD_CASE
  TORCH_CHECK(false, "group_softdelta fused FWD has no instance for D=", params.d, ", V=", params.dv);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> GroupSoftDeltaFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& neg_q, double scale,
    const c10::optional<torch::Tensor>& q_segment_idx,
    const c10::optional<torch::Tensor>& k_segment_idx,
    const c10::optional<torch::Tensor>& output_state) {
  CheckFwdInputs(q, k, v, neg_q);
  TORCH_CHECK(q_segment_idx.has_value() == k_segment_idx.has_value(),
              "group_softdelta FWD needs both or neither segment index");
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t B = q.size(0), Lq = q.size(1), H = q.size(2);
  const int64_t Lk = k.size(1), Hkv = k.size(2), V = v.size(3);
  const int64_t logical_heads = H + Hkv;
  auto output = torch::empty({B, Lq, logical_heads, V}, q.options());
  auto lse = torch::empty({B, logical_heads, Lq}, q.options().dtype(at::kFloat));

  AttentionFwdParams params;
  std::memset(&params, 0, sizeof(params));
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = output.data_ptr();
  params.softmax_lse_ptr = lse.data_ptr<float>();
  if (output_state.has_value()) {
    const torch::Tensor& state = output_state.value();
    TORCH_CHECK(state.scalar_type() == at::kFloat && state.sizes() == output.sizes() &&
                    state.is_contiguous(),
                "group_softdelta output_state must be FP32 [B, Lq, H + Hkv, V]");
    params.o_state_ptr = state.data_ptr<float>();
    params.o_state_batch_stride = state.stride(0);
    params.o_state_row_stride = state.stride(1);
    params.o_state_head_stride = state.stride(2);
  }
  params.q_batch_stride = q.stride(0);
  params.q_row_stride = q.stride(1);
  params.q_head_stride = q.stride(2);
  params.k_batch_stride = k.stride(0);
  params.k_row_stride = k.stride(1);
  params.k_head_stride = k.stride(2);
  params.v_batch_stride = v.stride(0);
  params.v_row_stride = v.stride(1);
  params.v_head_stride = v.stride(2);
  params.o_batch_stride = output.stride(0);
  params.o_row_stride = output.stride(1);
  params.o_head_stride = output.stride(2);
  params.neg_q_ptr = neg_q.data_ptr();
  params.neg_q_batch_stride = neg_q.stride(0);
  params.neg_q_row_stride = neg_q.stride(1);
  params.neg_q_head_stride = neg_q.stride(2);

  params.b = static_cast<int>(B);
  params.num_sequences = params.b;
  params.seqlen_q = static_cast<int>(Lq);
  params.seqlen_k = static_cast<int>(Lk);
  params.max_seqlen_q = params.seqlen_q;
  params.max_seqlen_k = params.seqlen_k;
  params.d = static_cast<int>(q.size(3));
  params.dv = static_cast<int>(V);
  params.h = static_cast<int>(logical_heads);
  params.h_k = static_cast<int>(Hkv);
  params.scale_softmax = static_cast<float>(scale);
  params.window_size_left = static_cast<int>(Lk - 1);
  params.window_size_right = 0;
  params.use_fast_reciprocal = true;
  const bool has_segment = q_segment_idx.has_value();
  if (has_segment) {
    TORCH_CHECK(q_segment_idx->scalar_type() == at::kLong && k_segment_idx->scalar_type() == at::kLong &&
                    q_segment_idx->is_contiguous() && k_segment_idx->is_contiguous() &&
                    q_segment_idx->size(0) == B && q_segment_idx->size(1) == Lq &&
                    k_segment_idx->size(0) == B && k_segment_idx->size(1) == Lk,
                "group_softdelta segment ids must be contiguous int64 [B, Lq] and [B, Lk]");
    params.q_segment_idx = q_segment_idx->data_ptr<int64_t>();
    params.k_segment_idx = k_segment_idx->data_ptr<int64_t>();
    params.k_segment_len = static_cast<int>(Lk);
  }
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kBFloat16) {
    DispatchFwd<cutlass::bfloat16_t>(params, has_segment, stream);
  } else {
    DispatchFwd<cutlass::half_t>(params, has_segment, stream);
  }
  return {output, lse};
}

std::tuple<torch::Tensor, torch::Tensor> GroupSoftDeltaFwdVarlen(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& neg_q, const torch::Tensor& cu_seqlens,
    int64_t max_seqlen, double scale) {
  CheckFwdInputs(q, k, v, neg_q);
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t B = q.size(0), L = q.size(1), H = q.size(2);
  const int64_t Hkv = k.size(2), V = v.size(3);
  TORCH_CHECK(k.size(1) == L && v.size(1) == L && neg_q.size(1) == L,
              "group_softdelta varlen FWD needs equal Q / K lengths");
  auto output = torch::empty({B, L, Hkv + H, V}, q.options());
  auto lse = torch::empty({B, Hkv + H, L}, q.options().dtype(at::kFloat));
  auto readers = output.view({B * L, Hkv + H, V});
  auto read = readers.narrow(1, Hkv, H);
  auto neg_read = readers.narrow(1, 0, Hkv);
  auto read_lse = torch::empty({H, B * L}, lse.options());
  auto neg_read_lse = torch::empty({Hkv, B * L}, lse.options());
  auto rows = [&](const torch::Tensor& x) { return x.reshape({B * L, x.size(2), x.size(3)}); };
  attention::hopper::CausalAttentionSM90VarlenFwdInto(
      rows(q), rows(k), rows(v), cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, L - 1, scale,
      /*strict_past=*/false, read, read_lse);
  attention::hopper::CausalAttentionSM90VarlenFwdInto(
      rows(neg_q), rows(k), rows(v), cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, L - 1, scale,
      /*strict_past=*/true, neg_read, neg_read_lse);
  lse.narrow(1, Hkv, H).copy_(read_lse.view({H, B, L}).permute({1, 0, 2}));
  lse.narrow(1, 0, Hkv).copy_(neg_read_lse.view({Hkv, B, L}).permute({1, 0, 2}));
  return {output, lse};
}

}  // namespace ops
}  // namespace xattn

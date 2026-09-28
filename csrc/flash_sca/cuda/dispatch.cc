#include "flash_sca/cuda/instantiate.h"

namespace xattn {
namespace ops {

#define XATTN_FLASH_SCA_DISPATCH_FWD_CASE(HeadDim, HeadDimV, DType)      \
  do {                                                                  \
    if (D == HeadDim && V == HeadDimV) {                                \
      XATTN_FLASH_SCA_FWD_CASE_NAME(HeadDim, HeadDimV, DType)(           \
          q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx,    \
          k_segment_idx, has_prev, has_segment, y, lse);                \
      return;                                                           \
    }                                                                   \
  } while (false);

#define XATTN_FLASH_SCA_DISPATCH_BWD_CASE(HeadDim, HeadDimV, DType)      \
  do {                                                                  \
    if (D == HeadDim && V == HeadDimV) {                                \
      XATTN_FLASH_SCA_BWD_CASE_NAME(HeadDim, HeadDimV, DType)(           \
          dy, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,       \
          q_segment_idx, k_segment_idx, has_prev, has_segment, dq, dk,  \
          dv, prev_dk, prev_dv, deterministic);                         \
      return;                                                           \
    }                                                                   \
  } while (false);

void FlashSCAFwdFP16(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& y, torch::Tensor& lse) {
  const int64_t D = q.size(3);
  const int64_t V = v.size(3);
  XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(XATTN_FLASH_SCA_DISPATCH_FWD_CASE,
                                        FP16)
  TORCH_CHECK(false, "FlashSCA fwd currently supports only "
              "D,V in {32, 64, 128, 256}; got D=", D, ", V=", V);
}

void FlashSCAFwdBF16(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& y, torch::Tensor& lse) {
  const int64_t D = q.size(3);
  const int64_t V = v.size(3);
  XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(XATTN_FLASH_SCA_DISPATCH_FWD_CASE,
                                        BF16)
  TORCH_CHECK(false, "FlashSCA fwd currently supports only "
              "D,V in {32, 64, 128, 256}; got D=", D, ", V=", V);
}

void FlashSCABwdFP16(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& dq, torch::Tensor& dk, torch::Tensor& dv,
    c10::optional<torch::Tensor>& prev_dk,
    c10::optional<torch::Tensor>& prev_dv, bool deterministic) {
  const int64_t D = q.size(3);
  const int64_t V = v.size(3);
  XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(XATTN_FLASH_SCA_DISPATCH_BWD_CASE,
                                        FP16)
  TORCH_CHECK(false, "FlashSCA bwd currently supports only "
              "D,V in {32, 64, 128, 256}; got D=", D, ", V=", V);
}

void FlashSCABwdBF16(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& dq, torch::Tensor& dk, torch::Tensor& dv,
    c10::optional<torch::Tensor>& prev_dk,
    c10::optional<torch::Tensor>& prev_dv, bool deterministic) {
  const int64_t D = q.size(3);
  const int64_t V = v.size(3);
  XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(XATTN_FLASH_SCA_DISPATCH_BWD_CASE,
                                        BF16)
  TORCH_CHECK(false, "FlashSCA bwd currently supports only "
              "D,V in {32, 64, 128, 256}; got D=", D, ", V=", V);
}

#undef XATTN_FLASH_SCA_DISPATCH_BWD_CASE
#undef XATTN_FLASH_SCA_DISPATCH_FWD_CASE

}  // namespace ops
}  // namespace xattn

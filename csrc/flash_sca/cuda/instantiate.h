#pragma once

#include <c10/util/Optional.h>
#include <torch/types.h>

namespace xattn {
namespace ops {

void FlashSCAFwdFP16(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& y, torch::Tensor& lse);

void FlashSCAFwdBF16(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& y, torch::Tensor& lse);

void FlashSCABwdFP16(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& dq, torch::Tensor& dk, torch::Tensor& dv,
    c10::optional<torch::Tensor>& prev_dk,
    c10::optional<torch::Tensor>& prev_dv, bool deterministic);

void FlashSCABwdBF16(
    const torch::Tensor& dy, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& y, const torch::Tensor& lse,
    int64_t chunk_size, float scale, const torch::Tensor& prev_k,
    const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,
    const torch::Tensor& k_segment_idx, bool has_prev, bool has_segment,
    torch::Tensor& dq, torch::Tensor& dk, torch::Tensor& dv,
    c10::optional<torch::Tensor>& prev_dk,
    c10::optional<torch::Tensor>& prev_dv, bool deterministic);

#define XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(M, DType) \
  M(32, 32, DType)                                      \
  M(32, 64, DType)                                      \
  M(32, 128, DType)                                     \
  M(32, 256, DType)                                     \
  M(64, 32, DType)                                      \
  M(64, 64, DType)                                      \
  M(64, 128, DType)                                     \
  M(64, 256, DType)                                     \
  M(128, 32, DType)                                     \
  M(128, 64, DType)                                     \
  M(128, 128, DType)                                    \
  M(128, 256, DType)                                    \
  M(256, 32, DType)                                     \
  M(256, 64, DType)                                     \
  M(256, 128, DType)                                    \
  M(256, 256, DType)

#define XATTN_FLASH_SCA_FWD_CASE_NAME(HeadDim, HeadDimV, DType) \
  FlashSCAFwd##DType##Hdim##HeadDim##Vdim##HeadDimV

#define XATTN_FLASH_SCA_BWD_CASE_NAME(HeadDim, HeadDimV, DType) \
  FlashSCABwd##DType##Hdim##HeadDim##Vdim##HeadDimV

#define XATTN_FLASH_SCA_DECLARE_FWD_CASE(HeadDim, HeadDimV, DType)      \
  void XATTN_FLASH_SCA_FWD_CASE_NAME(HeadDim, HeadDimV, DType)(         \
      const torch::Tensor& q, const torch::Tensor& k,                  \
      const torch::Tensor& v, int64_t chunk_size, float scale,         \
      const torch::Tensor& prev_k, const torch::Tensor& prev_v,        \
      const torch::Tensor& q_segment_idx,                              \
      const torch::Tensor& k_segment_idx, bool has_prev,               \
      bool has_segment, torch::Tensor& y, torch::Tensor& lse);

#define XATTN_FLASH_SCA_DECLARE_BWD_CASE(HeadDim, HeadDimV, DType)      \
  void XATTN_FLASH_SCA_BWD_CASE_NAME(HeadDim, HeadDimV, DType)(         \
      const torch::Tensor& dy, const torch::Tensor& q,                 \
      const torch::Tensor& k, const torch::Tensor& v,                  \
      const torch::Tensor& y, const torch::Tensor& lse,                \
      int64_t chunk_size, float scale, const torch::Tensor& prev_k,    \
      const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx, \
      const torch::Tensor& k_segment_idx, bool has_prev,               \
      bool has_segment, torch::Tensor& dq, torch::Tensor& dk,          \
      torch::Tensor& dv, c10::optional<torch::Tensor>& prev_dk,        \
      c10::optional<torch::Tensor>& prev_dv, bool deterministic);

XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(XATTN_FLASH_SCA_DECLARE_FWD_CASE,
                                      FP16)
XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(XATTN_FLASH_SCA_DECLARE_FWD_CASE,
                                      BF16)
XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(XATTN_FLASH_SCA_DECLARE_BWD_CASE,
                                      FP16)
XATTN_FLASH_SCA_FOR_EACH_HEAD_DIM_PAIR(XATTN_FLASH_SCA_DECLARE_BWD_CASE,
                                      BF16)

#undef XATTN_FLASH_SCA_DECLARE_BWD_CASE
#undef XATTN_FLASH_SCA_DECLARE_FWD_CASE

}  // namespace ops
}  // namespace xattn

#define XATTN_FLASH_SCA_FWD_CASE_INSTANTIATE(                           \
    DType, Element, HeadDim, HeadDimV)                                  \
  namespace xattn {                                                      \
  namespace ops {                                                       \
  void XATTN_FLASH_SCA_FWD_CASE_NAME(HeadDim, HeadDimV, DType)(          \
      const torch::Tensor& q, const torch::Tensor& k,                   \
      const torch::Tensor& v, int64_t chunk_size, float scale,          \
      const torch::Tensor& prev_k, const torch::Tensor& prev_v,         \
      const torch::Tensor& q_segment_idx,                               \
      const torch::Tensor& k_segment_idx, bool has_prev,                \
      bool has_segment, torch::Tensor& y, torch::Tensor& lse) {         \
    FlashSCAFwdCase<Element, HeadDim, HeadDimV>(                        \
        q, k, v, chunk_size, scale, prev_k, prev_v, q_segment_idx,      \
        k_segment_idx, has_prev, has_segment, y, lse);                  \
  }                                                                     \
  }                                                                     \
  }

#define XATTN_FLASH_SCA_BWD_CASE_INSTANTIATE(                           \
    DType, Element, HeadDim, HeadDimV)                                  \
  namespace xattn {                                                      \
  namespace ops {                                                       \
  void XATTN_FLASH_SCA_BWD_CASE_NAME(HeadDim, HeadDimV, DType)(          \
      const torch::Tensor& dy, const torch::Tensor& q,                  \
      const torch::Tensor& k, const torch::Tensor& v,                   \
      const torch::Tensor& y, const torch::Tensor& lse,                 \
      int64_t chunk_size, float scale, const torch::Tensor& prev_k,     \
      const torch::Tensor& prev_v, const torch::Tensor& q_segment_idx,  \
      const torch::Tensor& k_segment_idx, bool has_prev,                \
      bool has_segment, torch::Tensor& dq, torch::Tensor& dk,           \
      torch::Tensor& dv, c10::optional<torch::Tensor>& prev_dk,         \
      c10::optional<torch::Tensor>& prev_dv, bool deterministic) {      \
    FlashSCABwdCase<Element, HeadDim, HeadDimV>(                        \
        dy, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,         \
        q_segment_idx, k_segment_idx, has_prev, has_segment, dq, dk,    \
        dv, prev_dk, prev_dv, deterministic);                           \
  }                                                                     \
  }                                                                     \
  }

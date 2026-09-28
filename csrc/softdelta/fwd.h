#pragma once

#include <cuda_runtime_api.h>
#include <torch/types.h>

#include "attention/hopper/fwd_params.h"
#include "attention/semantics/visibility.h"

namespace xattn {
namespace ops {

enum class FlashSoftDeltaVisibility : int64_t {
  kFull = 0,
  kSlidingWindow = 1,
  kSlidingChunk = 2,
};

struct FlashSoftDeltaFwdParams : AttentionFwdParams {
  const void* __restrict__ gate_ptr;
  index_t gate_batch_stride;
  index_t gate_row_stride;
  index_t gate_head_stride;
  index_t gate_group_stride;
  int gate_group_dim;
  bool emit_ungated_pair;
};

template <
    typename Element, typename ElementOut, int kHeadDim,
    int kHeadDimV, typename Visibility>
void RunFlashSoftDeltaFwdSm90VD(
    AttentionFwdParams& params, cudaStream_t stream);

torch::Tensor FlashSoftDeltaFwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    int64_t span,
    double scale,
    int64_t visibility);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FlashSoftDeltaTrainingFwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    int64_t span,
    double scale,
    int64_t visibility,
    const c10::optional<torch::Tensor>& output_state = c10::nullopt);


}  // namespace ops
}  // namespace xattn

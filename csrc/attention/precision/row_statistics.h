#pragma once
#include <torch/types.h>
namespace xattn::ops::attention::precision {
// Recompute split softmax statistics and anchored dP without materializing P.
// Bounds [B,Nq] are inclusive key indices; right<left denotes an empty row.
// Optional segment arrays may contain arbitrary IDs; equality filters support.
// Optional exact [B] int32: nonzero when a batch's bounds already equal its
// segment support (nondecreasing runs), which skips the per-key segment test.
// Returns delta [B,Hq,Nq], log_z [B,Hq,Nq], metadata [6,B,Hq,Nq].
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> RowStatistics(
    const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,
    const torch::Tensor& g,const torch::Tensor& left,const torch::Tensor& right,
    const c10::optional<torch::Tensor>& q_segment,
    const c10::optional<torch::Tensor>& k_segment,double scale,
    const c10::optional<torch::Tensor>& exact);
}
namespace xattn::ops::attention::precision {
void FallbackDV(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& scaled_g,
    const torch::Tensor& left,const torch::Tensor& right,const torch::Tensor& log_z,
    const torch::Tensor& metadata,const torch::Tensor& flags,torch::Tensor dv,
    const c10::optional<torch::Tensor>& q_segment,
    const c10::optional<torch::Tensor>& k_segment,double scale);
}

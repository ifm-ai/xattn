#pragma once
#include <torch/types.h>
namespace xattn::ops::attention::precision {
// Exact equality runs of complete K vectors, encoded as int32 [B,Hkv,Nk].
torch::Tensor KeyEqualityRuns(const torch::Tensor& k);
// If a row's entire visible interval has one K vector, softmax invariance
// implies dQ=0 exactly. Segment filtering can only shrink that interval.
void EnforceConstantKeys(const torch::Tensor& runs,const torch::Tensor& metadata,
                         torch::Tensor dq);
}

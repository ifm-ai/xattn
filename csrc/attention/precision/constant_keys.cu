#include "attention/precision/constant_keys.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace xattn::ops::attention::precision {
namespace {
template<class E>
__global__ void FindKeyChanges(const E* k,int* changes,int64_t rows,int n,int h,int d){
 const int64_t row=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
 const int lane=threadIdx.x%32;if(row>=rows)return;
 int j=row%n,head=row/n%h,b=row/(int64_t(n)*h);
 bool changed=j==0;
 if(j>0){
  int64_t offset=((int64_t(b)*n+j)*h+head)*d;
  for(int c=lane;c<d;c+=32)changed|=float(k[offset+c])!=float(k[offset-int64_t(h)*d+c]);
 }
 unsigned mask=__ballot_sync(0xffffffff,changed);
 if(lane==0)changes[row]=int(mask!=0);
}
__global__ void ZeroConstantKeyQueries(const int* runs,const float* meta,float* dq,
 int64_t count,int nq,int nk,int h,int hk,int d,int batch){
 // One warp per query/head. The invariant is shared by every feature; avoid
 // repeating six integer divisions and metadata reads for each output scalar.
 const int lane=threadIdx.x%32;
 const int64_t query=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
 const int64_t queries=count/d;
 if(query>=queries)return;
 int zero=0;
 if(lane==0){
  int head=query%h,i=query/h%nq,b=query/(int64_t(h)*nq);
  int64_t row=(int64_t(b)*h+head)*nq+i,rows=int64_t(batch)*h*nq;
  int left=max(0,int(meta[4*rows+row])),right=min(nk-1,int(meta[5*rows+row]));
  if(right<left)zero=1;
  else {
   int64_t base=(int64_t(b)*hk+head/(h/hk))*nk;
   zero=runs[base+left]==runs[base+right];
  }
 }
 zero=__shfl_sync(0xffffffff,zero,0);
 if(zero)for(int c=lane;c<d;c+=32)dq[query*d+c]=0.f;
}
}
torch::Tensor KeyEqualityRuns(const torch::Tensor& k){
 TORCH_CHECK(k.is_cuda() && k.dim()==4 && k.is_contiguous(),"K runs require contiguous CUDA [B,N,H,D]");
 TORCH_CHECK(k.scalar_type()==at::kHalf || k.scalar_type()==at::kBFloat16,"K runs require FP16/BF16");
 TORCH_CHECK(k.size(1)<(1<<24),"K run length exceeds precision metadata capacity");
 c10::cuda::CUDAGuard guard(k.device());
 auto changes=torch::empty({k.size(0),k.size(2),k.size(1)},k.options().dtype(at::kInt));
 if(!changes.numel())return changes;
 auto stream=at::cuda::getCurrentCUDAStream(k.get_device());
 AT_DISPATCH_REDUCED_FLOATING_TYPES(k.scalar_type(),"precision_key_runs",[&]{
  FindKeyChanges<scalar_t><<<(changes.numel()+3)/4,128,0,stream>>>(k.data_ptr<scalar_t>(),changes.data_ptr<int>(),changes.numel(),k.size(1),k.size(2),k.size(3));
 });C10_CUDA_KERNEL_LAUNCH_CHECK();return changes.cumsum(-1,at::kInt);
}
void EnforceConstantKeys(const torch::Tensor& runs,const torch::Tensor& meta,torch::Tensor dq){
 TORCH_CHECK(dq.is_cuda() && dq.dim()==4 && dq.scalar_type()==at::kFloat && dq.is_contiguous(),"constant-K correction requires contiguous FP32 dQ");
 for(const auto& t:{runs,meta})TORCH_CHECK(t.is_cuda() && t.device()==dq.device() && t.is_contiguous(),"constant-K metadata must be contiguous on dQ device");
 TORCH_CHECK(runs.dim()==3 && runs.scalar_type()==at::kInt && runs.size(0)==dq.size(0) && runs.size(1)>0 && dq.size(2)%runs.size(1)==0,"invalid K equality runs");
 TORCH_CHECK(meta.scalar_type()==at::kFloat && meta.numel()==6*dq.size(0)*dq.size(1)*dq.size(2),"invalid precision metadata");
 if(!dq.numel())return;c10::cuda::CUDAGuard guard(dq.device());auto stream=at::cuda::getCurrentCUDAStream(dq.get_device());
 ZeroConstantKeyQueries<<<((dq.numel()/dq.size(3))+3)/4,128,0,stream>>>(runs.data_ptr<int>(),meta.data_ptr<float>(),dq.data_ptr<float>(),dq.numel(),dq.size(1),runs.size(2),dq.size(2),runs.size(1),dq.size(3),dq.size(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

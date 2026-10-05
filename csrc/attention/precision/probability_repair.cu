#include "attention/precision/probability_repair.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace xattn::ops::attention::precision {
namespace {
template<class E>
__global__ void RepairProbabilityKernel(const E* g,const double* stats,
 const float* meta,float* dv,int* flags,int64_t elements,int n,int nk,int h,int hk,int d,int bcount) {
 int64_t x=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(x>=elements)return;
 int c=x%d,head=x/d%h,i=x/(int64_t(d)*h)%n,b=x/(int64_t(d)*h*n),kh=head/(h/hk);
 int64_t row=(int64_t(b)*h+head)*n+i,count=int64_t(bcount)*h*n;
 int anchor=int(meta[3*count+row]);if(anchor<0)return;
 double mass=stats[count+row],pa=stats[3*count+row];
 float delta=float(mass-1.);
 bool safe=anchor<nk && isfinite(mass) && mass>0. && isfinite(pa) && pa>=0. &&
           (delta<=0.f || double(delta)<=pa*(1.-2e-6));
 if(!safe) {if(c==0) atomicExch(flags+b*hk+kh,1);return;}
 float correction=delta*float(g[x]);
 if(correction!=0.f) atomicAdd(dv+((int64_t(b)*nk+anchor)*hk+kh)*d+c,-correction);
}
}
torch::Tensor ProbabilityRepair(const torch::Tensor& g,const torch::Tensor& stats,
                               const torch::Tensor& meta,torch::Tensor dv) {
 TORCH_CHECK(g.is_cuda() && g.dim()==4 && dv.dim()==4,"probability repair needs rank-four CUDA tensors");
 TORCH_CHECK(g.scalar_type()==at::kHalf || g.scalar_type()==at::kBFloat16,"dO must be FP16/BF16");
 for(const auto& x:{g,stats,meta,dv}) TORCH_CHECK(x.is_cuda() && x.device()==g.device() && x.is_contiguous(),"probability repair tensors must be contiguous and on one CUDA device");
 TORCH_CHECK(dv.scalar_type()==at::kFloat && stats.scalar_type()==at::kDouble && meta.scalar_type()==at::kFloat,"invalid probability repair workspace dtype");
 TORCH_CHECK(dv.size(0)==g.size(0) && dv.size(3)==g.size(3) && dv.size(2)>0 && g.size(2)%dv.size(2)==0,"invalid dV or GQA shape");
 int64_t rows=g.size(0)*g.size(1)*g.size(2);
 TORCH_CHECK(stats.numel()==4*rows && meta.numel()==6*rows,"invalid row-statistics shape");
 c10::cuda::CUDAGuard guard(g.device());auto flags=torch::zeros({g.size(0),dv.size(2)},g.options().dtype(at::kInt));
 if(!g.numel())return flags;auto stream=at::cuda::getCurrentCUDAStream(g.get_device());
 AT_DISPATCH_REDUCED_FLOATING_TYPES(g.scalar_type(),"precision_probability_repair",[&]{
 RepairProbabilityKernel<scalar_t><<<(g.numel()+255)/256,256,0,stream>>>(g.data_ptr<scalar_t>(),stats.data_ptr<double>(),meta.data_ptr<float>(),dv.data_ptr<float>(),flags.data_ptr<int>(),g.numel(),g.size(1),dv.size(1),g.size(2),dv.size(2),g.size(3),g.size(0));
 });C10_CUDA_KERNEL_LAUNCH_CHECK();return flags;
}
}

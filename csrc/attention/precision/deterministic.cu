#include "attention/precision/deterministic_cached.h"
#include "attention/precision/deterministic_cached_bounds.h"
#include "attention/precision/deterministic.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>
namespace xattn::ops::attention::precision {
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> DeterministicBackward(
 const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,const torch::Tensor& g,
 const torch::Tensor& delta,const torch::Tensor& logz,const torch::Tensor& stats,const torch::Tensor& meta,
 const torch::Tensor& left,const torch::Tensor& right,const c10::optional<torch::Tensor>& qs,
 const c10::optional<torch::Tensor>& ks,double scale,int level){
 TORCH_CHECK(q.dim()==4,"deterministic operands require rank-four Q");
 // Long-row bounds search checks monotonicity on GPU and retains the exact scan
 // for independently reset/nonmonotone supports. Short rows use cache-only D05.
 auto fn=q.size(1)>=8192 ? cached_bounds::DeterministicBackward : cached::DeterministicBackward;
 return fn(q,k,v,g,delta,logz,stats,meta,left,right,qs,ks,scale,level);
}
}

namespace xattn::ops::attention::precision {
namespace {
template<class E>
__global__ void RepairQuery(const E* K,const double* stats,const float* meta,float* dq,
 int64_t elements,int nq,int nk,int h,int hk,int d,int batch,float scale){
 int64_t x=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(x>=elements)return;
 int c=x%d,head=x/d%h,i=x/(int64_t(d)*h)%nq,b=x/(int64_t(d)*h*nq);
 int64_t row=(int64_t(b)*h+head)*nq+i,count=int64_t(batch)*h*nq;int anchor=int(meta[3*count+row]);
 if(anchor<0 || anchor>=nk)return;
 dq[x]-=scale*float(stats[row])*float(K[((int64_t(b)*nk+anchor)*hk+head/(h/hk))*d+c]);
}
template<class E>
__global__ void RepairKeys(const E* Q,const E* G,const double* stats,const float* meta,
 const int64_t* order,float* dk,float* dv,int* flags,int nq,int nk,int h,int hk,int d,int vd,int batch,float scale,int level){
 const int key=blockIdx.x,bh=blockIdx.y,b=bh/hk,kh=bh%hk,c=threadIdx.x;
 const int rows=nq*(h/hk);const int64_t count=int64_t(batch)*h*nq,base=int64_t(bh)*rows;
 int lo=0,hi=rows;
 while(lo<hi){int mid=(lo+hi)/2;int64_t r=order[base+mid];int head=kh*(h/hk)+r/nq,i=r%nq;
  int anchor=int(meta[3*count+(int64_t(b)*h+head)*nq+i]);if(anchor<key)lo=mid+1;else hi=mid;}
 float ck=0.f,cv=0.f;
 for(int p=lo;p<rows;++p){int64_t r=order[base+p];int head=kh*(h/hk)+r/nq,i=r%nq;
  int64_t row=(int64_t(b)*h+head)*nq+i;if(int(meta[3*count+row])!=key)break;
  if(c<d)ck+=(scale*float(stats[row]))*float(Q[((int64_t(b)*nq+i)*h+head)*d+c]);
  if(level==4){double mass=stats[count+row],pa=stats[3*count+row];float delta=float(mass-1.);
   bool safe=isfinite(mass) && mass>0. && isfinite(pa) && pa>=0. && (delta<=0.f || double(delta)<=pa*(1.-2e-6));
   if(!safe){if(c==0)atomicExch(flags+bh,1);}
   else if(c<vd)cv+=delta*float(G[((int64_t(b)*nq+i)*h+head)*vd+c]);
  }
 }
 if(c<d)dk[((int64_t(b)*nk+key)*hk+kh)*d+c]-=ck;
 if(level==4 && c<vd)dv[((int64_t(b)*nk+key)*hk+kh)*vd+c]-=cv;
}
}
torch::Tensor DeterministicRepair(const torch::Tensor& q,const torch::Tensor& k,
 const torch::Tensor& g,const torch::Tensor& stats,const torch::Tensor& meta,const torch::Tensor& order,
 torch::Tensor dq,torch::Tensor dk,torch::Tensor dv,double scale,int level){
 c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
 TORCH_CHECK(order.is_cuda() && order.device()==q.device() && order.scalar_type()==at::kLong && order.is_contiguous(),"invalid deterministic anchor ordering");
 auto flags=torch::zeros({q.size(0),k.size(2)},q.options().dtype(at::kInt));
 if(!q.numel())return flags;
 AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(),"precision_deterministic_repair",[&]{
  RepairQuery<scalar_t><<<(q.numel()+255)/256,256,0,stream>>>(k.data_ptr<scalar_t>(),stats.data_ptr<double>(),meta.data_ptr<float>(),dq.data_ptr<float>(),q.numel(),q.size(1),k.size(1),q.size(2),k.size(2),q.size(3),q.size(0),float(scale));
  RepairKeys<scalar_t><<<dim3(k.size(1),q.size(0)*k.size(2)),256,0,stream>>>(q.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),stats.data_ptr<double>(),meta.data_ptr<float>(),order.data_ptr<int64_t>(),dk.data_ptr<float>(),dv.data_ptr<float>(),flags.data_ptr<int>(),q.size(1),k.size(1),q.size(2),k.size(2),q.size(3),g.size(3),q.size(0),float(scale),level);
 });C10_CUDA_KERNEL_LAUNCH_CHECK();return flags;
}
}

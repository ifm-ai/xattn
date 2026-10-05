#include "attention/precision/deterministic.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>
namespace xattn::ops::attention::precision::uncached {
namespace {
namespace wm=nvcuda::wmma;
__device__ inline float ToFloat(__half x){return __half2float(x);}
__device__ inline float ToFloat(__nv_bfloat16 x){return __bfloat162float(x);}
template<class E> __device__ inline E ToLow(float x){
 if constexpr(std::is_same_v<E,__half>)return __float2half_rn(x);
 else return __float2bfloat16_rn(x);
}
template<class E> __device__ inline void Copy16(E* dst,const E* src,bool valid){
 unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(src),"r"(valid?16:0));
}
__device__ inline void WaitCopy(){asm volatile("cp.async.commit_group;");asm volatile("cp.async.wait_group 0;");}
using Acc=wm::fragment<wm::accumulator,16,16,16,float>;
template<class E> struct Tile {
 E a[256],b[256],ds[256],dsl[256],p[256],pl[256];
 float s[256],dp[256],out[256];
};
// All reductions have one owner. dQ CTAs visit K in increasing order;
// dK/dV CTAs visit query heads and query positions in increasing order.
// Main contractions remain low-dtype Tensor Core operations.
template<class E,int Bucket,int Level,bool Query>
__global__ void DeterministicMain(
 const E* Q,const E* K,const E* V,const E* G,const float* delta,const float* logz,
 double* stats,const float* meta,const int* L,const int* R,const int64_t* QI,const int64_t* KI,
 float* DQ,float* DK,float* DV,int nq,int nk,int h,int hk,int d,int dv,int batch,float alpha) {
 __shared__ __align__(32) Tile<E> t;
 const int lane=threadIdx.x,bh=blockIdx.y,b=bh/(Query?h:hk),head0=bh%(Query?h:hk);
 const int fixed=blockIdx.x*16,group=h/hk;
 const int64_t count=int64_t(batch)*h*nq;
 Acc ak[Bucket/16],av[Bucket/16];
 #pragma unroll
 for(int c=0;c<Bucket/16;++c){wm::fill_fragment(ak[c],0.f);wm::fill_fragment(av[c],0.f);}
 double rho=0.,mass=0.,pa=0.;
 const int head_begin=Query?head0:head0*group,head_end=Query?head0+1:(head0+1)*group;
 for(int head=head_begin;head<head_end;++head) {
  const int kh=head/group;
  int begin=0,end=Query?nk:nq;
  if constexpr(Query) {
   int lo=lane<16 && fixed+lane<nq ? L[b*nq+fixed+lane] : nk;
   int hi=lane<16 && fixed+lane<nq ? R[b*nq+fixed+lane] : -1;
   for(int off=16;off;off/=2){lo=min(lo,__shfl_xor_sync(0xffffffff,lo,off));hi=max(hi,__shfl_xor_sync(0xffffffff,hi,off));}
   begin=max(lo,0)/16*16;end=min(nk,hi+1);
  } else {
   // Only skip whole query tiles whose visible interval cannot touch this K tile.
   // The scan uses bounds, not segment assumptions, so independently reset runs
   // remain correct. Accumulation order of every live contribution is unchanged.
   int first=nq,last=-1;
   for(int i=lane;i<nq;i+=32)if(L[b*nq+i]<fixed+16 && R[b*nq+i]>=fixed){first=min(first,i);last=max(last,i);}
   for(int off=16;off;off/=2){first=min(first,__shfl_xor_sync(0xffffffff,first,off));last=max(last,__shfl_xor_sync(0xffffffff,last,off));}
   begin=first/16*16;end=last+1;
  }
  for(int pos=begin;pos<end;pos+=16) {
   const int q0=Query?fixed:pos,k0=Query?pos:fixed;
   Acc ss,dd;wm::fill_fragment(ss,0.f);wm::fill_fragment(dd,0.f);
   for(int c0=0;c0<d;c0+=16){
    {int x=lane*8,r=x/16,c=c0+x%16;
     bool qa=q0+r<nq && c<d,kb=k0+r<nk && c<d;
     Copy16(t.a+x,qa?Q+((int64_t(b)*nq+q0+r)*h+head)*d+c:Q,qa);
     Copy16(t.b+x,kb?K+((int64_t(b)*nk+k0+r)*hk+kh)*d+c:K,kb);WaitCopy();}
    __syncwarp();wm::fragment<wm::matrix_a,16,16,16,E,wm::row_major>a;
    wm::fragment<wm::matrix_b,16,16,16,E,wm::col_major>bt;
    wm::load_matrix_sync(a,t.a,16);wm::load_matrix_sync(bt,t.b,16);wm::mma_sync(ss,a,bt,ss);__syncwarp();
   }
   wm::store_matrix_sync(t.s,ss,16,wm::mem_row_major);
   for(int c0=0;c0<dv;c0+=16){
    {int x=lane*8,r=x/16,c=c0+x%16;
     bool ga=q0+r<nq && c<dv,vb=k0+r<nk && c<dv;
     Copy16(t.a+x,ga?G+((int64_t(b)*nq+q0+r)*h+head)*dv+c:G,ga);
     Copy16(t.b+x,vb?V+((int64_t(b)*nk+k0+r)*hk+kh)*dv+c:V,vb);WaitCopy();}
    __syncwarp();wm::fragment<wm::matrix_a,16,16,16,E,wm::row_major>a;
    wm::fragment<wm::matrix_b,16,16,16,E,wm::col_major>bt;
    wm::load_matrix_sync(a,t.a,16);wm::load_matrix_sync(bt,t.b,16);wm::mma_sync(dd,a,bt,dd);__syncwarp();
   }
   wm::store_matrix_sync(t.dp,dd,16,wm::mem_row_major);__syncwarp();
   for(int x=lane;x<256;x+=32){int i=q0+x/16,j=k0+x%16;float p=0.f,ds=0.f;
    bool valid=i<nq && j<nk && j>=L[b*nq+i] && j<=R[b*nq+i];
    if(QI && valid)valid=QI[b*nq+i]==KI[b*nk+j];
    if(valid){int64_t row=(int64_t(b)*h+head)*nq+i;
     float z=__fmul_rn(__fsub_rn(t.s[x],meta[row]),alpha*1.4426950408889634f)-logz[row]*1.4426950408889634f;
     float rs=Level==4?meta[2*count+row]:1.f;
     float dp=__fsub_rn(__fmul_rn(t.dp[x],rs),meta[count+row])-delta[row];
     if constexpr(Level>=2){float root=exp2f(0.5f*z);ds=root*(root*dp);float sr=root*sqrtf(rs);p=sr*sr;}
     else {p=exp2f(z);ds=p*dp;}
    }
    t.ds[x]=ToLow<E>(ds);t.p[x]=ToLow<E>(p);
    if constexpr(Level==4){t.dsl[x]=ToLow<E>(ds-ToFloat(t.ds[x]));t.pl[x]=ToLow<E>(p-ToFloat(t.p[x]));}
   }
   __syncwarp();
   if constexpr(Query && Level>=3){if(lane<16 && q0+lane<nq){
    int64_t row=(int64_t(b)*h+head)*nq+q0+lane;
    double inv=1./double(meta[2*count+row]);int anchor=int(meta[3*count+row]);
    for(int j=0;j<16;++j){int x=lane*16+j;rho+=double(ToFloat(t.ds[x]));
     if constexpr(Level==4){rho+=double(ToFloat(t.dsl[x]));double p=double(ToFloat(t.p[x]))+double(ToFloat(t.pl[x]));mass+=p*inv;if(k0+j==anchor)pa+=p*inv;}
    }
   }}
   #pragma unroll
   for(int c0=0;c0<Bucket;c0+=16){
    if(c0<d){
     {int x=lane*8,r=x/16,c=c0+x%16;
      bool valid=(Query?k0+r<nk:q0+r<nq) && c<d;
      const E* src=Query?K:Q;
      if(valid)src=Query?K+((int64_t(b)*nk+k0+r)*hk+kh)*d+c:Q+((int64_t(b)*nq+q0+r)*h+head)*d+c;
      Copy16(t.b+x,src,valid);WaitCopy();}
     __syncwarp();wm::fragment<wm::matrix_a,16,16,16,E,std::conditional_t<Query,wm::row_major,wm::col_major>>a;
     wm::fragment<wm::matrix_b,16,16,16,E,wm::row_major>bt;
     wm::load_matrix_sync(a,t.ds,16);wm::load_matrix_sync(bt,t.b,16);wm::mma_sync(ak[c0/16],a,bt,ak[c0/16]);
     if constexpr(Level==4){wm::load_matrix_sync(a,t.dsl,16);wm::mma_sync(ak[c0/16],a,bt,ak[c0/16]);}
     __syncwarp();
    }
    if constexpr(!Query){if(c0<dv){
     {int x=lane*8,r=x/16,c=c0+x%16;bool valid=q0+r<nq && c<dv;
      Copy16(t.b+x,valid?G+((int64_t(b)*nq+q0+r)*h+head)*dv+c:G,valid);WaitCopy();}
     __syncwarp();wm::fragment<wm::matrix_a,16,16,16,E,wm::col_major>a;
     wm::fragment<wm::matrix_b,16,16,16,E,wm::row_major>bt;
     wm::load_matrix_sync(a,t.p,16);wm::load_matrix_sync(bt,t.b,16);wm::mma_sync(av[c0/16],a,bt,av[c0/16]);
     if constexpr(Level==4){wm::load_matrix_sync(a,t.pl,16);wm::mma_sync(av[c0/16],a,bt,av[c0/16]);}
     __syncwarp();
    }}
   }
  }
 }
 #pragma unroll
 for(int c0=0;c0<Bucket;c0+=16){
  if(c0<d){wm::store_matrix_sync(t.out,ak[c0/16],16,wm::mem_row_major);__syncwarp();
   for(int x=lane;x<256;x+=32){int r=fixed+x/16,c=c0+x%16;if(r<(Query?nq:nk) && c<d){int64_t out=((int64_t(b)*(Query?nq:nk)+r)*(Query?h:hk)+head0)*d+c;(Query?DQ:DK)[out]=alpha*t.out[x];}}
   __syncwarp();}
  if constexpr(!Query){if(c0<dv){wm::store_matrix_sync(t.out,av[c0/16],16,wm::mem_row_major);__syncwarp();
   for(int x=lane;x<256;x+=32){int r=fixed+x/16,c=c0+x%16;if(r<nk && c<dv)DV[((int64_t(b)*nk+r)*hk+head0)*dv+c]=t.out[x];}__syncwarp();}}
 }
 if constexpr(Query && Level>=3){if(lane<16 && fixed+lane<nq){int64_t row=(int64_t(b)*h+head0)*nq+fixed+lane;stats[row]=rho;if constexpr(Level==4){stats[count+row]=mass;stats[3*count+row]=pa;}}}
}
} // namespace
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> DeterministicBackward(
 const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,const torch::Tensor& g,
 const torch::Tensor& delta,const torch::Tensor& logz,const torch::Tensor& stats,const torch::Tensor& meta,
 const torch::Tensor& left,const torch::Tensor& right,const c10::optional<torch::Tensor>& qs,
 const c10::optional<torch::Tensor>& ks,double scale,int level){
 TORCH_CHECK(level>=2 && level<=4,"invalid deterministic precision level");
 TORCH_CHECK(q.is_cuda() && q.dim()==4 && (q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16),"invalid Q");
 for(auto& x:{q,k,v,g})TORCH_CHECK(x.is_cuda() && x.device()==q.device() && x.is_contiguous() && x.dim()==4 && x.scalar_type()==q.scalar_type(),"invalid deterministic operands");
 TORCH_CHECK(q.size(3)<=256 && v.size(3)<=256 && q.size(3)>0 && v.size(3)>0 && q.size(3)%8==0 && v.size(3)%8==0,"unsupported dimension");
 for(const auto& t:{q,k,v,g}) TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"deterministic operands need 16-byte alignment");
 c10::cuda::CUDAGuard guard(q.device());auto opts=q.options().dtype(at::kFloat);
 auto dq=torch::zeros(q.sizes(),opts),dk=torch::zeros(k.sizes(),opts),dv=torch::zeros(v.sizes(),opts);
 if(!q.numel())return {dq,dk,dv};auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
 AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(),"precision_deterministic",[&]{
  using E=std::conditional_t<std::is_same_v<scalar_t,c10::Half>,__half,__nv_bfloat16>;
  auto launch=[&](auto bucket,auto lv){constexpr int D=decltype(bucket)::value,Level=decltype(lv)::value;

   DeterministicMain<E,D,Level,true><<<dim3((q.size(1)+15)/16,q.size(0)*q.size(2)),32,0,stream>>>(reinterpret_cast<E*>(q.data_ptr()),reinterpret_cast<E*>(k.data_ptr()),reinterpret_cast<E*>(v.data_ptr()),reinterpret_cast<E*>(g.data_ptr()),delta.data_ptr<float>(),logz.data_ptr<float>(),stats.data_ptr<double>(),meta.data_ptr<float>(),left.data_ptr<int>(),right.data_ptr<int>(),qs?qs->data_ptr<int64_t>():nullptr,ks?ks->data_ptr<int64_t>():nullptr,dq.data_ptr<float>(),dk.data_ptr<float>(),dv.data_ptr<float>(),int(q.size(1)),int(k.size(1)),int(q.size(2)),int(k.size(2)),int(q.size(3)),int(v.size(3)),int(q.size(0)),float(scale));
   DeterministicMain<E,D,Level,false><<<dim3((k.size(1)+15)/16,k.size(0)*k.size(2)),32,0,stream>>>(reinterpret_cast<E*>(q.data_ptr()),reinterpret_cast<E*>(k.data_ptr()),reinterpret_cast<E*>(v.data_ptr()),reinterpret_cast<E*>(g.data_ptr()),delta.data_ptr<float>(),logz.data_ptr<float>(),stats.data_ptr<double>(),meta.data_ptr<float>(),left.data_ptr<int>(),right.data_ptr<int>(),qs?qs->data_ptr<int64_t>():nullptr,ks?ks->data_ptr<int64_t>():nullptr,dq.data_ptr<float>(),dk.data_ptr<float>(),dv.data_ptr<float>(),int(q.size(1)),int(k.size(1)),int(q.size(2)),int(k.size(2)),int(q.size(3)),int(v.size(3)),int(q.size(0)),float(scale));

  };
  auto levels=[&](auto bucket){if(level==2)launch(bucket,std::integral_constant<int,2>{});else if(level==3)launch(bucket,std::integral_constant<int,3>{});else launch(bucket,std::integral_constant<int,4>{});};
  int width=std::max(q.size(3),v.size(3));if(width<=64)levels(std::integral_constant<int,64>{});else if(width<=128)levels(std::integral_constant<int,128>{});else levels(std::integral_constant<int,256>{});
 });C10_CUDA_KERNEL_LAUNCH_CHECK();return {dq,dk,dv};
}
}

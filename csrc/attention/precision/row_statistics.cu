#include "attention/precision/row_statistics_sm90.cuh"
#include "attention/precision/row_statistics.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>
#include <cmath>
namespace xattn::ops::attention::precision {
namespace {
namespace wm=nvcuda::wmma;
}
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> RowStatistics(
 const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,
 const torch::Tensor& g,const torch::Tensor& left,const torch::Tensor& right,
 const c10::optional<torch::Tensor>& qs,const c10::optional<torch::Tensor>& ks,double scale,
 const c10::optional<torch::Tensor>& exact) {
 TORCH_CHECK(q.is_cuda() && q.dim()==4,"Q must be CUDA [B,N,H,D]");
 TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16,"precision statistics need FP16/BF16");
 for(const auto& t:{q,k,v,g}) TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous() && t.dim()==4 && t.scalar_type()==q.scalar_type(),"Q/K/V/dO must be contiguous on one device with matching dtype");
 TORCH_CHECK(q.size(0)==k.size(0) && k.sizes().slice(0,3)==v.sizes().slice(0,3) && q.size(3)==k.size(3) && q.sizes().slice(0,3)==g.sizes().slice(0,3) && g.size(3)==v.size(3),"incompatible attention shapes");
 TORCH_CHECK(k.size(2)>0 && q.size(2)>0 && q.size(2)%k.size(2)==0,"invalid GQA mapping");
 TORCH_CHECK(k.size(1)<(1<<24),"precision metadata requires key indices below 2^24");
 TORCH_CHECK(q.size(3)>0 && q.size(3)<=256 && q.size(3)%8==0 && v.size(3)>0 && v.size(3)<=256 && v.size(3)%8==0,"precision statistics require Q/K and V widths divisible by 8 in 8..256");
 for(const auto& t:{q,k,v,g}) TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"precision statistics require 16-byte aligned operands");
 TORCH_CHECK(std::isfinite(scale),"scale must be finite");
 for(const auto& t:{left,right}) TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.scalar_type()==at::kInt && t.is_contiguous() && t.dim()==2 && t.size(0)==q.size(0) && t.size(1)==q.size(1),"bounds must be contiguous int32 [B,Nq]");
 TORCH_CHECK(qs.has_value()==ks.has_value(),"both segment arrays are required");
 if(qs) for(int side=0;side<2;++side) {
  const auto& t=side ? *ks : *qs;int64_t n=side ? k.size(1) : q.size(1);
  TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.scalar_type()==at::kLong && t.is_contiguous() && t.dim()==2 && t.size(0)==q.size(0) && t.size(1)==n,"invalid precision segment metadata");
 }
 if(exact)TORCH_CHECK(qs && exact->is_cuda() && exact->device()==q.device() && exact->scalar_type()==at::kInt && exact->is_contiguous() && exact->numel()==q.size(0),"exact-bounds flags must be contiguous int32 [B] with segments");
 c10::cuda::CUDAGuard guard(q.device());auto opts=q.options().dtype(at::kFloat);
 auto delta=torch::empty({q.size(0),q.size(2),q.size(1)},opts);auto logz=torch::empty_like(delta);
 auto meta=torch::empty({6,q.size(0),q.size(2),q.size(1)},opts);
 if(!q.numel()) return {delta,logz,meta};
 auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
 AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(),"precision_row_statistics",[&]{
  using E=std::conditional_t<std::is_same_v<scalar_t,c10::Half>,cutlass::half_t,cutlass::bfloat16_t>;
  auto rectangular=[&](auto bucket,auto vbucket,auto tag){
    constexpr int D=decltype(bucket)::value,DV=decltype(vbucket)::value;
    constexpr bool segments=decltype(tag)::value;
    // One K/V stage when it raises CTAs per SM (D+DV between 128 and 512);
    // 64/64 and 256/256 keep the double-buffered ring.
    constexpr int Stages=D+DV>128 && D+DV<512 ? 1 : 2;
    // Two warpgroups share each K/V tile (128 query rows per CTA) when two such
    // CTAs still fit per SM; each row's arithmetic is independent of Groups.
    constexpr int Groups=sizeof(StatisticsShared<E,D,DV,Stages,2>)<=113*1024 ? 2 : 1;
    auto kernel=RowStatisticsSm90<E,D,DV,segments,Stages,Groups>;
    constexpr int smem=sizeof(StatisticsShared<E,D,DV,Stages,Groups>);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    // 1-D grid; the kernel orders query blocks heaviest first.
    const dim3 grid(unsigned(((q.size(1)+64*Groups-1)/(64*Groups))*q.size(0)*q.size(2)));
    kernel<<<grid,128*Groups,smem,stream>>>(
      reinterpret_cast<E*>(q.data_ptr()),reinterpret_cast<E*>(k.data_ptr()),reinterpret_cast<E*>(v.data_ptr()),reinterpret_cast<E*>(g.data_ptr()),left.data_ptr<int>(),right.data_ptr<int>(),qs ? qs->data_ptr<int64_t>() : nullptr,ks ? ks->data_ptr<int64_t>() : nullptr,exact ? exact->data_ptr<int>() : nullptr,delta.data_ptr<float>(),logz.data_ptr<float>(),meta.data_ptr<float>(),q.size(1),k.size(1),q.size(2),k.size(2),q.size(3),v.size(3),q.size(0),float(scale));
  };
  auto segments=[&](auto bucket,auto vbucket){if(qs)rectangular(bucket,vbucket,std::true_type{});else rectangular(bucket,vbucket,std::false_type{});};
  auto values=[&](auto bucket){int width=v.size(3);if(width<=64)segments(bucket,std::integral_constant<int,64>{});else if(width<=128)segments(bucket,std::integral_constant<int,128>{});else segments(bucket,std::integral_constant<int,256>{});};
  int width=q.size(3);if(width<=64)values(std::integral_constant<int,64>{});else if(width<=128)values(std::integral_constant<int,128>{});else values(std::integral_constant<int,256>{});

 });C10_CUDA_KERNEL_LAUNCH_CHECK();return {delta,logz,meta};
}
}

namespace xattn::ops::attention::precision {
namespace {
template<class E,int VBucket,bool Segments>
__global__ void FallbackDVKernel(
 const E* Q,const E* K,const E* G,const int* L,const int* R,
 const int64_t* QI,const int64_t* KI,const float* LOGZ,const float* META,
 const int* FLAGS,float* DV,int nq,int nk,int h,int hk,int d,int dv,int batch,float scale) {
 // CTA = (16-key tile, batch*KV head) on grid (tiles, B*Hkv), or (tiles*B*Hkv)
 // once B*Hkv exceeds grid.y's limit; both decode alike when B*Hkv == 1.
 int tile=blockIdx.x,bh=blockIdx.y;
 if(gridDim.y==1){const int tiles=(nk+15)/16;bh=tile/tiles;tile%=tiles;}
 const int b=bh/hk,kh=bh%hk;
 if(!FLAGS[bh])return;
 __shared__ __align__(32) E sa[256],sb[256];
 __shared__ __align__(32) float score[256],prob[256];
 const int lane=threadIdx.x,k0=tile*16;
 constexpr int Slots=VBucket/8;
 float accum[Slots];
 #pragma unroll
 for(int t=0;t<Slots;++t)accum[t]=0.f;
 const int64_t count=int64_t(batch)*h*nq;
 for(int head=kh*(h/hk);head<(kh+1)*(h/hk);++head) {
  for(int q0=0;q0<nq;q0+=16) {
   if(lane<32) {
    wm::fragment<wm::accumulator,16,16,16,float> sum;wm::fill_fragment(sum,0.f);
    for(int c0=0;c0<d;c0+=16) {
     for(int t=lane;t<256;t+=32) {
      int row=t/16,c=t%16;
      sa[t]=q0+row<nq && c0+c<d ? Q[((int64_t(b)*nq+q0+row)*h+head)*d+c0+c] : E{};
      sb[t]=k0+row<nk && c0+c<d ? K[((int64_t(b)*nk+k0+row)*hk+kh)*d+c0+c] : E{};
     }
     __syncwarp();
     wm::fragment<wm::matrix_a,16,16,16,E,wm::row_major> a;
     wm::fragment<wm::matrix_b,16,16,16,E,wm::col_major> bt;
     wm::load_matrix_sync(a,sa,16);wm::load_matrix_sync(bt,sb,16);wm::mma_sync(sum,a,bt,sum);__syncwarp();
    }
    wm::store_matrix_sync(score,sum,16,wm::mem_row_major);
   }
   __syncthreads();
   for(int t=lane;t<256;t+=128) {
    int i=q0+t/16,j=k0+t%16;
    bool valid=i<nq && j<nk;
    if(valid) valid=j>=L[int64_t(b)*nq+i] && j<=R[int64_t(b)*nq+i];
    if constexpr(Segments) if(valid)valid=QI[int64_t(b)*nq+i]==KI[int64_t(b)*nk+j];
    float value=0.f;
    if(valid) {
     int64_t row=(int64_t(b)*h+head)*nq+i;
     float shifted=__fmul_rn(__fsub_rn(score[t],META[row]),scale*1.4426950408889634f);
     float root=exp2f(.5f*(shifted-LOGZ[row]*1.4426950408889634f));
     root=__fmul_rn(root,sqrtf(META[2*count+row]));
     value=__fmul_rn(root,root);
    }
    prob[t]=value;
   }
   __syncthreads();
   #pragma unroll
   for(int t=0;t<Slots;++t) {
    int output=lane+t*128,key=output/dv,c=output%dv;
    if(output<16*dv && k0+key<nk) {
     for(int j=0;j<16 && q0+j<nq;++j) {
      float g=float(reinterpret_cast<const std::conditional_t<std::is_same_v<E,__half>,c10::Half,c10::BFloat16>*>(G)[((int64_t(b)*nq+q0+j)*h+head)*dv+c]);
      accum[t]=fmaf(prob[j*16+key],g,accum[t]);
     }
    }
   }
   __syncthreads();
  }
 }
 #pragma unroll
 for(int t=0;t<Slots;++t) {
  int output=lane+t*128,key=output/dv,c=output%dv;
  if(output<16*dv && k0+key<nk)DV[((int64_t(b)*nk+k0+key)*hk+kh)*dv+c]=accum[t];
 }
}
}
void FallbackDV(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& g,
    const torch::Tensor& left,const torch::Tensor& right,const torch::Tensor& logz,
    const torch::Tensor& meta,const torch::Tensor& flags,torch::Tensor dv,
    const c10::optional<torch::Tensor>& qs,const c10::optional<torch::Tensor>& ks,double scale) {
 TORCH_CHECK(q.is_cuda() && q.dim()==4 && k.dim()==4 && g.dim()==4 && dv.dim()==4,"fallback requires CUDA attention tensors");
 TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16,"fallback requires low-precision operands");
 TORCH_CHECK(k.scalar_type()==q.scalar_type() && g.scalar_type()==q.scalar_type() && dv.scalar_type()==at::kFloat,"invalid fallback dtypes");
 for(const auto& t:{q,k,g,left,right,logz,meta,flags,dv})TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous(),"fallback tensors must be contiguous on one device");
 TORCH_CHECK(q.size(0)==k.size(0) && q.size(3)==k.size(3) && q.sizes().slice(0,3)==g.sizes().slice(0,3),"incompatible fallback operands");
 TORCH_CHECK(dv.sizes().slice(0,3)==k.sizes().slice(0,3) && dv.size(3)==g.size(3) && dv.size(3)>0 && dv.size(3)<=256,"fallback supports value dimension 1..256");
 TORCH_CHECK(k.size(2)>0 && q.size(2)>0 && q.size(2)%k.size(2)==0,"invalid fallback head mapping");
 TORCH_CHECK(left.scalar_type()==at::kInt && right.scalar_type()==at::kInt && flags.scalar_type()==at::kInt && logz.scalar_type()==at::kFloat && meta.scalar_type()==at::kFloat,"invalid fallback metadata dtype");
 int64_t rows=q.size(0)*q.size(2)*q.size(1);
 TORCH_CHECK(left.numel()==q.size(0)*q.size(1) && right.sizes()==left.sizes() && logz.numel()==rows && meta.numel()==6*rows && flags.numel()==k.size(0)*k.size(2),"invalid fallback metadata sizes");
 TORCH_CHECK(qs.has_value()==ks.has_value(),"both fallback segment arrays are required");
 if(qs)for(int side=0;side<2;++side) {
  const auto& t=side ? *ks : *qs;int64_t n=side ? k.size(1) : q.size(1);
  TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous() && t.scalar_type()==at::kLong && t.numel()==q.size(0)*n,"invalid fallback segments");
 }
 TORCH_CHECK(std::isfinite(scale),"scale must be finite");
 if(!dv.numel())return;c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
 // B*Hkv rides grid.y up to its 65535 limit; beyond it the CTAs fold into x.
 const int64_t tiles=(k.size(1)+15)/16,heads=k.size(0)*k.size(2);
 TORCH_CHECK(heads<=65535 || tiles*heads<=INT_MAX,"fallback dV grid exceeds CUDA limits");
 const dim3 grid=heads<=65535 ? dim3(unsigned(tiles),unsigned(heads)) : dim3(unsigned(tiles*heads));
 AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(),"precision_fallback_dv",[&]{
  using E=std::conditional_t<std::is_same_v<scalar_t,c10::Half>,__half,__nv_bfloat16>;
  auto launch=[&](auto width,auto segment){constexpr int Width=decltype(width)::value;constexpr bool Segments=decltype(segment)::value;
   FallbackDVKernel<E,Width,Segments><<<grid,128,0,stream>>>(reinterpret_cast<E*>(q.data_ptr()),reinterpret_cast<E*>(k.data_ptr()),reinterpret_cast<E*>(g.data_ptr()),left.data_ptr<int>(),right.data_ptr<int>(),qs ? qs->data_ptr<int64_t>() : nullptr,ks ? ks->data_ptr<int64_t>() : nullptr,logz.data_ptr<float>(),meta.data_ptr<float>(),flags.data_ptr<int>(),dv.data_ptr<float>(),q.size(1),k.size(1),q.size(2),k.size(2),q.size(3),dv.size(3),q.size(0),float(scale));};
  auto dispatch=[&](auto width){if(qs)launch(width,std::true_type{});else launch(width,std::false_type{});};
  if(dv.size(3)<=32)dispatch(std::integral_constant<int,32>{});
  else if(dv.size(3)<=64)dispatch(std::integral_constant<int,64>{});
  else if(dv.size(3)<=128)dispatch(std::integral_constant<int,128>{});
  else dispatch(std::integral_constant<int,256>{});
 });C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

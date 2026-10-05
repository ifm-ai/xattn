#pragma once
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include "attention/cuda/detail/cute_compat.h"
namespace xattn::ops::attention::precision {
namespace {
using namespace cute;
__device__ __forceinline__ void StatisticsCopy16(void* destination,const void* source,bool valid){
 unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(destination));
 int bytes=valid?16:0;
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address),"l"(source),"r"(bytes));
}
__device__ __forceinline__ void StatisticsCopyWait(){
 asm volatile("cp.async.commit_group;");
 asm volatile("cp.async.wait_group 0;");
}

template<class E,int D> struct StatisticsShared {
 using Atom=decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K,E,Int<64>,Int<D>>());
 using Layout=decltype(tile_to_shape(Atom{},Shape<_64,Int<D>>{}));
 alignas(128) E q[cosize_v<Layout>],k[cosize_v<Layout>],v[cosize_v<Layout>],g[cosize_v<Layout>];
 int lower,upper;
};
template<class E,int D,bool Segments>
__global__ void RowStatisticsSm90(const E* Q,const E* K,const E* V,const E* G,
 const int* L,const int* R,const int64_t* QI,const int64_t* KI,float* Delta,float* LogZ,float* Meta,
 int nq,int nk,int h,int hk,int d,int dv,int batch,float alpha){
 extern __shared__ __align__(128) char buffer[];
 auto& s=*reinterpret_cast<StatisticsShared<E,D>*>(buffer);
 using Layout=typename StatisticsShared<E,D>::Layout;
 using Mma=decltype(make_tiled_mma(GMMA::ss_op_selector<E,E,float,Shape<_64,_64,Int<D>>,GMMA::Major::K,GMMA::Major::K>()));
 const int tid=threadIdx.x,bh=blockIdx.y,b=bh/h,head=bh%h,kh=head/(h/hk),q0=blockIdx.x*64;
 if(tid==0){s.lower=nk;s.upper=-1;}__syncthreads();
 if(tid<64 && q0+tid<nq){atomicMin(&s.lower,L[b*nq+q0+tid]);atomicMax(&s.upper,R[b*nq+q0+tid]);}
 auto sq=make_tensor(make_smem_ptr(s.q),Layout{}),sk=make_tensor(make_smem_ptr(s.k),Layout{});
 auto sv=make_tensor(make_smem_ptr(s.v),Layout{}),sg=make_tensor(make_smem_ptr(s.g),Layout{});
 for(int x=tid;x<64*D/8;x+=128){int i=x/(D/8),c=(x%(D/8))*8;
  bool validq=q0+i<nq && c+7<d,validg=q0+i<nq && c+7<dv;
  const E* qp=validq?Q+((int64_t(b)*nq+q0+i)*h+head)*d+c:Q;
  const E* gp=validg?G+((int64_t(b)*nq+q0+i)*h+head)*dv+c:G;
  StatisticsCopy16(&sq(i,c),qp,validq);StatisticsCopy16(&sg(i,c),gp,validg);
 }
 StatisticsCopyWait();__syncthreads();
 Mma mma;auto wg=mma.get_slice(_0{});auto thr=mma.get_thread_slice(tid);
 auto aq=flash::mma_partition_fragment_AB<true>(wg,sq);auto bk=flash::mma_partition_fragment_AB<false>(wg,sk);
 auto ag=flash::mma_partition_fragment_AB<true>(wg,sg);auto bv=flash::mma_partition_fragment_AB<false>(wg,sv);
 auto coord=thr.partition_C(make_identity_tensor(Shape<_64,_64>{}));
 auto rc=make_tensor(coord.data(),flash::convert_layout_acc_rowcol<false>(coord.layout()));
 constexpr int Rows=decltype(size<0>(rc))::value;
 constexpr int Cols=decltype(size<1>(rc))::value;
 float maxima[Rows],anchor_dp[Rows],normalizers[Rows],means[Rows];int anchors[Rows],left[Rows],right[Rows],first_visible[Rows],last_visible[Rows];int64_t qids[Rows];
 #pragma unroll
 for(int mi=0;mi<Rows;++mi){int qi=q0+int(get<0>(rc(mi,0)));maxima[mi]=0.f;anchor_dp[mi]=0.f;normalizers[mi]=0.f;means[mi]=0.f;anchors[mi]=-1;first_visible[mi]=nk;last_visible[mi]=-1;
  left[mi]=qi<nq?L[b*nq+qi]:nk;right[mi]=qi<nq?R[b*nq+qi]:-1;qids[mi]=0;if constexpr(Segments)if(qi<nq)qids[mi]=QI[b*nq+qi];}
 int begin=max(s.lower,0)/64*64,end=min(s.upper+1,nk);
 for(int k0=begin;k0<end;k0+=64){
  for(int x=tid;x<64*D/8;x+=128){int j=x/(D/8),c=(x%(D/8))*8;
   bool validk=k0+j<nk && c+7<d,validv=k0+j<nk && c+7<dv;
   const E* kp=validk?K+((int64_t(b)*nk+k0+j)*hk+kh)*d+c:K;
   const E* vp=validv?V+((int64_t(b)*nk+k0+j)*hk+kh)*dv+c:V;
   StatisticsCopy16(&sk(j,c),kp,validk);StatisticsCopy16(&sv(j,c),vp,validv);
  }
  StatisticsCopyWait();
  cutlass::arch::fence_view_async_shared();__syncthreads();
  auto ss=partition_fragment_C(mma,Shape<_64,_64>{});auto dd=partition_fragment_C(mma,Shape<_64,_64>{});
  flash::gemm<true,0>(mma,aq,bk,ss);flash::gemm<true,0>(mma,ag,bv,dd);
  auto scores=make_tensor(ss.data(),flash::convert_layout_acc_rowcol<false>(ss.layout()));
  auto dp=make_tensor(dd.data(),scores.layout());
  #pragma unroll
  for(int mi=0;mi<Rows;++mi){
   const int qi=q0+int(get<0>(rc(mi,0)));
   float next_max=maxima[mi],next_dp=anchor_dp[mi];int next_anchor=anchors[mi];
   #pragma unroll
   for(int ni=0;ni<Cols;++ni){int key=k0+int(get<1>(rc(mi,ni)));bool valid=qi<nq && key<nk && key>=left[mi] && key<=right[mi];
    if constexpr(Segments)if(valid)valid=KI[b*nk+key]==qids[mi];
    if(valid){first_visible[mi]=min(first_visible[mi],key);last_visible[mi]=max(last_visible[mi],key);}
    if(valid && (next_anchor<0 || scores(mi,ni)>next_max || (scores(mi,ni)==next_max && key<next_anchor))){next_max=scores(mi,ni);next_dp=dp(mi,ni);next_anchor=key;}
   }
   #pragma unroll
   for(int off=1;off<4;off*=2){float other=__shfl_xor_sync(0xffffffff,next_max,off),other_dp=__shfl_xor_sync(0xffffffff,next_dp,off);int other_anchor=__shfl_xor_sync(0xffffffff,next_anchor,off);
    if(other_anchor>=0 && (next_anchor<0 || other>next_max || (other==next_max && other_anchor<next_anchor))){next_max=other;next_dp=other_dp;next_anchor=other_anchor;}}
   if(next_anchor>=0){
    const unsigned row_mask=__activemask();
    float factor=anchors[mi]<0?0.f:exp2f(__fmul_rn(__fsub_rn(maxima[mi],next_max),alpha*0.7213475204444817f));
    float roots[Cols],partial_z=0.f;
    #pragma unroll
    for(int ni=0;ni<Cols;++ni){int key=k0+int(get<1>(rc(mi,ni)));bool valid=qi<nq && key<nk && key>=left[mi] && key<=right[mi];
     if constexpr(Segments)if(valid)valid=KI[b*nk+key]==qids[mi];
     roots[ni]=valid?exp2f(__fmul_rn(__fsub_rn(scores(mi,ni),next_max),alpha*0.7213475204444817f)):0.f;partial_z+=roots[ni]*roots[ni];}
    #pragma unroll
    for(int off=1;off<4;off*=2)partial_z+=__shfl_xor_sync(row_mask,partial_z,off);
    float nz=(normalizers[mi]*factor)*factor+partial_z,inv_root=rsqrtf(nz),partial=0.f;
    #pragma unroll
    for(int ni=0;ni<Cols;++ni){float root=roots[ni]*inv_root;if(root>0.f)partial+=root*(root*(dp(mi,ni)-next_dp));}
    #pragma unroll
    for(int off=1;off<4;off*=2)partial+=__shfl_xor_sync(row_mask,partial,off);
    means[mi]=factor*(factor*((normalizers[mi]/nz)*(means[mi]+(anchor_dp[mi]-next_dp))))+partial;
    normalizers[mi]=nz;maxima[mi]=next_max;anchor_dp[mi]=next_dp;anchors[mi]=next_anchor;
   }
  }
  __syncthreads();
 }
 // Bounds for invariant guards must describe admitted keys, including segment
 // filtering, rather than the larger causal/window interval.
 #pragma unroll
 for(int mi=0;mi<Rows;++mi){
  #pragma unroll
  for(int off=1;off<4;off*=2){
   first_visible[mi]=min(first_visible[mi],__shfl_xor_sync(0xffffffff,first_visible[mi],off));
   last_visible[mi]=max(last_visible[mi],__shfl_xor_sync(0xffffffff,last_visible[mi],off));
  }
 }
 if((tid&3)==0){
  #pragma unroll
  for(int mi=0;mi<Rows;++mi){int qi=q0+int(get<0>(rc(mi,0)));if(qi<nq){int64_t count=int64_t(batch)*h*nq,idx=int64_t(bh)*nq+qi;
   Delta[idx]=normalizers[mi]>0.f?means[mi]:0.f;LogZ[idx]=normalizers[mi]>0.f?logf(normalizers[mi]):INFINITY;
   Meta[idx]=maxima[mi];Meta[count+idx]=anchor_dp[mi];Meta[2*count+idx]=1.f;Meta[3*count+idx]=float(anchors[mi]);Meta[4*count+idx]=float(first_visible[mi]);Meta[5*count+idx]=float(last_visible[mi]);}}
 }

}
} // namespace
} // namespace xattn::ops::attention::precision

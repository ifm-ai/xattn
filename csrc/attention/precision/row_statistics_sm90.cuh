#pragma once
#include <climits>
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
__device__ __forceinline__ void StatisticsCommit(){asm volatile("cp.async.commit_group;");}
template<int N> __device__ __forceinline__ void StatisticsWait(){asm volatile("cp.async.wait_group %0;" :: "n"(N));}

// Q/dO stay resident; K/V tiles of 64 keys use a Stages-deep cp.async ring.
// Separate D and DV layouts avoid padded work when the widths differ.
template<class E,int D,int DV,int Stages,int Groups=1> struct StatisticsShared {
 using Atom=decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K,E,Int<64>,Int<D>>());
 using Layout=decltype(tile_to_shape(Atom{},Shape<_64,Int<D>>{}));
 using AtomV=decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K,E,Int<64>,Int<DV>>());
 using LayoutV=decltype(tile_to_shape(AtomV{},Shape<_64,Int<DV>>{}));
 alignas(128) E q[Groups][cosize_v<Layout>];
 alignas(128) E g[Groups][cosize_v<LayoutV>];
 alignas(128) E k[Stages][cosize_v<Layout>];
 alignas(128) E v[Stages][cosize_v<LayoutV>];
 int lower,upper;
};

// Each of the Groups warpgroups owns one 64-row query block; they share the
// CTA's K/V tiles. The 1-D grid enumerates blocks heaviest first (query block
// descending, heads interleaved) so causal tails finish early. Tiles fully
// inside a row's [left,right] skip mask tests; rows with no admitted key in a
// tile skip it. Exact[b] != 0 states that [left,right] already equals each
// row's segment support, so batch b skips the per-key segment test.
template<class E,int D,int DV,bool Segments,int Stages,int Groups=1>
__global__ void __launch_bounds__(128*Groups,Groups==2?2:1) RowStatisticsSm90(const E* Q,const E* K,const E* V,const E* G,
 const int* L,const int* R,const int64_t* QI,const int64_t* KI,const int* Exact,float* Delta,float* LogZ,float* Meta,
 int nq,int nk,int h,int hk,int d,int dv,int batch,float alpha){
 extern __shared__ __align__(128) char buffer[];
 using Shared=StatisticsShared<E,D,DV,Stages,Groups>;
 auto& s=*reinterpret_cast<Shared*>(buffer);
 using Layout=typename Shared::Layout;
 using LayoutV=typename Shared::LayoutV;
 using Mma=decltype(make_tiled_mma(GMMA::ss_op_selector<E,E,float,Shape<_64,_64,Int<D>>,GMMA::Major::K,GMMA::Major::K>()));
 using MmaV=decltype(make_tiled_mma(GMMA::ss_op_selector<E,E,float,Shape<_64,_64,Int<DV>>,GMMA::Major::K,GMMA::Major::K>()));
 // Groups warpgroups own consecutive 64-row blocks and share each K/V tile.
 const int heads=batch*h,qblocks=(nq+64*Groups-1)/(64*Groups),linear=blockIdx.x;
 const int tid=threadIdx.x,group=tid/128,ltid=tid%128,bh=linear%heads,b=bh/h,head=bh%h,kh=head/(h/hk);
 const int cta_q0=(qblocks-1-linear/heads)*64*Groups,q0=cta_q0+group*64;
 const bool segments=Segments && !(Exact && Exact[b]);
 if(tid==0){s.lower=nk;s.upper=-1;}__syncthreads();
 if(tid<64*Groups && cta_q0+tid<nq){atomicMin(&s.lower,L[b*nq+cta_q0+tid]);atomicMax(&s.upper,R[b*nq+cta_q0+tid]);}
 auto sq=make_tensor(make_smem_ptr(s.q[group]),Layout{});auto sg=make_tensor(make_smem_ptr(s.g[group]),LayoutV{});
 for(int x=ltid;x<64*D/8;x+=128){int i=x/(D/8),c=(x%(D/8))*8;bool valid=q0+i<nq && c+7<d;
  StatisticsCopy16(&sq(i,c),valid?Q+((int64_t(b)*nq+q0+i)*h+head)*d+c:Q,valid);}
 for(int x=ltid;x<64*DV/8;x+=128){int i=x/(DV/8),c=(x%(DV/8))*8;bool valid=q0+i<nq && c+7<dv;
  StatisticsCopy16(&sg(i,c),valid?G+((int64_t(b)*nq+q0+i)*h+head)*dv+c:G,valid);}
 StatisticsCommit();
 __syncthreads();
 const int begin=max(s.lower,0)/64*64,end=min(s.upper+1,nk);
 // Each thread copies fixed (row, column) chunks; only the key offset moves.
 constexpr int Threads=128*Groups,ChunksK=64*D/8/Threads,RowsK=Threads/(D/8),ChunksV=64*DV/8/Threads,RowsV=Threads/(DV/8);
 const int jk=tid/(D/8),ck=(tid%(D/8))*8,jv=tid/(DV/8),cv=(tid%(DV/8))*8;
 const bool col_k=ck+7<d,col_v=cv+7<dv;
 const E* const kbase=K+(int64_t(b)*nk*hk+kh)*d+ck;
 const E* const vbase=V+(int64_t(b)*nk*hk+kh)*dv+cv;
 auto load_kv=[&](int k0,int stage){
  auto sk=make_tensor(make_smem_ptr(s.k[stage]),Layout{});auto sv=make_tensor(make_smem_ptr(s.v[stage]),LayoutV{});
  #pragma unroll
  for(int t=0;t<ChunksK;++t){int j=jk+t*RowsK;bool valid=k0+j<nk && col_k;
   StatisticsCopy16(&sk(j,ck),valid?kbase+int64_t(k0+j)*hk*d:K,valid);}
  #pragma unroll
  for(int t=0;t<ChunksV;++t){int j=jv+t*RowsV;bool valid=k0+j<nk && col_v;
   StatisticsCopy16(&sv(j,cv),valid?vbase+int64_t(k0+j)*hk*dv:V,valid);}
 };
 #pragma unroll
 for(int t=0;t<Stages-1;++t){if(begin+t*64<end)load_kv(begin+t*64,t);StatisticsCommit();}
 Mma mma;auto wg=mma.get_slice(_0{});auto thr=mma.get_thread_slice(ltid);
 MmaV mma_v;auto wg_v=mma_v.get_slice(_0{});
 auto aq=flash::mma_partition_fragment_AB<true>(wg,sq);
 auto ag=flash::mma_partition_fragment_AB<true>(wg_v,sg);
 auto coord=thr.partition_C(make_identity_tensor(Shape<_64,_64>{}));
 auto rc=make_tensor(coord.data(),flash::convert_layout_acc_rowcol<false>(coord.layout()));
 constexpr int Rows=decltype(size<0>(rc))::value;
 constexpr int Cols=decltype(size<1>(rc))::value;
 float maxima[Rows],anchor_dp[Rows],normalizers[Rows],means[Rows];int anchors[Rows],left[Rows],right[Rows],first_visible[Rows],last_visible[Rows];int64_t qids[Rows];
 #pragma unroll
 for(int mi=0;mi<Rows;++mi){int qi=q0+int(get<0>(rc(mi,0)));maxima[mi]=0.f;anchor_dp[mi]=0.f;normalizers[mi]=0.f;means[mi]=0.f;anchors[mi]=-1;first_visible[mi]=nk;last_visible[mi]=-1;
  left[mi]=qi<nq?L[b*nq+qi]:nk;right[mi]=qi<nq?R[b*nq+qi]:-1;qids[mi]=0;if constexpr(Segments)if(qi<nq)qids[mi]=QI[b*nq+qi];}
 int stage=0;
 for(int k0=begin;k0<end;k0+=64){
  {int next=k0+(Stages-1)*64,ns=(stage+Stages-1)%Stages;if(next<end)load_kv(next,ns);StatisticsCommit();}
  StatisticsWait<Stages-1>();
  cutlass::arch::fence_view_async_shared();__syncthreads();
  auto sk=make_tensor(make_smem_ptr(s.k[stage]),Layout{});auto sv=make_tensor(make_smem_ptr(s.v[stage]),LayoutV{});
  auto bk=flash::mma_partition_fragment_AB<false>(wg,sk);auto bv=flash::mma_partition_fragment_AB<false>(wg_v,sv);
  auto ss=partition_fragment_C(mma,Shape<_64,_64>{});auto dd=partition_fragment_C(mma_v,Shape<_64,_64>{});
  flash::gemm<true,-1>(mma,aq,bk,ss);flash::gemm<true,-1>(mma_v,ag,bv,dd);
  warpgroup_wait<0>();
  auto scores=make_tensor(ss.data(),flash::convert_layout_acc_rowcol<false>(ss.layout()));
  auto dp=make_tensor(dd.data(),scores.layout());
  #pragma unroll
  for(int mi=0;mi<Rows;++mi){
   const int qi=q0+int(get<0>(rc(mi,0)));
   // Row status is uniform over the 4-lane quad that owns this row. Masks
   // come from warp-wide ballots so each shuffle names exactly its lanes.
   const bool empty=qi>=nq || k0>right[mi] || k0+63<left[mi];
   const bool full=!empty && !segments && k0>=left[mi] && k0+63<=right[mi] && k0+63<nk;
   const unsigned full_mask=__ballot_sync(0xffffffffu,full);
   const unsigned part_mask=__ballot_sync(0xffffffffu,!empty && !full);
   float next_max=maxima[mi],next_dp=anchor_dp[mi];int next_anchor=anchors[mi];
   if(full){
    float tile_max=scores(mi,0);
    #pragma unroll
    for(int ni=1;ni<Cols;++ni)tile_max=fmaxf(tile_max,scores(mi,ni));
    #pragma unroll
    for(int off=1;off<4;off*=2)tile_max=fmaxf(tile_max,__shfl_xor_sync(full_mask,tile_max,off));
    first_visible[mi]=min(first_visible[mi],k0);last_visible[mi]=max(last_visible[mi],k0+63);
    // Earlier tiles hold smaller keys, so equality keeps the earlier anchor.
    const bool better=next_anchor<0 || tile_max>next_max;
    const unsigned better_mask=__ballot_sync(full_mask,better);
    if(better){
     int key=INT_MAX;float kdp=0.f;
     #pragma unroll
     for(int ni=Cols-1;ni>=0;--ni)if(scores(mi,ni)==tile_max){key=k0+int(get<1>(rc(mi,ni)));kdp=dp(mi,ni);}
     #pragma unroll
     for(int off=1;off<4;off*=2){int ok=__shfl_xor_sync(better_mask,key,off);float od=__shfl_xor_sync(better_mask,kdp,off);if(ok<key){key=ok;kdp=od;}}
     if(key!=INT_MAX){next_max=tile_max;next_dp=kdp;next_anchor=key;}
    }
   } else if(!empty) {
    #pragma unroll
    for(int ni=0;ni<Cols;++ni){int key=k0+int(get<1>(rc(mi,ni)));bool valid=key<nk && key>=left[mi] && key<=right[mi];
     if constexpr(Segments)if(valid && segments)valid=KI[b*nk+key]==qids[mi];
     if(valid){first_visible[mi]=min(first_visible[mi],key);last_visible[mi]=max(last_visible[mi],key);}
     if(valid && (next_anchor<0 || scores(mi,ni)>next_max || (scores(mi,ni)==next_max && key<next_anchor))){next_max=scores(mi,ni);next_dp=dp(mi,ni);next_anchor=key;}
    }
    #pragma unroll
    for(int off=1;off<4;off*=2){float other=__shfl_xor_sync(part_mask,next_max,off),other_dp=__shfl_xor_sync(part_mask,next_dp,off);int other_anchor=__shfl_xor_sync(part_mask,next_anchor,off);
     if(other_anchor>=0 && (next_anchor<0 || other>next_max || (other==next_max && other_anchor<next_anchor))){next_max=other;next_dp=other_dp;next_anchor=other_anchor;}}
   }
   const unsigned row_mask=__ballot_sync(0xffffffffu,!empty && next_anchor>=0);
   if(!empty && next_anchor>=0){
    float factor=anchors[mi]<0?0.f:exp2f(__fmul_rn(__fsub_rn(maxima[mi],next_max),alpha*0.7213475204444817f));
    float roots[Cols],partial_z=0.f;
    if(full){
     #pragma unroll
     for(int ni=0;ni<Cols;++ni){roots[ni]=exp2f(__fmul_rn(__fsub_rn(scores(mi,ni),next_max),alpha*0.7213475204444817f));partial_z+=roots[ni]*roots[ni];}
    } else {
     #pragma unroll
     for(int ni=0;ni<Cols;++ni){int key=k0+int(get<1>(rc(mi,ni)));bool valid=key<nk && key>=left[mi] && key<=right[mi];
      if constexpr(Segments)if(valid && segments)valid=KI[b*nk+key]==qids[mi];
      roots[ni]=valid?exp2f(__fmul_rn(__fsub_rn(scores(mi,ni),next_max),alpha*0.7213475204444817f)):0.f;partial_z+=roots[ni]*roots[ni];}
    }
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
  stage=(stage+1)%Stages;
 }
 StatisticsWait<0>();
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

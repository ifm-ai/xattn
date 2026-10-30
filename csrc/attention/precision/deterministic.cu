#include "attention/precision/deterministic.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
namespace xattn::ops::attention::precision {
namespace {
constexpr int kChunk=64;
// (chunk or key, batch*KV head) of a repair CTA. The grid is (n, b*hk), or
// (n*b*hk) once b*hk exceeds grid.y's limit; both decode alike when b*hk == 1.
__device__ __forceinline__ int2 RepairBlock(int n){
 int x=blockIdx.x,bh=blockIdx.y;
 if(gridDim.y==1){bh=x/n;x%=n;}
 return make_int2(x,bh);
}
// One warp per (batch, query, head) row: the anchor share of dQ.
template<class E>
__global__ void RepairQuery(const E* K,const double* stats,const float* meta,float* dq,
 int64_t queries,int nq,int nk,int h,int hk,int d,int batch,float scale){
 const int64_t query=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;if(query>=queries)return;
 const int lane=threadIdx.x%32,head=query%h,i=query/h%nq,b=query/(int64_t(h)*nq);
 const int64_t row=(int64_t(b)*h+head)*nq+i,count=int64_t(batch)*h*nq;const int anchor=int(meta[3*count+row]);
 if(anchor<0 || anchor>=nk)return;
 const float coefficient=scale*float(stats[row]);
 const E* key=K+((int64_t(b)*nk+anchor)*hk+head/(h/hk))*d;
 for(int c=lane;c<d;c+=32)dq[query*d+c]-=coefficient*float(key[c]);
}
// Rows of one KV head sorted (stably) by anchor key are cut into fixed chunks
// of kChunk positions. Each chunk sums its runs of equal keys in order: a run
// holding all rows of its key updates dK/dV directly; a run continuing from
// the previous chunk goes to head[chunk], one continuing into the next chunk
// to tail[chunk]. RepairSpans then adds tail, heads in chunk order. Fixed
// chunking makes the result independent of scheduling, and a sink key's rows
// are reduced by many chunks in parallel.
template<class E>
__global__ void RepairChunks(const E* Q,const E* G,const double* stats,const float* meta,
 const int64_t* order,const int64_t* starts,float* dk,float* dv,float* head,float* tail,int* flags,
 int nq,int nk,int h,int hk,int d,int vd,int batch,float scale){
 __shared__ int keys[kChunk];__shared__ int64_t qoff[kChunk],goff[kChunk];__shared__ float coef[kChunk],mass[kChunk];__shared__ bool safe_mass[kChunk];
 const int rows=nq*(h/hk),chunks=(rows+kChunk-1)/kChunk,width=max(d,vd);
 const int2 block=RepairBlock(chunks);const int chunk=block.x,bh=block.y,b=bh/hk,kh=bh%hk,c=threadIdx.x;
 const int64_t count=int64_t(batch)*h*nq,base=int64_t(bh)*rows,p0=int64_t(chunk)*kChunk;
 const int n=int(rows-p0<kChunk ? rows-p0 : int64_t(kChunk));
 if(c<n){
  const int64_t r=order[base+p0+c];const int head_index=kh*(h/hk)+r/nq,i=r%nq;
  const int64_t row=(int64_t(b)*h+head_index)*nq+i;
  const int key=int(meta[3*count+row]);
  keys[c]=key>=0 && key<nk ? key : -1;
  qoff[c]=((int64_t(b)*nq+i)*h+head_index)*d;goff[c]=((int64_t(b)*nq+i)*h+head_index)*vd;
  coef[c]=scale*float(stats[row]);mass[c]=0.f;safe_mass[c]=false;
  if(keys[c]>=0){double m=stats[count+row],pa=stats[3*count+row];float delta=float(m-1.);
   bool safe=isfinite(m) && m>0. && isfinite(pa) && pa>=0. && (delta<=0.f || double(delta)<=pa*(1.-2e-6));
   if(!safe)atomicExch(flags+bh,1);else{mass[c]=delta;safe_mass[c]=true;}}
 }
 __syncthreads();
 float ck=0.f,cv=0.f;int start=0;
 for(int p=0;p<n;++p){
  const int key=keys[p];
  if(key>=0){
   if(c<d)ck+=coef[p]*float(Q[qoff[p]+c]);
   if(c<vd && safe_mass[p])cv+=mass[p]*float(G[goff[p]+c]);
  }
  if(p+1<n && keys[p+1]==key)continue;
  if(key>=0){
   const int64_t lo=starts[int64_t(bh)*(nk+1)+key],hi=starts[int64_t(bh)*(nk+1)+key+1];
   const bool continued=start==0 && lo<p0,continues=p+1==n && hi>p0+n;
   const int64_t slot=(int64_t(bh)*chunks+chunk)*width;
   if(continued){if(c<d)head[2*slot+c]=ck;if(c<vd)head[2*slot+width+c]=cv;}
   else if(continues){if(c<d)tail[2*slot+c]=ck;if(c<vd)tail[2*slot+width+c]=cv;}
   else{if(c<d)dk[((int64_t(b)*nk+key)*hk+kh)*d+c]-=ck;if(c<vd)dv[((int64_t(b)*nk+key)*hk+kh)*vd+c]-=cv;}
  }
  ck=0.f;cv=0.f;start=p+1;
 }
}
template<class E>
__global__ void RepairSpans(const int64_t* starts,const float* head,const float* tail,float* dk,float* dv,
 int nk,int hk,int d,int vd,int chunks){
 const int2 block=RepairBlock(nk);const int key=block.x,bh=block.y,b=bh/hk,kh=bh%hk,c=threadIdx.x,width=max(d,vd);
 const int64_t lo=starts[int64_t(bh)*(nk+1)+key],hi=starts[int64_t(bh)*(nk+1)+key+1];
 if(hi-lo<1 || lo/kChunk==(hi-1)/kChunk)return;
 const int64_t first=lo/kChunk,last=(hi-1)/kChunk,slot=int64_t(bh)*chunks;
 float ck=c<d?tail[2*(slot+first)*width+c]:0.f,cv=c<vd?tail[2*(slot+first)*width+width+c]:0.f;
 for(int64_t x=first+1;x<=last;++x){
  if(c<d)ck+=head[2*(slot+x)*width+c];
  if(c<vd)cv+=head[2*(slot+x)*width+width+c];
 }
 if(c<d)dk[((int64_t(b)*nk+key)*hk+kh)*d+c]-=ck;
 if(c<vd)dv[((int64_t(b)*nk+key)*hk+kh)*vd+c]-=cv;
}
}
torch::Tensor DeterministicRepair(const torch::Tensor& q,const torch::Tensor& k,
 const torch::Tensor& g,const torch::Tensor& stats,const torch::Tensor& meta,const torch::Tensor& order,
 torch::Tensor dq,torch::Tensor dk,torch::Tensor dv,double scale){
 c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
 TORCH_CHECK(order.is_cuda() && order.device()==q.device() && order.scalar_type()==at::kLong && order.is_contiguous(),"invalid deterministic anchor ordering");
 auto flags=torch::zeros({q.size(0),k.size(2)},q.options().dtype(at::kInt));
 if(!q.numel())return flags;
 const int64_t b=q.size(0),hk=k.size(2),rows=q.size(1)*(q.size(2)/hk),nk=k.size(1);
 TORCH_CHECK(order.numel()==b*hk*rows,"invalid deterministic anchor ordering");
 TORCH_CHECK(q.size(3)<=256 && g.size(3)<=256,"deterministic repair supports D/V up to 256");
 // Per-key ranges of the stable anchor order (one batched search).
 auto anchors=meta.select(0,3).reshape({b,hk,rows}).gather(-1,order.reshape({b,hk,rows}));
 auto keys=at::arange(nk+1,anchors.options()).expand({b,hk,nk+1}).contiguous();
 auto starts=at::searchsorted(anchors,keys).contiguous();
 const int chunks=int((rows+kChunk-1)/kChunk),width=int(std::max(q.size(3),g.size(3)));
 auto head=torch::empty({b*hk,chunks,2,width},q.options().dtype(at::kFloat)),tail=torch::empty_like(head);
 // b*hk rides grid.y up to its 65535 limit; beyond it the CTAs fold into x.
 const bool fold=b*hk>65535;
 TORCH_CHECK(!fold || std::max<int64_t>(chunks,nk)*b*hk<=INT_MAX,"deterministic repair grid exceeds CUDA limits");
 auto grid=[&](int64_t n){return fold ? dim3(unsigned(n*b*hk)) : dim3(unsigned(n),unsigned(b*hk));};
 AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(),"precision_deterministic_repair",[&]{
  const int64_t queries=q.numel()/q.size(3);
  RepairQuery<scalar_t><<<(queries+7)/8,256,0,stream>>>(k.data_ptr<scalar_t>(),stats.data_ptr<double>(),meta.data_ptr<float>(),dq.data_ptr<float>(),queries,q.size(1),k.size(1),q.size(2),k.size(2),q.size(3),q.size(0),float(scale));
  RepairChunks<scalar_t><<<grid(chunks),256,0,stream>>>(q.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),stats.data_ptr<double>(),meta.data_ptr<float>(),order.data_ptr<int64_t>(),starts.data_ptr<int64_t>(),dk.data_ptr<float>(),dv.data_ptr<float>(),head.data_ptr<float>(),tail.data_ptr<float>(),flags.data_ptr<int>(),q.size(1),k.size(1),q.size(2),k.size(2),q.size(3),g.size(3),q.size(0),float(scale));
  RepairSpans<scalar_t><<<grid(nk),256,0,stream>>>(starts.data_ptr<int64_t>(),head.data_ptr<float>(),tail.data_ptr<float>(),dk.data_ptr<float>(),dv.data_ptr<float>(),k.size(1),k.size(2),q.size(3),g.size(3),chunks);
 });C10_CUDA_KERNEL_LAUNCH_CHECK();return flags;
}
}

#include "attention/precision/main.h"
#include "attention/precision/rectangular.h"
#include "attention/precision/mha_direct.h"
#include "attention/hopper/bwd_params.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
namespace xattn::ops::attention::precision {
#define DECL(D,T,V) void Main_##D##_##T##_##V(AttentionBwdParams,cudaStream_t);
#define VIS(D,T) DECL(D,T,0) DECL(D,T,1) DECL(D,T,2)
VIS(64,bf16) VIS(128,bf16) VIS(256,bf16)
VIS(64,fp16) VIS(128,fp16) VIS(256,fp16)
#undef VIS
#undef DECL
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> MainBackward(
 const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,
 const torch::Tensor& g,const torch::Tensor& delta,const torch::Tensor& logz,
 const torch::Tensor& stats,const torch::Tensor& meta,
 const c10::optional<torch::Tensor>& qs,const c10::optional<torch::Tensor>& ks,
 const c10::optional<torch::Tensor>& qp,double scale,int level,int visibility,int span,bool strict,bool reset) {
 TORCH_CHECK(level>=2 && level<=4,"precision mainloop needs level 2, 3, or 4");
 TORCH_CHECK(visibility>=0 && visibility<=2,"invalid attention visibility");
 TORCH_CHECK(q.is_cuda() && q.dim()==4,"Q must be CUDA [B,N,H,D]");
 TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16,"precision mainloop requires FP16/BF16");
 for(const auto& t:{q,k,v,g})TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.dim()==4 && t.is_contiguous() && t.scalar_type()==q.scalar_type(),"Q/K/V/dO must be contiguous tensors of one dtype/device");
 TORCH_CHECK(q.size(0)==k.size(0) && k.sizes().slice(0,3)==v.sizes().slice(0,3) && q.sizes().slice(0,3)==g.sizes().slice(0,3) && q.size(3)==k.size(3) && g.size(3)==v.size(3),"incompatible precision operands");
 TORCH_CHECK(k.size(2)>0 && q.size(2)>0 && q.size(2)%k.size(2)==0,"invalid GQA mapping");
 TORCH_CHECK(q.size(3)>0 && q.size(3)<=256 && v.size(3)>0 && v.size(3)<=256 && q.size(3)%8==0 && v.size(3)%8==0,"precision mainloop requires D/V multiples of 8 up to 256");
 TORCH_CHECK(std::isfinite(scale) && scale>0.,"precision mainloop requires a positive finite scale");
 TORCH_CHECK(visibility==0 || (visibility==1 ? span>=0 : span>0),"invalid local attention span");
 int64_t count=q.size(0)*q.size(2)*q.size(1);
 for(const auto& t:{delta,logz,stats,meta})TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous(),"precision workspace must be contiguous on Q's device");
 TORCH_CHECK(delta.scalar_type()==at::kFloat && logz.scalar_type()==at::kFloat && meta.scalar_type()==at::kFloat && stats.scalar_type()==at::kDouble,"invalid precision workspace dtype");
 TORCH_CHECK(delta.numel()==count && logz.numel()==count && meta.numel()==6*count && stats.numel()==4*count,"invalid precision workspace shape");
 TORCH_CHECK(qs.has_value()==ks.has_value(),"both segment arrays are required");
 if(qs)for(int side=0;side<2;++side) {
  const auto& t=side ? *ks : *qs;int64_t n=side ? k.size(1) : q.size(1);
  TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous() && t.scalar_type()==at::kLong && t.dim()==2 && t.size(0)==q.size(0) && t.size(1)==n,"invalid mainloop segments");
 }
 if(qp)TORCH_CHECK(qp->is_cuda() && qp->device()==q.device() && qp->is_contiguous() && qp->scalar_type()==at::kInt && qp->numel()==q.size(0)*q.size(1),"invalid reset query positions");
 TORCH_CHECK(!reset || visibility==2,"reset positions only apply to sliding chunks");
 c10::cuda::CUDAGuard guard(q.device());
 // HP4 rectangular kernels are retained only in the measured favorable region.
 auto dimension_bucket=[](int64_t d){return d<=64?64:d<=128?128:256;};
 const int db=dimension_bucket(q.size(3)),vb=dimension_bucket(v.size(3));
 const bool use_rectangular=level<=3 || db==256;
 if(q.size(2)==k.size(2) && (db==vb || use_rectangular)) {
   return mha_direct::MainBackward(q,k,v,g,delta,logz,stats,meta,qs,ks,qp,
                                  scale,level,visibility,span,strict,reset);
 }
 if(db!=vb && use_rectangular) {
   return rectangular::MainBackward(q,k,v,g,delta,logz,stats,meta,qs,ks,qp,
                                   scale,level,visibility,span,strict,reset);
 }
 int max_dimension=std::max(q.size(3),v.size(3));
 int storage_dimension=max_dimension<=64 ? 64 : max_dimension<=128 ? 128 : 256;
 // GQA epilogues can write the complete template bucket even for partial D/V.
 // Keep all accumulators padded; trim only after every kernel has finished.
 auto dq=torch::empty({q.size(0),q.size(1),q.size(2),storage_dimension},q.options().dtype(at::kFloat));
 auto dk=torch::empty({k.size(0),k.size(1),k.size(2),storage_dimension},q.options().dtype(at::kFloat));
 auto dv=torch::empty({v.size(0),v.size(1),v.size(2),storage_dimension},q.options().dtype(at::kFloat));
 if(!q.numel()){return {torch::zeros(q.sizes(),q.options().dtype(at::kFloat)),torch::zeros(k.sizes(),q.options().dtype(at::kFloat)),torch::zeros(v.sizes(),q.options().dtype(at::kFloat))};}
 AttentionBwdParams p{};p.q_ptr=q.data_ptr();p.k_ptr=k.data_ptr();p.v_ptr=v.data_ptr();p.dy_ptr=g.data_ptr();
 p.lse_ptr=logz.data_ptr();p.dq_ptr=dq.data_ptr();p.dk_ptr=dk.data_ptr();p.dv_ptr=dv.data_ptr();
 p.hp_delta=delta.data_ptr<float>();p.hp_rho=reinterpret_cast<float*>(stats.data_ptr());p.hp_mass=meta.data_ptr<float>();p.high_precision_level=level;
 p.b=q.size(0);p.num_sequences=p.b;p.seqlen_q=q.size(1);p.seqlen_k=k.size(1);p.max_seqlen_q=p.seqlen_q;p.max_seqlen_k=p.seqlen_k;
 p.d=q.size(3);p.dv=v.size(3);p.h=q.size(2);p.h_k=k.size(2);p.scale_softmax=float(scale);
 p.window_size_left=visibility==0 ? p.seqlen_k : visibility==1 ? span : 2*span-1;
 p.window_size_right=strict ? -1 : 0;p.attention_chunk=visibility==2 && !(reset && qs.has_value()) ? span : 0;
 p.y_is_fp32=true;p.num_sm=at::cuda::getDeviceProperties(q.get_device())->multiProcessorCount;
 if(qs){p.q_segment_idx=qs->data_ptr<int64_t>();p.k_segment_idx=ks->data_ptr<int64_t>();p.k_segment_len=p.seqlen_k;}
 p.q_chunk_positions=qp ? qp->data_ptr<int>() : nullptr;p.reset_attention_chunk=reset && qs.has_value() ? span : 0;
#define STRIDE(NAME,TENSOR) p.NAME##_batch_stride=TENSOR.stride(0);p.NAME##_row_stride=TENSOR.stride(1);p.NAME##_head_stride=TENSOR.stride(2);
 STRIDE(q,q) STRIDE(k,k) STRIDE(v,v) STRIDE(dy,g) STRIDE(dq,dq) STRIDE(dk,dk) STRIDE(dv,dv)
#undef STRIDE
 auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
 using Fn=void(*)(AttentionBwdParams,cudaStream_t);
 static const Fn bf16[3][3]={{Main_64_bf16_0,Main_64_bf16_1,Main_64_bf16_2},{Main_128_bf16_0,Main_128_bf16_1,Main_128_bf16_2},{Main_256_bf16_0,Main_256_bf16_1,Main_256_bf16_2}};
 static const Fn fp16[3][3]={{Main_64_fp16_0,Main_64_fp16_1,Main_64_fp16_2},{Main_128_fp16_0,Main_128_fp16_1,Main_128_fp16_2},{Main_256_fp16_0,Main_256_fp16_1,Main_256_fp16_2}};
 int maxdim=std::max(p.d,p.dv),bucket=maxdim<=64 ? 0 : maxdim<=128 ? 1 : 2;
 (q.scalar_type()==at::kBFloat16 ? bf16 : fp16)[bucket][visibility](p,stream);
 return {dq.narrow(3,0,p.d).contiguous(),dk.narrow(3,0,p.d).contiguous(),dv.narrow(3,0,p.dv).contiguous()};
}
}

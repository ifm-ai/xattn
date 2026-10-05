#include "attention/precision/preprocess.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
namespace xattn::ops::attention::precision {
namespace {
template <typename E>
__global__ void CenterKernel(const E* x, E* y, float* shifts, int n, int h, int d) {
  __shared__ int bad[8][32];
  const int lane=threadIdx.x, part=threadIdx.y;
  const int feature=blockIdx.x*32+lane, b=blockIdx.y;
  const int width=h*d;
  const int64_t base=int64_t(b)*n*width+feature;
  const float shift=feature<width && n>0 ? float(x[base]) : 0.f;
  int invalid=!isfinite(shift);
  if (feature<width) {
    for (int i=part;i<n;i+=8) {
      float original=float(x[base+int64_t(i)*width]);
      E residual=E(original-shift);
      // Compare in FP64, so adding a large shift cannot hide a lost small value.
      invalid |= double(float(residual))+double(shift)!=double(original);
    }
  }
  bad[part][lane]=invalid;
  __syncthreads();
  for (int k=0;k<8;++k) invalid |= bad[k][lane];
  const float selected=invalid ? 0.f : shift;
  if (part==0 && feature<width) shifts[int64_t(b)*width+feature]=selected;
  if (feature<width) for (int i=part;i<n;i+=8) {
    const int64_t offset=base+int64_t(i)*width;
    y[offset]=invalid ? x[offset] : E(float(x[offset])-selected);
  }
}
template <typename E>
__global__ void CheckCenterTiles(const E* x,int* flags,int n,int width) {
  __shared__ int bad[8][32];
  int lane=threadIdx.x,part=threadIdx.y,feature=blockIdx.x*32+lane,b=blockIdx.y;
  int begin=blockIdx.z*128;
  float shift=feature<width ? float(x[int64_t(b)*n*width+feature]) : 0.f;
  int invalid=!isfinite(shift);
  if(feature<width) for(int i=begin+part;i<min(n,begin+128);i+=8) {
    float original=float(x[(int64_t(b)*n+i)*width+feature]);
    E residual=E(original-shift);
    invalid |= double(float(residual))+double(shift)!=double(original);
  }
  bad[part][lane]=invalid;__syncthreads();
  if(part==0 && feature<width) {
    for(int j=0;j<8;++j)invalid |= bad[j][lane];
    if(invalid)atomicExch(flags+int64_t(b)*width+feature,1);
  }
}
template <typename E>
__global__ void ApplyCenter(const E* x,const int* flags,E* y,float* shifts,int64_t count,int n,int width) {
  int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(idx>=count)return;
  int feature=idx%width,b=idx/(int64_t(n)*width);
  float shift=flags[int64_t(b)*width+feature] ? 0.f : float(x[int64_t(b)*n*width+feature]);
  y[idx]=shift==0.f ? x[idx] : E(float(x[idx])-shift);
  if((idx/width)%n==0)shifts[int64_t(b)*width+feature]=shift;
}
template <typename E>
__global__ void ScaleKernel(const E* x,E* y,float* scales,int n,int h,int d) {
  const int row=blockIdx.x;
  const int lane=threadIdx.x;
  const int64_t offset=int64_t(row)*d;
  float maximum=0.f;
  for (int c=lane;c<d;c+=32) maximum=fmaxf(maximum,fabsf(float(x[offset+c])));
  for (int delta=16;delta;delta/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,delta));
  int exponent=maximum>=1.f && isfinite(maximum) ? ilogbf(maximum) : 0;
  exponent=min(120,max(0,exponent));exponent &= ~1;
  float scale=ldexpf(1.f,exponent);
  bool invalid=false;
  for (int c=lane;c<d;c+=32) {
    const float original=float(x[offset+c]);
    E low=E(original/scale);
    invalid |= double(float(low))*double(scale)!=double(original);
  }
  invalid=__any_sync(0xffffffff,invalid);
  if (invalid) scale=1.f;
  for (int c=lane;c<d;c+=32) y[offset+c]=invalid ? x[offset+c] : E(float(x[offset+c])/scale);
  if (lane==0) {
    const int b=row/(n*h),i=(row/h)%n,head=row%h;
    scales[(int64_t(b)*h+head)*n+i]=scale;
  }
}
void Validate(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim()==4 && x.is_contiguous(),"precision preprocessing requires contiguous CUDA [B,N,H,D]");
  TORCH_CHECK(x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16,"precision preprocessing requires FP16/BF16");
  TORCH_CHECK(x.size(2)>0 && x.size(3)>0,"head count and dimension must be positive");
}
}
std::tuple<torch::Tensor,torch::Tensor> Center(const torch::Tensor& input) {
  Validate(input);c10::cuda::CUDAGuard guard(input.device());
  auto output=torch::empty_like(input);auto shifts=torch::empty({input.size(0),input.size(2),input.size(3)},input.options().dtype(at::kFloat));
  if (!input.numel()) {shifts.zero_();return {output,shifts};}
  auto stream=at::cuda::getCurrentCUDAStream(input.get_device());
  torch::Tensor flags;
  if(input.size(1)>128) flags=torch::zeros({input.size(0),input.size(2)*input.size(3)},input.options().dtype(at::kInt));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(),"precision_center",[&]{
    if(input.size(1)<=128) {
    CenterKernel<scalar_t><<<dim3((input.size(2)*input.size(3)+31)/32,input.size(0)),dim3(32,8),0,stream>>>(input.data_ptr<scalar_t>(),output.data_ptr<scalar_t>(),shifts.data_ptr<float>(),input.size(1),input.size(2),input.size(3));
    } else {
      int width=input.size(2)*input.size(3);
      CheckCenterTiles<scalar_t><<<dim3((width+31)/32,input.size(0),(input.size(1)+127)/128),dim3(32,8),0,stream>>>(input.data_ptr<scalar_t>(),flags.data_ptr<int>(),input.size(1),width);
      ApplyCenter<scalar_t><<<(input.numel()+255)/256,256,0,stream>>>(input.data_ptr<scalar_t>(),flags.data_ptr<int>(),output.data_ptr<scalar_t>(),shifts.data_ptr<float>(),input.numel(),input.size(1),width);
    }
  });C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,shifts};
}
std::tuple<torch::Tensor,torch::Tensor> ScaleDO(const torch::Tensor& input) {
  Validate(input);c10::cuda::CUDAGuard guard(input.device());
  auto output=torch::empty_like(input);auto scales=torch::empty({input.size(0),input.size(2),input.size(1)},input.options().dtype(at::kFloat));
  if (!input.numel()) return {output,scales};
  auto stream=at::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(),"precision_scale_do",[&]{
    ScaleKernel<scalar_t><<<input.numel()/input.size(3),32,0,stream>>>(input.data_ptr<scalar_t>(),output.data_ptr<scalar_t>(),scales.data_ptr<float>(),input.size(1),input.size(2),input.size(3));
  });C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,scales};
}
}  // namespace xattn::ops::attention::precision

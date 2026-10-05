#include "attention/hopper/bwd_template.cuh"
namespace xattn::ops::attention::precision::rectangular {
void Main_64_256_fp16_0(AttentionBwdParams p,cudaStream_t stream) {
 using V=attention::semantics::CausalFullVisibility;
 auto launch=[&](auto segment,auto level){
  constexpr bool HasSegment=decltype(segment)::value;constexpr int Level=decltype(level)::value;
  RunAttentionBwdSm90Kernel<64,256,64,cutlass::half_t,false,false,HasSegment,64,true,Level==4,V,flash::AttentionBwdSm90,0,0,0,false,V::kIsSlidingChunk,V::kIsSlidingChunk,false,false,false,false,Level>(p,stream);
 };
 auto dispatch=[&](auto level){if(p.q_segment_idx)launch(std::true_type{},level);else launch(std::false_type{},level);};
 switch(p.high_precision_level){case 2:dispatch(std::integral_constant<int,2>{});break;case 3:dispatch(std::integral_constant<int,3>{});break;default:TORCH_CHECK(false,"invalid precision level");}
}
}

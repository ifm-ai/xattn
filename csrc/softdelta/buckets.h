#pragma once

// (D, V) buckets of the fused SM90 softdelta reads, the causal_flash_attn
// buckets: D and V each rounded up to 32, 64, 96, 128, 160, 192 or 256; every
// pair has a forward and a backward instance (instantiations/, written by
// generate_instantiations.py). X(D, V) per bucket.
#define XATTN_SOFTDELTA_FOR_EACH_V(X, D) \
  X(D, 32) X(D, 64) X(D, 96) X(D, 128) X(D, 160) X(D, 192) X(D, 256)
#define XATTN_SOFTDELTA_FOR_EACH_BUCKET(X)                                    \
  XATTN_SOFTDELTA_FOR_EACH_V(X, 32) XATTN_SOFTDELTA_FOR_EACH_V(X, 64)   \
  XATTN_SOFTDELTA_FOR_EACH_V(X, 96) XATTN_SOFTDELTA_FOR_EACH_V(X, 128)  \
  XATTN_SOFTDELTA_FOR_EACH_V(X, 160) XATTN_SOFTDELTA_FOR_EACH_V(X, 192) \
  XATTN_SOFTDELTA_FOR_EACH_V(X, 256)

namespace xattn {
namespace ops {

// The dense (segment-free) backward's square instance for a (D, V) bucket pair,
// 0 for its own instance: causal_flash_attn's non-deterministic equal-length
// remaps (DispatchCausalFlashAttnBwd), except D 64 / V 256, which is faster on
// its own instance. Segment-masked backward runs every pair as is.
constexpr int SoftDeltaBwdDenseSquareDim(int d, int v) {
  return d == 64 && v == 96 ? 96
      : (d == 192 && (v == 128 || v == 160)) || (d == 160 && (v == 128 || v == 160 || v == 192)) ||
              (d == 128 && v == 192)
          ? 192
      : (d == 128 && v == 256) || (d == 256 && (v == 128 || v == 160 || v == 192)) ? 256
      : 0;
}

// Smallest bucket dimension holding d (0 past 256).
inline int SoftDeltaBucketDim(int d) {
  for (int bucket : {32, 64, 96, 128, 160, 192, 256}) {
    if (d <= bucket) {
      return bucket;
    }
  }
  return 0;
}

}  // namespace ops
}  // namespace xattn

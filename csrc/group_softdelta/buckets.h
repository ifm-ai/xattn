#pragma once

// (D, V) buckets of the fused SM90 group_softdelta reads, the causal_flash_attn
// buckets: D and V each rounded up to 32, 64, 96, 128, 160, 192 or 256; every
// pair has a forward and a backward instance (instantiations/, written by
// generate_instantiations.py). X(D, V) per bucket.
#if defined(XATTN_GROUP_SOFTDELTA_DEV_BUCKETS) && XATTN_GROUP_SOFTDELTA_DEV_BUCKETS == 1
// Development builds of the group_softdelta sources alone: one bucket.
#define XATTN_GROUP_SOFTDELTA_FOR_EACH_BUCKET(X) X(128, 128)
#elif defined(XATTN_GROUP_SOFTDELTA_DEV_BUCKETS) && XATTN_GROUP_SOFTDELTA_DEV_BUCKETS == 2
// Development builds: a cross-section of the tile configurations.
#define XATTN_GROUP_SOFTDELTA_FOR_EACH_BUCKET(X)                                \
  X(32, 32) X(64, 64) X(64, 96) X(96, 96) X(128, 64) X(128, 128) X(160, 160)    \
  X(192, 128) X(192, 192) X(256, 128) X(256, 256) X(64, 256)
#else
#define XATTN_GROUP_SOFTDELTA_FOR_EACH_V(X, D) \
  X(D, 32) X(D, 64) X(D, 96) X(D, 128) X(D, 160) X(D, 192) X(D, 256)
#define XATTN_GROUP_SOFTDELTA_FOR_EACH_BUCKET(X)                                    \
  XATTN_GROUP_SOFTDELTA_FOR_EACH_V(X, 32) XATTN_GROUP_SOFTDELTA_FOR_EACH_V(X, 64)   \
  XATTN_GROUP_SOFTDELTA_FOR_EACH_V(X, 96) XATTN_GROUP_SOFTDELTA_FOR_EACH_V(X, 128)  \
  XATTN_GROUP_SOFTDELTA_FOR_EACH_V(X, 160) XATTN_GROUP_SOFTDELTA_FOR_EACH_V(X, 192) \
  XATTN_GROUP_SOFTDELTA_FOR_EACH_V(X, 256)
#endif

namespace xattn {
namespace ops {

// The dense (segment-free) backward's square instance for a (D, V) bucket pair,
// 0 for its own instance: causal_flash_attn's non-deterministic equal-length
// remaps (DispatchCausalFlashAttnBwd), except D 64 / V 256, faster on its own
// instance here (16/4 B4 T4096: 2868 us for 20 heads vs causal's remapped
// 2908 us for 16). Segment-masked backward runs every pair as is.
constexpr int GroupSoftDeltaBwdDenseSquareDim(int d, int v) {
  return d == 64 && v == 96 ? 96
      : (d == 192 && (v == 128 || v == 160)) || (d == 160 && (v == 128 || v == 160 || v == 192)) ||
              (d == 128 && v == 192)
          ? 192
      : (d == 128 && v == 256) || (d == 256 && (v == 128 || v == 160 || v == 192)) ? 256
      : 0;
}

// Smallest bucket dimension holding d (0 past 256).
inline int GroupSoftDeltaBucketDim(int d) {
  for (int bucket : {32, 64, 96, 128, 160, 192, 256}) {
    if (d <= bucket) {
      return bucket;
    }
  }
  return 0;
}

}  // namespace ops
}  // namespace xattn

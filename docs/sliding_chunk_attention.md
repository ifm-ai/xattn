# Sliding Chunk Attention (FlashSCA)

[Back to supported attention](../README.md#supported-attention)

```python
from xattn import flash_sca

y = flash_sca(
    q,
    k,
    v,
    chunk_size=2048,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    reset_chunk_pos_per_seq=False,
    deterministic=False,
    attn_method="default",
    high_precision_level=0,
)
```

Tensor layouts:

- `q`: `[batch, seqlen, q_heads, qk_dim]`
- `k`: `[batch, seqlen, kv_heads, qk_dim]`
- `v`: `[batch, seqlen, kv_heads, v_dim]`
- `prev_k`: optional `[batch, chunk_size, kv_heads, qk_dim]`
- `prev_v`: optional `[batch, chunk_size, kv_heads, v_dim]`
- `q_segment_idx`: optional int64 `[batch, seqlen]`
- `k_segment_idx`: optional int64 `[batch, seqlen]` without previous K/V,
  or `[batch, chunk_size + seqlen]` with previous K/V
- `segment_idx`: optional int64 `[batch, seqlen]`, or
  `[batch, chunk_size + seqlen]` with previous K/V
- `bos_mask`: optional CUDA bool `[batch, seqlen]`, or
  `[batch, chunk_size + seqlen]` with previous K/V
- `high_precision_level`: `0` or `1`

Standalone forward/backward:

```python
from xattn import flash_sca_fwd, flash_sca_bwd

y, lse = flash_sca_fwd(
    q, k, v,
    chunk_size=2048,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    reset_chunk_pos_per_seq=False,
    attn_method="default",
    high_precision_level=1,
)
dq, dk, dv, dprev_k, dprev_v = flash_sca_bwd(
    dy, q, k, v, y, lse,
    chunk_size=2048,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    reset_chunk_pos_per_seq=False,
    deterministic=True,
    attn_method="default",
    high_precision_level=1,
)
```

Output layouts:

- `y`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `lse`: `[batch, q_heads, seqlen]`, FP32
- `dy`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `dq`, `dk`, `dv`: same shapes and dtypes as `q`, `k`, `v`
- `dprev_k`, `dprev_v`: same shapes and dtypes as previous K/V, or `None`

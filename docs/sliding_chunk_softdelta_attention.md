# Sliding Chunk SoftDelta Attention

[Back to supported attention](../README.md#supported-attention)

```python
from xattn import sliding_chunk_softdelta_attention

k_read = None

y = sliding_chunk_softdelta_attention(
    q, k, k_read, v, g,
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
    high_precision_level=0,
)
```

Tensor layouts:

- `q`: `[batch, seqlen, q_heads, qk_dim]`
- `k`: `[batch, seqlen, kv_heads, qk_dim]`
- `k_read`: Tensor or `None`: `[batch, seqlen, kv_heads, qk_dim]`
- `v`: `[batch, seqlen, kv_heads, v_dim]`
- `g`: `[batch, seqlen, q_heads]`, gate logits
- `prev_k`: optional `[batch, chunk_size, kv_heads, qk_dim]`
- `prev_v`: optional `[batch, chunk_size, kv_heads, v_dim]`
- `q_segment_idx`: optional int64 `[batch, seqlen]`
- `k_segment_idx`, `segment_idx`: optional int64 `[batch, prev_seqlen + seqlen]`
- `bos_mask`: optional CUDA bool `[batch, prev_seqlen + seqlen]`
- `high_precision_level`: `0` or `1`
- `prev_seqlen`: `chunk_size` with previous K/V, otherwise `0`

Standalone forward/backward:

```python
from xattn import sliding_chunk_softdelta_attention_fwd, sliding_chunk_softdelta_attention_bwd

k_read = None

y, neg_read, lse = sliding_chunk_softdelta_attention_fwd(
    q, k, k_read, v, g,
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
    high_precision_level=1,
)
dq, dk, dv, dg, dk_read, dprev_k, dprev_v = sliding_chunk_softdelta_attention_bwd(
    dy, q, k, k_read, v, g, y, neg_read, lse,
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
    high_precision_level=1,
)
```

Output layouts:

- `y`, `dy`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `neg_read`: `[batch, seqlen, kv_heads, v_dim]`, input dtype at level 0, FP32 at level 1
- `lse`: `[batch, kv_heads + q_heads, seqlen]`, FP32, the negative reads first
- `dq`, `dk`, `dv`, `dg`: same shapes and dtypes as `q`, `k`, `v`, `g`
- `dk_read`: same shape and dtype as `k_read`, or `None` when `k_read=None`
- `dprev_k`, `dprev_v`: same shapes and dtypes as previous K/V, or `None`

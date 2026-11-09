# Sliding Window Attention (FlashSWA)

[Back to supported attention](../README.md#supported-attention)

```python
from xattn import flash_swa

y = flash_swa(
    q,
    k,
    v,
    window_size=2048,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=False,
    high_precision_level=0,
)
```

Tensor layouts:

- `q`: `[batch, seqlen, q_heads, qk_dim]`
- `k`: `[batch, seqlen, kv_heads, qk_dim]`
- `v`: `[batch, seqlen, kv_heads, v_dim]`
- `prev_k`: optional `[batch, prev_seqlen, kv_heads, qk_dim]`
- `prev_v`: optional `[batch, prev_seqlen, kv_heads, v_dim]`
- `q_segment_idx`: optional int64 `[batch, seqlen]`
- `k_segment_idx`, `segment_idx`: optional int64 `[batch, prev_seqlen + seqlen]`
- `bos_mask`: optional CUDA bool `[batch, prev_seqlen + seqlen]`
- `high_precision_level`: `0` or `1`
- `prev_seqlen`: previous K/V sequence length, or `0`

Standalone forward/backward:

```python
from xattn import flash_swa_fwd, flash_swa_bwd

y, lse = flash_swa_fwd(
    q, k, v,
    window_size=2048,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    high_precision_level=1,
)
dq, dk, dv, dprev_k, dprev_v = flash_swa_bwd(
    dy, q, k, v, y, lse,
    window_size=2048,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=True,
    high_precision_level=1,
)
```

Output layouts:

- `y`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `lse`: `[batch, q_heads, seqlen]`, FP32
- `dy`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `dq`, `dk`, `dv`: same shapes and dtypes as `q`, `k`, `v`
- `dprev_k`, `dprev_v`: same shapes and dtypes as previous K/V, or `None`

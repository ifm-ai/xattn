# Causal Flash Attention

[Back to supported attention](../README.md#supported-attention)

```python
from xattn import causal_flash_attn

y = causal_flash_attn(
    q, k, v,
    scale=None,
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
- `q_segment_idx`, `k_segment_idx`, `segment_idx`: optional int64 `[batch, seqlen]`
- `bos_mask`: optional CUDA bool `[batch, seqlen]`
- `high_precision_level`: `0` or `1`

Standalone forward/backward:

```python
from xattn import causal_flash_attn_fwd, causal_flash_attn_bwd

y, lse = causal_flash_attn_fwd(
    q, k, v,
    scale=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    high_precision_level=1,
)
dq, dk, dv = causal_flash_attn_bwd(
    dy, q, k, v, y, lse,
    scale=None,
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

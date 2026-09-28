# Causal Flash Attention

[Back to supported attention](../README.md#supported-attention)

Inputs share a CUDA device and FP16/BF16 dtype. An SM90 build and GPU
are required; use `backend="auto"` or `"sm90"`.

The examples below use equal query and current key/value sequence lengths.

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
    high_precision_output=False,
)
```

Tensor layouts:

- `q`: `[batch, seqlen, q_heads, qk_dim]`
- `k`: `[batch, seqlen, kv_heads, qk_dim]`
- `v`: `[batch, seqlen, kv_heads, v_dim]`
- `q_segment_idx`, `k_segment_idx`, `segment_idx`: optional int64 `[batch, seqlen]`
- `bos_mask`: optional CUDA bool `[batch, seqlen]`

Causal within each segment; `q_heads % kv_heads == 0`.
`scale=None` uses `1 / sqrt(qk_dim)`. Segment metadata forms are mutually exclusive.

Standalone forward/backward:

```python
from xattn import causal_flash_attn_fwd, causal_flash_attn_bwd

y, y_fp32, lse = causal_flash_attn_fwd(
    q, k, v,
    scale=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    high_precision_output=True,
)
dq, dk, dv = causal_flash_attn_bwd(
    dy, q, k, v, y_fp32, lse,
    scale=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=True,
)
```

Output layouts:

- `y`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `y_fp32`: `[batch, seqlen, q_heads, v_dim]`, FP32 before casting; `None` when disabled
- `lse`: `[batch, q_heads, seqlen]`, FP32
- `dy`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `dq`, `dk`, `dv`: same shapes and dtypes as `q`, `k`, `v`

Forward always returns `(y, y_fp32, lse)`; `y_fp32` is `None` when
`high_precision_output=False`, including under `no_grad`/inference mode.
Backward's `y` can be the original dtype or FP32; its dtype selects the
computation. Pass `y_fp32` as `y` to use high precision. Reuse forward's options.
Standalone calls do not build an autograd graph or populate `.grad`.
The autograd API returns `y` and saves FP32 state internally when enabled for training.

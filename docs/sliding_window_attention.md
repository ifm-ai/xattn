# Sliding Window Attention (FlashSWA)

[Back to supported attention](../README.md#supported-attention)

Inputs share a CUDA device and FP16/BF16 dtype. An SM90 build and GPU
are required; use `backend="auto"` or `"sm90"`.

The examples below use equal query and current key/value sequence lengths.

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
    high_precision_output=False,
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

`prev_seqlen=0` without previous K/V. `window_size` is the inclusive left radius.
`q_heads % kv_heads == 0`; segment metadata forms are mutually exclusive.

Standalone forward/backward:

```python
from xattn import flash_swa_fwd, flash_swa_bwd

y, y_fp32, lse = flash_swa_fwd(
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
    high_precision_output=True,
)
dq, dk, dv, dprev_k, dprev_v = flash_swa_bwd(
    dy, q, k, v, y_fp32, lse,
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
)
```

Output layouts:

- `y`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `y_fp32`: `[batch, seqlen, q_heads, v_dim]`, FP32 before casting; `None` when disabled
- `lse`: `[batch, q_heads, seqlen]`, FP32
- `dy`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `dq`, `dk`, `dv`: same shapes and dtypes as `q`, `k`, `v`
- `dprev_k`, `dprev_v`: same shapes and dtypes as previous K/V, or `None`

Forward always returns `(y, y_fp32, lse)`; `y_fp32` is `None` when
`high_precision_output=False`, including under `no_grad`/inference mode.
Backward's `y` can be the original dtype or FP32; its dtype selects the
computation. Pass `y_fp32` as `y` to use high precision. Reuse forward's options.
Standalone calls do not build an autograd graph or populate `.grad`.
The autograd API returns `y` and saves FP32 state internally when enabled for training.

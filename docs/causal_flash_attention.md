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
    high_precision_level=0,
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

- `high_precision_level` (int, default `0`): `0` disables repairs; `1` saves FP32 backward state and uses subtract-before-scale (SBS) softmax to reduce output-rounding and cancellation error; `2` adds K/V centering and stable softmax-gradient evaluation to reduce common-offset leakage; `3` adds dS row-sum repair (`sum(dS)=0`) for consistent dQ/dK corrections; `4` adds dual-BF16 P/dS operands, range protection and nonnegative probability-mass repair (`sum(P)=1`) to improve dQ/dK/dV. Only integers 0–4 are accepted. Outputs and gradients keep the input dtype; constraints hold up to floating-point rounding.

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
    high_precision_level=1,
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
    high_precision_level=1,
)
```

Output layouts:

- `y`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `y_fp32`: `[batch, seqlen, q_heads, v_dim]`, FP32 before casting; `None` when disabled
- `lse`: `[batch, q_heads, seqlen]`, FP32
- `dy`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `dq`, `dk`, `dv`: same shapes and dtypes as `q`, `k`, `v`

Forward always returns `(y, y_fp32, lse)`; `y_fp32` is `None` when
`high_precision_level=0`, including under `no_grad`/inference mode.
Pass the same `high_precision_level` to forward and backward. At level 0,
pass `y` to backward; at levels 1–4, pass `y_fp32`. The state dtype does not
select the backward algorithm. Reuse forward's other options and LSE.
Standalone calls do not build an autograd graph or populate `.grad`.
The autograd API returns `y` and saves FP32 state internally at levels 1–4.

SM90 requires a positive batch size and Q/K width and flattened V width to be
multiples of 8 in `[8, 256]`. Unsupported widths are rejected before native
launch. Finite zero/negative scales use an exact Q zero/sign transform so
masking remains defined. Shape coverage does not guarantee bounded relative
error for arbitrary ill-conditioned or overflowing floating-point inputs.

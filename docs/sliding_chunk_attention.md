# Sliding Chunk Attention (FlashSCA)

[Back to supported attention](../README.md#supported-attention)

Inputs share a CUDA device and FP16/BF16 dtype. Use `backend="auto"`, `"cuda"`, or `"sm90"`; the build must include the target GPU.
SM90 supports right-aligned short queries, per-segment chunk resets, BOS masks,
and FP32 output state. A build targeting only SM90 accepts `auto` or `sm90`.

Attention is causal within the current and previous logical chunks.
`reset_chunk_pos_per_seq=True` resets chunk positions at each segment boundary.

The examples below use equal query and current key/value sequence lengths.

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
    high_precision_output=False,
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

GQA and MQA are supported when `q_heads` is divisible by `kv_heads`.

`segment_idx`, the `q_segment_idx`/`k_segment_idx` pair, and `bos_mask` are
mutually exclusive.

`scale` multiplies the QK logits before softmax; `None` uses
`1 / sqrt(qk_dim)`.

Standalone forward/backward:

```python
from xattn import flash_sca_fwd, flash_sca_bwd

y, y_fp32, lse = flash_sca_fwd(
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
    high_precision_output=True,
)
dq, dk, dv, dprev_k, dprev_v = flash_sca_bwd(
    dy, q, k, v, y_fp32, lse,
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

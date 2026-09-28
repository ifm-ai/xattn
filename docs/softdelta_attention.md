# SoftDelta Attention

[Back to supported attention](../README.md#supported-attention)

The autograd API accepts `backend="auto"`, `"sm90"`, or `"torch"`.
`auto` selects SM90 kernels on SM90 GPUs and the PyTorch reference elsewhere.
Standalone calls require an SM90 build and GPU, with `auto` or `sm90`.

The output is `reader1 - sigmoid(g) * reader2`. Reader 1 includes the aligned
query position; reader 2 uses only strictly earlier visible keys.

The examples below use equal query and current key/value sequence lengths.

```python
from xattn import softdelta_attention

y = softdelta_attention(
    q, k, v, g,
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

- `q`: `[batch, seqlen, 2 * q_heads, qk_dim]`, interleaved q1/q2 heads
- `k`: `[batch, seqlen, kv_heads, qk_dim]`
- `v`: `[batch, seqlen, kv_heads, groups, group_dim]`
- `g`: `[batch, seqlen, q_heads, groups, 1]`, gate logits
- `prev_k`: optional `[batch, prev_seqlen, kv_heads, qk_dim]`
- `prev_v`: optional `[batch, prev_seqlen, kv_heads, groups, group_dim]`
- `q_segment_idx`: optional int64 `[batch, seqlen]`
- `k_segment_idx`, `segment_idx`: optional int64 `[batch, prev_seqlen + seqlen]`
- `bos_mask`: optional CUDA bool `[batch, prev_seqlen + seqlen]`

`q/k/v/g` share a device and FP16/BF16 dtype. `q_heads % kv_heads == 0`.
`prev_seqlen=0` without previous K/V; segment metadata forms are mutually exclusive.
`scale=None` uses `1 / sqrt(qk_dim)`.

Standalone forward/backward:

```python
from xattn import softdelta_attention_fwd, softdelta_attention_bwd

y, readers, lse = softdelta_attention_fwd(
    q, k, v, g,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=False,
    high_precision_output=True,
)
dq, dk, dv, dg, dprev_k, dprev_v = softdelta_attention_bwd(
    dy, q, k, v, g, readers, lse,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=False,
)
```

Output layouts:

- `y`, `dy`: `[batch, seqlen, q_heads, groups, group_dim]`, input dtype
- `readers`: `[batch, seqlen, 2 * q_heads, groups, group_dim]`, FP32 when high precision is enabled, otherwise input dtype
- `lse`: `[batch, 2 * q_heads, seqlen]`, FP32, interleaved q1/q2 heads
- `dq`, `dk`, `dv`, `dg`: same shapes and dtypes as `q`, `k`, `v`, `g`
- `dprev_k`, `dprev_v`: same shapes and dtypes as previous K/V, or `None`

Standalone calls require SM90 and do not build an autograd graph or populate `.grad`.
Forward always returns `(y, readers, lse)`, including under
`no_grad`/inference mode. Backward automatically selects precision from
`readers.dtype`. Reuse forward's options and LSE.
Final `y` alone is insufficient for backward: both reader outputs are needed.
The autograd API returns only `y` and retains high-precision reader state internally.

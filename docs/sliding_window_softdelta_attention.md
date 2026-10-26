# Sliding Window SoftDelta Attention

[Back to supported attention](../README.md#supported-attention)

Applies SoftDelta within a causal sliding window. `window_size` is the
inclusive left radius.

The autograd API accepts `backend="auto"`, `"sm90"`, or `"torch"`.
`auto` selects SM90 kernels on SM90 GPUs and the PyTorch reference elsewhere.
Standalone calls require an SM90 build and GPU, with `auto` or `sm90`.

The output is `reader1 - sigmoid(g) * reader2`. Reader 1 includes the aligned
query position; reader 2 uses only strictly earlier visible keys.

The examples below use equal query and current key/value sequence lengths.

```python
from xattn import sliding_window_softdelta_attention

y = sliding_window_softdelta_attention(
    q, k, v, g,
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

- `high_precision_level` (int, default `0`): `0` disables repairs; `1` saves FP32 backward state and uses subtract-before-scale (SBS) softmax to reduce output-rounding and cancellation error; `2` adds K/V centering and stable softmax-gradient evaluation to reduce common-offset leakage; `3` adds dS row-sum repair (`sum(dS)=0`) for consistent dQ/dK corrections; `4` adds dual-BF16 P/dS operands, range protection and nonnegative probability-mass repair (`sum(P)=1`) to improve dQ/dK/dV. Only integers 0–4 are accepted. Outputs and gradients keep the input dtype; constraints hold up to floating-point rounding.

Standalone forward/backward:

```python
from xattn import sliding_window_softdelta_attention_fwd, sliding_window_softdelta_attention_bwd

y, readers, lse = sliding_window_softdelta_attention_fwd(
    q, k, v, g,
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
    high_precision_level=1,
)
dq, dk, dv, dg, dprev_k, dprev_v = sliding_window_softdelta_attention_bwd(
    dy, q, k, v, g, readers, lse,
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
    high_precision_level=1,
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
`no_grad`/inference mode. Pass the same `high_precision_level` to backward;
`readers.dtype` does not select the algorithm. Reuse forward's other options and LSE.
Final `y` alone is insufficient for backward: both reader outputs are needed.
The autograd API returns only `y` and retains high-precision reader state internally.

SM90 requires a positive batch size and Q/K width and flattened V width to be
multiples of 8 in `[8, 256]`. Unsupported widths are rejected before native
launch. Finite zero/negative scales use an exact Q zero/sign transform so
masking remains defined. Shape coverage does not guarantee bounded relative
error for arbitrary ill-conditioned or overflowing floating-point inputs.

## Group SoftDelta (in development)

```python
from xattn import sliding_window_group_softdelta_attention

y = sliding_window_group_softdelta_attention(
    q, k, v, g,
    window_size=2048,
    neg_q=None,
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
- `g`: `[batch, seqlen, q_heads]`, gate logits
- `neg_q`: optional `[batch, seqlen, kv_heads, qk_dim]`, same device and dtype as `q`
- `prev_k`: optional `[batch, prev_seqlen, kv_heads, qk_dim]`
- `prev_v`: optional `[batch, prev_seqlen, kv_heads, v_dim]`
- `q_segment_idx`: optional int64 `[batch, seqlen]`
- `k_segment_idx`, `segment_idx`: optional int64 `[batch, prev_seqlen + seqlen]`
- `bos_mask`: optional CUDA bool `[batch, prev_seqlen + seqlen]`
- `prev_seqlen`: previous K/V sequence length, or `0`

Standalone forward/backward:

```python
from xattn import (
    sliding_window_group_softdelta_attention_bwd,
    sliding_window_group_softdelta_attention_fwd,
)

y, readers, lse = sliding_window_group_softdelta_attention_fwd(
    q, k, v, g,
    window_size=2048,
    neg_q=None,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=False,
    high_precision_level=1,
)
dq, dk, dv, dg, dneg_q, dprev_k, dprev_v = sliding_window_group_softdelta_attention_bwd(
    dy, q, k, v, g, readers, lse,
    window_size=2048,
    neg_q=None,
    scale=None,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
    segment_idx=None,
    bos_mask=None,
    backend="auto",
    deterministic=False,
    high_precision_level=1,
)
```

Output layouts:

- `y`, `dy`: `[batch, seqlen, q_heads, v_dim]`, input dtype
- `readers`: `[batch, seqlen, kv_heads + q_heads, v_dim]`, FP32 when high precision is enabled, otherwise input dtype
- `lse`: `[batch, kv_heads + q_heads, seqlen]`, FP32
- `dq`, `dk`, `dv`, `dg`: same shapes and dtypes as `q`, `k`, `v`, `g`
- `dneg_q`: same shape and dtype as `neg_q`, or `None` when `neg_q=None`
- `dprev_k`, `dprev_v`: same shapes and dtypes as previous K/V, or `None`

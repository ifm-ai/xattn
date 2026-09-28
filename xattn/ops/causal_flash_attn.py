# Author: Shicheng Wen

"""Causal flash attention using xattn's existing CUDA attention kernels."""

from typing import Optional, Tuple

from torch import Tensor

from ._causal_attention import (
    _causal_attention,
    _causal_attention_bwd,
    _causal_attention_fwd,
)

__all__ = ["causal_flash_attn", "causal_flash_attn_bwd", "causal_flash_attn_fwd"]


def causal_flash_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Optional[float] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    deterministic: bool = False,
    high_precision_output: bool = False,
) -> Tensor:
    """Run causal flash attention with autograd and optional segment isolation.

    Q/K/V have layouts ``[B, Lq, Hq, D]``, ``[B, Lkv, Hkv, D]``, and
    ``[B, Lkv, Hkv, V]``, with ``0 < Lq <= Lkv``. MHA/GQA/MQA are supported when Hq is divisible by
    Hkv. Q is aligned to the right of KV: query ``i`` can attend every key
    ``j <= i + Lkv - Lq`` with the same segment ID, including its aligned
    position. Segment IDs isolate keys without resetting token positions. There is no local window or chunk
    boundary; bidirectional attention is not supported.

    ``segment_idx`` covers all Lkv tokens and its last Lq entries label Q. Alternatively, supply both Q/K
    segment indices, or a boolean
    ``bos_mask`` over all keys. These three metadata forms are exclusive.
    Without metadata, all tokens in each batch row share one segment.

    ``scale=None`` uses ``1/sqrt(D)``. Inputs must be CUDA fp16/bf16 on an
    SM90 GPU with ``backend='auto'`` or ``'sm90'``. Output is
    ``[B, Lq, Hq, V]`` with the input dtype. ``high_precision_output=True``
    retains FP32 output state for backward without changing the public dtype.
    """
    return _causal_attention(
        q,
        k,
        v,
        None,
        scale,
        None,
        None,
        q_segment_idx,
        k_segment_idx,
        segment_idx=segment_idx,
        bos_mask=bos_mask,
        backend=backend,
        deterministic=deterministic,
        high_precision_output=high_precision_output,
        api_name="causal_flash_attn",
    )


def causal_flash_attn_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Optional[float] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    high_precision_output: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Return ``(y, y_fp32, lse)``; ``y_fp32`` is None without high precision.

    ``y`` keeps the Q/K/V dtype. ``y_fp32`` is the same attention output
    before casting, with the same shape and FP32 dtype. ``lse`` is FP32
    with shape ``[B, Hq, Lq]``. Both outputs are written by one forward,
    including in no_grad/inference mode; no autograd bridge is created.
    Pass ``y_fp32`` as backward's ``y`` for high-precision state, while
    keeping ``y_grad`` in the Q/K/V dtype.
    """
    return _causal_attention_fwd(
        q,
        k,
        v,
        None,
        scale,
        None,
        None,
        q_segment_idx,
        k_segment_idx,
        segment_idx=segment_idx,
        bos_mask=bos_mask,
        backend=backend,
        high_precision_output=high_precision_output,
        api_name="causal_flash_attn_fwd",
    )


def causal_flash_attn_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    scale: Optional[float] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    deterministic: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return flash-attention gradients ``(dq, dk, dv)``.

    Pass ``y`` and ``lse`` from :func:`causal_flash_attn_fwd` with the same
    Q/K/V, scale, and segment metadata. ``y.dtype`` automatically
    selects FP32 or input-precision state; no precision flag is needed.
    ``y_grad`` must have the Q/K/V dtype in either case.
    """
    return _causal_attention_bwd(
        y_grad,
        q,
        k,
        v,
        y,
        lse,
        None,
        scale,
        None,
        None,
        q_segment_idx,
        k_segment_idx,
        segment_idx=segment_idx,
        bos_mask=bos_mask,
        backend=backend,
        deterministic=deterministic,
        api_name="causal_flash_attn_bwd",
    )[:3]

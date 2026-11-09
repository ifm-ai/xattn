from ._attention_core import validate_sm90_dimensions, validate_attention_scale
from ._backward_precision import _dispatch_nonpositive_scale
from ._backward_precision import validate_high_precision_level
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
    high_precision_level: int = 0,
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
    ``[B, Lq, Hq, V]`` with the input dtype. ``high_precision_level=1`` runs
    the forward softmax in high precision and the precision-repair backward.
    """
    validate_high_precision_level(high_precision_level, api_name="causal_flash_attn")
    validate_attention_scale(scale)
    validate_sm90_dimensions(q, v, backend, api_name="causal_flash_attn")
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(causal_flash_attn, locals())
    if high_precision_level:
        from ._precision_backend import apply as precision_apply
        return precision_apply(q, k, v, visibility=0,
            scale=scale, q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
            segment_idx=segment_idx, bos_mask=bos_mask, backend=backend,
            deterministic=deterministic, high_precision_level=high_precision_level)
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
    high_precision_level: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Return ``(y, lse)``.

    ``y`` keeps the Q/K/V dtype; ``lse`` is FP32 with shape ``[B, Hq, Lq]``.
    ``high_precision_level=1`` runs the softmax in high precision, also in
    no_grad/inference mode; no autograd bridge is created.
    """
    validate_high_precision_level(high_precision_level, api_name="causal_flash_attn_fwd")
    validate_attention_scale(scale)
    validate_sm90_dimensions(q, v, backend, api_name="causal_flash_attn_fwd")
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(causal_flash_attn_fwd, locals())
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
        high_precision_level=high_precision_level,
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
    high_precision_level: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return flash-attention gradients ``(dq, dk, dv)``.

    Pass ``y`` and ``lse`` from :func:`causal_flash_attn_fwd` with the same
    Q/K/V, scale, segment metadata and ``high_precision_level``. ``y`` and
    ``y_grad`` have the Q/K/V dtype; level 1 recomputes the softmax
    statistics and does not read ``y`` or ``lse``.
    """
    validate_high_precision_level(high_precision_level, api_name="causal_flash_attn_bwd")
    validate_attention_scale(scale)
    validate_sm90_dimensions(q, v, backend, api_name="causal_flash_attn_bwd")
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(causal_flash_attn_bwd, locals())
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
        high_precision_level=high_precision_level,
        api_name="causal_flash_attn_bwd",
    )[:3]

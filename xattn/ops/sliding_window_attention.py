from ._attention_core import validate_sm90_dimensions, validate_attention_scale
from ._backward_precision import _dispatch_nonpositive_scale
from ._backward_precision import validate_high_precision_level
# Author: Shicheng Wen

"""Public sliding-window attention API."""

from typing import Optional, Tuple

from torch import Tensor

from ._causal_attention import (
    _causal_attention,
    _causal_attention_bwd,
    _causal_attention_fwd,
    _validate_window_size,
)
from ._causal_attention import (
    _flash_swa_varlen as _flash_swa_varlen,
)
from ._causal_attention import (
    _flash_swa_varlen_bwd as _flash_swa_varlen_bwd,
)
from ._causal_attention import (
    _flash_swa_varlen_fwd as _flash_swa_varlen_fwd,
)

__all__ = ["flash_swa", "flash_swa_bwd", "flash_swa_fwd"]


def flash_swa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    deterministic: bool = False,
    high_precision_level: int = 0,
) -> Tensor:
    """Run causal sliding-window attention with optional segment metadata.

    Q may be shorter than current K/V and is right-aligned to total K/V,
    including previous K/V. Segment equality is applied after alignment.
    Common ``segment_idx`` and ``bos_mask`` describe total K/V; Q uses its
    suffix.

    ``window_size`` is an inclusive left radius. Query position ``i`` attends
    keys in ``[i + Lk - Lq - window_size, i + Lk - Lq]``.

    ``high_precision_level=1`` runs the forward softmax in high precision and
    the precision-repair backward. The output keeps the input dtype.
    """
    validate_high_precision_level(high_precision_level, api_name="flash_swa")
    validate_attention_scale(scale)
    validate_sm90_dimensions(q, v, backend, api_name="flash_swa")
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(flash_swa, locals())
    if high_precision_level:
        from ._precision_backend import apply as precision_apply
        return precision_apply(q, k, v, visibility=1, window_size=window_size, prev_k=prev_k, prev_v=prev_v,
            scale=scale, q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
            segment_idx=segment_idx, bos_mask=bos_mask, backend=backend,
            deterministic=deterministic, high_precision_level=high_precision_level)
    _validate_window_size(window_size)
    return _causal_attention(
        q,
        k,
        v,
        window_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        segment_idx=segment_idx,
        bos_mask=bos_mask,
        backend=backend,
        deterministic=deterministic,
    )


def flash_swa_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
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
    validate_high_precision_level(high_precision_level, api_name="flash_swa_fwd")
    validate_attention_scale(scale)
    validate_sm90_dimensions(q, v, backend, api_name="flash_swa_fwd")
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(flash_swa_fwd, locals())
    _validate_window_size(window_size)
    return _causal_attention_fwd(
        q,
        k,
        v,
        window_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        segment_idx=segment_idx,
        bos_mask=bos_mask,
        backend=backend,
        high_precision_level=high_precision_level,
    )


def flash_swa_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    window_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    deterministic: bool = False,
    high_precision_level: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Run FlashSWA BWD using the requested high_precision_level.

    Pass ``y`` and ``lse`` from :func:`flash_swa_fwd` with the same level.
    ``y`` and ``y_grad`` have the Q/K/V dtype; level 1 recomputes the
    softmax statistics and does not read ``y`` or ``lse``.
    """
    validate_high_precision_level(high_precision_level, api_name="flash_swa_bwd")
    validate_attention_scale(scale)
    validate_sm90_dimensions(q, v, backend, api_name="flash_swa_bwd")
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(flash_swa_bwd, locals())
    _validate_window_size(window_size)
    return _causal_attention_bwd(
        y_grad,
        q,
        k,
        v,
        y,
        lse,
        window_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        segment_idx=segment_idx,
        bos_mask=bos_mask,
        backend=backend,
        deterministic=deterministic,
        high_precision_level=high_precision_level,
    )

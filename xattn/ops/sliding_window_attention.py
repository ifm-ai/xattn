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
    high_precision_output: bool = False,
) -> Tensor:
    """Run causal sliding-window attention with optional segment metadata.

    Q may be shorter than current K/V and is right-aligned to total K/V,
    including previous K/V. Segment equality is applied after alignment.
    Common ``segment_idx`` and ``bos_mask`` describe total K/V; Q uses its
    suffix.

    ``window_size`` is an inclusive left radius. Query position ``i`` attends
    keys in ``[i + Lk - Lq - window_size, i + Lk - Lq]``.

    ``high_precision_output=True`` saves FP32 attention state for backward.
    The public output keeps the input dtype.
    """
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
        high_precision_output=high_precision_output,
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
    high_precision_output: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Return ``(y, y_fp32, lse)``; ``y_fp32`` is None without high precision.

    ``y`` keeps the Q/K/V dtype. ``y_fp32`` is the same attention output
    before casting, with the same shape and FP32 dtype. ``lse`` is FP32
    with shape ``[B, Hq, L]``. Both outputs are written by one forward,
    including in no_grad/inference mode; no autograd bridge is created.
    Pass ``y_fp32`` as backward's ``y`` for high-precision state, while
    keeping ``y_grad`` in the Q/K/V dtype.
    """
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
        high_precision_output=high_precision_output,
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
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Run FlashSWA BWD, automatically selecting state precision by y.dtype.

    ``y`` may be FP32 or input dtype; ``y_grad`` must match Q/K/V dtype.
    """
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
    )

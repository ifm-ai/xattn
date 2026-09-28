from typing import Optional, Tuple

import torch
from torch import Tensor


def resolve_segment_indices(
    segment_idx: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    *,
    batch_size: int,
    q_length: int,
    k_length: int,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Resolve the public common-index shorthand into independent Q/K views."""
    if segment_idx is not None:
        if q_segment_idx is not None or k_segment_idx is not None:
            raise ValueError(
                "segment_idx cannot be combined with q_segment_idx or "
                "k_segment_idx"
            )
        if tuple(segment_idx.shape) != (batch_size, k_length):
            raise ValueError(
                f"segment_idx must have shape [{batch_size}, {k_length}]"
            )
        q_idx = (
            segment_idx
            if q_length == k_length
            else segment_idx[:, -q_length:]
        )
        return q_idx, segment_idx
    if (q_segment_idx is None) != (k_segment_idx is None):
        raise ValueError(
            "q_segment_idx and k_segment_idx must be provided together"
        )
    return q_segment_idx, k_segment_idx


def validate_segment_indices(
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    *,
    batch_size: int,
    q_length: int,
    k_length: int,
    device: torch.device,
    api_name: str,
) -> None:
    """Validate segment metadata without reading CUDA values."""
    if (q_segment_idx is None) != (k_segment_idx is None):
        raise ValueError(
            "q_segment_idx and k_segment_idx must be provided together"
        )
    if q_segment_idx is None:
        return
    assert k_segment_idx is not None
    if q_segment_idx.dtype != torch.int64:
        raise ValueError(f"{api_name} q_segment_idx must be torch.int64")
    if k_segment_idx.dtype != torch.int64:
        raise ValueError(f"{api_name} k_segment_idx must be torch.int64")
    if q_segment_idx.device != device or k_segment_idx.device != device:
        raise ValueError(
            f"{api_name} segment metadata must share the Q/K/V device"
        )
    if tuple(q_segment_idx.shape) != (batch_size, q_length):
        raise ValueError(
            f"{api_name} q_segment_idx must have shape "
            f"[{batch_size}, {q_length}]"
        )
    if tuple(k_segment_idx.shape) != (batch_size, k_length):
        raise ValueError(
            f"{api_name} k_segment_idx must have shape "
            f"[{batch_size}, {k_length}]"
        )

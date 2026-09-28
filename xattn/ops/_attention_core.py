# Author: Shicheng Wen

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor


class AttentionVisibility(Enum):
    CAUSAL_FULL = "causal_full"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_CHUNK = "sliding_chunk"


class AttentionBoundary(Enum):
    INCLUSIVE = "inclusive"
    STRICT_PAST = "strict_past"


@dataclass(frozen=True)
class AttentionSemantics:
    visibility: AttentionVisibility
    boundary: AttentionBoundary
    span: Optional[int] = None
    reset_position_per_segment: bool = False

    def __post_init__(self) -> None:
        if self.visibility is AttentionVisibility.CAUSAL_FULL:
            if self.span is not None:
                raise ValueError("causal-full visibility does not take a span")
            return
        if isinstance(self.span, bool) or not isinstance(self.span, int):
            raise TypeError("local visibility span must be an integer")
        if self.visibility is AttentionVisibility.SLIDING_WINDOW:
            if self.span < 0:
                raise ValueError("sliding-window span must be nonnegative")
            return
        if self.span <= 0:
            raise ValueError("sliding-chunk span must be positive")


def resolve_scale(q: Tensor, scale: Optional[float]) -> float:
    value = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    if not math.isfinite(value):
        raise ValueError("attention scale must be finite")
    return value


def _run_coordinates(segment_idx: Tensor) -> Tuple[Tensor, Tensor]:
    """Return zero-based position and length for each contiguous run."""
    batch, length = segment_idx.shape
    positions = torch.arange(length, device=segment_idx.device).expand(
        batch, length
    )

    starts_run = torch.ones_like(segment_idx, dtype=torch.bool)
    if length > 1:
        starts_run[:, 1:] = segment_idx[:, 1:] != segment_idx[:, :-1]
    run_starts = torch.where(starts_run, positions, 0).cummax(dim=1).values

    ends_run = torch.ones_like(segment_idx, dtype=torch.bool)
    if length > 1:
        ends_run[:, :-1] = segment_idx[:, 1:] != segment_idx[:, :-1]
    end_markers = torch.where(ends_run, positions + 1, length)
    run_ends = torch.flip(
        torch.flip(end_markers, dims=(1,)).cummin(dim=1).values,
        dims=(1,),
    )
    return positions - run_starts, run_ends - run_starts


def _logical_positions(
    batch: int,
    q_length: int,
    k_length: int,
    device: torch.device,
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    reset_position_per_segment: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    if q_segment_idx is None:
        same_problem = torch.ones(
            batch,
            q_length,
            k_length,
            dtype=torch.bool,
            device=device,
        )
    else:
        assert k_segment_idx is not None
        same_problem = q_segment_idx.unsqueeze(2) == k_segment_idx.unsqueeze(1)

    if not reset_position_per_segment:
        q_position = torch.arange(q_length, device=device) + (
            k_length - q_length
        )
        k_position = torch.arange(k_length, device=device)
        return (
            q_position.expand(batch, q_length),
            k_position.expand(batch, k_length),
            same_problem,
        )

    if q_segment_idx is None:
        q_position = torch.arange(q_length, device=device) + (
            k_length - q_length
        )
        k_position = torch.arange(k_length, device=device)
        return (
            q_position.expand(batch, q_length),
            k_position.expand(batch, k_length),
            same_problem,
        )

    assert k_segment_idx is not None
    q_local, q_run_length = _run_coordinates(q_segment_idx)
    k_local, _ = _run_coordinates(k_segment_idx)
    matching_k_count = same_problem.sum(dim=-1)
    q_position = q_local + matching_k_count - q_run_length
    return q_position, k_local, same_problem


def build_attention_mask(
    *,
    batch: int,
    q_length: int,
    k_length: int,
    device: torch.device,
    semantics: AttentionSemantics,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
) -> Tensor:
    """Materialize the exact boolean support for a correctness plan."""
    q_position, k_position, same_problem = _logical_positions(
        batch,
        q_length,
        k_length,
        device,
        q_segment_idx,
        k_segment_idx,
        semantics.reset_position_per_segment,
    )
    q_position = q_position.unsqueeze(2)
    k_position = k_position.unsqueeze(1)

    if semantics.boundary is AttentionBoundary.INCLUSIVE:
        allowed = k_position <= q_position
    else:
        allowed = k_position < q_position

    if semantics.visibility is AttentionVisibility.SLIDING_WINDOW:
        assert semantics.span is not None
        allowed = allowed & (k_position >= q_position - semantics.span)
    elif semantics.visibility is AttentionVisibility.SLIDING_CHUNK:
        assert semantics.span is not None
        chunk_begin = torch.div(
            q_position,
            semantics.span,
            rounding_mode="floor",
        ) * semantics.span
        allowed = allowed & (k_position >= chunk_begin - semantics.span)

    return allowed & same_problem


def attention_read(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allowed: Tensor,
    scale: float,
) -> Tensor:
    """Run one normalized attention read with explicit empty-row behavior."""
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)

    q_heads_first = q.transpose(1, 2)
    k_heads_first = k.transpose(1, 2)
    v_heads_first = v.transpose(1, 2)
    accumulation_dtype = (
        torch.float64 if q.dtype == torch.float64 else torch.float32
    )
    scores = torch.matmul(
        q_heads_first.to(accumulation_dtype),
        k_heads_first.to(accumulation_dtype).transpose(-2, -1),
    ) * scale

    allowed_heads = allowed.unsqueeze(1)
    masked_scores = scores.masked_fill(~allowed_heads, float("-inf"))
    nonempty = allowed.any(dim=-1, keepdim=True).unsqueeze(1)
    safe_scores = torch.where(nonempty, masked_scores, torch.zeros_like(scores))
    probabilities = torch.softmax(safe_scores, dim=-1)
    probabilities = probabilities.masked_fill(~allowed_heads, 0.0)
    output = torch.matmul(
        probabilities,
        v_heads_first.to(accumulation_dtype),
    )
    return output.transpose(1, 2)

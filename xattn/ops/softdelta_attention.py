# Author: Shicheng Wen

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from ._sca_segment_alignment import needs_matched_reset

from ._backward_precision import use_high_precision_output
from ._attention_core import (
    AttentionBoundary,
    AttentionSemantics,
    AttentionVisibility,
    attention_read,
    build_attention_mask,
    resolve_scale,
)
from ._bos_metadata import (
    requires_row_aligned_k_runs,
    validate_bos_mask,
)
from ._bos_routing_heuristics import select_flash_softdelta_bos_routes
from ._flash_softdelta import (
    flash_chunk_bos_read,
    flash_chunk_inclusive_read,
    flash_chunk_strict_past_read,
    flash_softdelta_bos_metadata,
    flash_softdelta_bos_segment_idx,
    flash_window_bos_read,
    flash_softdelta_composed,
    flash_softdelta_fwd,
    flash_softdelta_paired_training,
    flash_softdelta_gate,
    flash_window_inclusive_read,
    flash_window_strict_past_read,
)
from ._segment_metadata import (
    resolve_segment_indices,
    validate_segment_indices,
)

__all__ = [
    "softdelta_attention",
    "softdelta_attention_bwd",
    "softdelta_attention_fwd",
    "sliding_window_softdelta_attention",
    "sliding_window_softdelta_attention_bwd",
    "sliding_window_softdelta_attention_fwd",
    "sliding_chunk_softdelta_attention",
    "sliding_chunk_softdelta_attention_bwd",
    "sliding_chunk_softdelta_attention_fwd",
]

_FUSED_HEAD_DIMS = frozenset((32, 64, 96, 128, 160, 192, 256))

# H200 BF16 MHA shape allowlists for fused reader-pair dispatch.
_CTA_READER_PAIR_KV_REUSE_SPANS = {
    AttentionVisibility.SLIDING_WINDOW: 2047,
    AttentionVisibility.SLIDING_CHUNK: 2048,
}
_CTA_READER_PAIR_KV_REUSE_HEAD_LAYOUTS = frozenset(((8, 8),))
_CTA_READER_PAIR_KV_REUSE_SHAPES = {
    False: {
        AttentionVisibility.SLIDING_WINDOW: frozenset(
            (
                (32, 32),
                (32, 64),
                (32, 192),
                (64, 32),
                (64, 64),
                (64, 256),
                (96, 32),
                (96, 64),
                (96, 96),
                (128, 32),
                (128, 64),
                (128, 96),
                (128, 128),
                (128, 160),
                (128, 192),
                (160, 32),
                (160, 64),
                (160, 96),
                (160, 128),
                (160, 160),
                (160, 256),
                (192, 32),
                (192, 64),
                (192, 128),
                (192, 192),
                (192, 256),
                (256, 256),
            )
        ),
        AttentionVisibility.SLIDING_CHUNK: frozenset(
            (
                (32, 32),
                (32, 64),
                (32, 192),
                (64, 32),
                (64, 64),
                (64, 256),
                (96, 32),
                (96, 64),
                (96, 96),
                (96, 192),
                (96, 256),
                (128, 32),
                (128, 64),
                (128, 96),
                (128, 160),
                (128, 192),
                (128, 256),
                (160, 32),
                (160, 64),
                (160, 96),
                (160, 256),
                (192, 32),
                (192, 64),
                (192, 160),
                (192, 256),
                (256, 32),
            )
        ),
    },
    True: {
        AttentionVisibility.SLIDING_WINDOW: frozenset(
            (
                (32, 32),
                (32, 64),
                (32, 128),
                (32, 160),
                (32, 192),
                (64, 32),
                (64, 64),
                (64, 96),
                (64, 160),
                (64, 192),
                (64, 256),
                (96, 32),
                (96, 64),
                (96, 96),
                (96, 128),
                (96, 160),
                (96, 192),
                (96, 256),
                (128, 32),
                (128, 64),
                (128, 96),
                (128, 128),
                (128, 160),
                (128, 192),
                (128, 256),
                (160, 32),
                (160, 64),
                (160, 96),
                (160, 128),
                (160, 160),
                (160, 192),
                (160, 256),
                (192, 32),
                (192, 64),
                (192, 96),
                (192, 128),
                (192, 160),
                (192, 192),
                (192, 256),
                (256, 32),
                (256, 64),
                (256, 96),
                (256, 128),
                (256, 192),
                (256, 256),
            )
        ),
        AttentionVisibility.SLIDING_CHUNK: frozenset(
            (
                (32, 32),
                (32, 64),
                (32, 192),
                (64, 32),
                (64, 64),
                (64, 160),
                (64, 192),
                (64, 256),
                (96, 32),
                (96, 64),
                (96, 96),
                (96, 128),
                (96, 160),
                (96, 192),
                (96, 256),
                (128, 32),
                (128, 64),
                (128, 96),
                (128, 128),
                (128, 160),
                (128, 192),
                (128, 256),
                (160, 32),
                (160, 64),
                (160, 96),
                (160, 192),
                (160, 256),
                (192, 32),
                (192, 64),
                (192, 96),
                (192, 160),
                (192, 192),
                (192, 256),
                (256, 32),
                (256, 64),
                (256, 96),
                (256, 128),
                (256, 192),
                (256, 256),
            )
        ),
    },
}

def _use_head_pair_parallel(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    visibility: AttentionVisibility,
    span: int,
    deterministic: bool,
) -> bool:
    """Return whether full-causal parallel-reader backward is enabled."""
    q_heads = q.shape[2] // 2
    value_dim = v.shape[3] * v.shape[4]
    return (
        visibility is AttentionVisibility.CAUSAL_FULL
        and q.dtype == torch.bfloat16
        and not deterministic
        and q.shape[0] == 1
        and (q_heads, k.shape[2]) == (8, 8)
        and q.shape[3] == 256
        and value_dim == 256
        and v.shape[3] == 4
        and torch.cuda.get_device_name(q.device) == "NVIDIA H200"
    )


def _use_cta_reader_pair_kv_reuse(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    visibility: AttentionVisibility,
    span: int,
    deterministic: bool,
    high_precision_output: bool,
) -> bool:
    """Return whether the runtime identity is in the K/V-reuse allowlist."""
    q_heads = q.shape[2] // 2
    value_dim = v.shape[3] * v.shape[4]
    return (
        q.dtype == torch.bfloat16
        and not deterministic
        and q.shape[0] == 1
        and v.shape[3] == 4
        and (q_heads, k.shape[2]) in _CTA_READER_PAIR_KV_REUSE_HEAD_LAYOUTS
        and _CTA_READER_PAIR_KV_REUSE_SPANS.get(visibility) == span
        and (q.shape[3], value_dim)
        in _CTA_READER_PAIR_KV_REUSE_SHAPES[high_precision_output].get(
            visibility, ()
        )
        and torch.cuda.get_device_name(q.device) == "NVIDIA H200"
    )


def _use_window_reader_pair_training(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    visibility: AttentionVisibility,
    span: int,
    deterministic: bool,
    high_precision_output: bool,
) -> bool:
    """Select paired window forward for supported training shapes."""
    return (
        visibility is AttentionVisibility.SLIDING_WINDOW
        and span == 255
        and not high_precision_output
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[0] == 1
        and q.shape[2] == 16
        and k.shape[2] in (1, 2, 8)
        and v.shape[3] == 4
        and torch.cuda.get_device_name(q.device) == "NVIDIA H200"
    )


@dataclass(frozen=True)
class _SoftDeltaPlan:
    primary: AttentionSemantics
    correction: AttentionSemantics


@dataclass(frozen=True)
class _SoftDeltaBosPlan:
    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    fwd_use_varlen: bool
    bwd_use_varlen: bool


def _make_plan(
    visibility: AttentionVisibility,
    span: Optional[int],
    reset_position_per_segment: bool,
) -> _SoftDeltaPlan:
    common = {
        "visibility": visibility,
        "span": span,
        "reset_position_per_segment": reset_position_per_segment,
    }
    return _SoftDeltaPlan(
        primary=AttentionSemantics(
            boundary=AttentionBoundary.INCLUSIVE,
            **common,
        ),
        correction=AttentionSemantics(
            boundary=AttentionBoundary.STRICT_PAST,
            **common,
        ),
    )


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("SoftDelta backend must be a string")
    normalized = backend.strip().lower().replace("-", "_")
    if normalized not in ("", "auto", "torch", "sm90"):
        raise ValueError(
            "SoftDelta backend must be one of: auto, torch, sm90"
        )
    return "auto" if normalized == "" else normalized


def _use_sm90_backend(backend: str, q: Tensor) -> bool:
    if backend == "torch":
        return False
    available = (
        q.is_cuda
        and torch.cuda.get_device_capability(q.device) == (9, 0)
    )
    if backend == "sm90":
        if not available:
            raise RuntimeError("Flash SoftDelta requires an SM90 CUDA tensor")
        return True
    return available


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    deterministic: bool,
    api_name: str,
    *,
    allow_short_q: bool = False,
) -> Tuple[int, int, int, int]:
    if not isinstance(deterministic, bool):
        raise TypeError(f"{api_name} deterministic must be a bool")
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(f"{api_name} expects 4D q/k tensors")
    if v.dim() != 5 or g.dim() != 5:
        raise ValueError(f"{api_name} expects 5D v/g tensors")
    if q.device != k.device or q.device != v.device or q.device != g.device:
        raise ValueError(f"{api_name} q/k/v/g must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != g.dtype:
        raise ValueError(f"{api_name} q/k/v/g must share a dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{api_name} supports only fp16 and bf16 inputs")

    batch, q_length, packed_q_heads, qk_dim = q.shape
    if batch <= 0 or q_length <= 0 or qk_dim <= 0:
        raise ValueError(f"{api_name} q dimensions must be positive")
    if packed_q_heads <= 0 or packed_q_heads % 2 != 0:
        raise ValueError(
            f"{api_name} q head axis must contain interleaved q1/q2 pairs"
        )
    q_heads = packed_q_heads // 2
    if k.shape[0] != batch or k.shape[:2] != v.shape[:2]:
        raise ValueError(f"{api_name} batch dimensions and k/v lengths must match")
    if q_length != k.shape[1] and not (allow_short_q and q_length <= k.shape[1]):
        raise ValueError(
            f"{api_name} requires q length <= k/v length" if allow_short_q else
            f"{api_name} current q/k lengths must match"
        )
    kv_heads = k.shape[2]
    if kv_heads <= 0 or v.shape[2] != kv_heads:
        raise ValueError(f"{api_name} k/v head counts must match and be positive")
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"{api_name} logical q head count must be divisible by KV heads"
        )
    if k.shape[-1] != qk_dim:
        raise ValueError(f"{api_name} q/k head dimensions must match")

    groups, group_dim = v.shape[3:]
    if groups <= 0 or group_dim <= 0:
        raise ValueError(f"{api_name} value group dimensions must be positive")
    expected_gate = (batch, q_length, q_heads, groups, 1)
    if tuple(g.shape) != expected_gate:
        raise ValueError(
            f"{api_name} g must have shape {list(expected_gate)}"
        )

    if (prev_k is None) != (prev_v is None):
        raise ValueError(f"{api_name} prev_k and prev_v must be provided together")
    if prev_k is None:
        return q_heads, groups, group_dim, 0
    assert prev_v is not None
    if prev_k.dim() != 4 or prev_v.dim() != 5:
        raise ValueError(f"{api_name} expects 4D prev_k and 5D prev_v")
    if prev_k.device != q.device or prev_v.device != q.device:
        raise ValueError(f"{api_name} previous and current tensors must share a device")
    if prev_k.dtype != q.dtype or prev_v.dtype != q.dtype:
        raise ValueError(f"{api_name} previous and current tensors must share a dtype")
    prev_length = prev_k.shape[1]
    if prev_length <= 0:
        raise ValueError(f"{api_name} previous sequence length must be positive")
    if tuple(prev_k.shape) != (batch, prev_length, kv_heads, qk_dim):
        raise ValueError(f"{api_name} prev_k shape is incompatible with q/k")
    if tuple(prev_v.shape) != (
        batch,
        prev_length,
        kv_heads,
        groups,
        group_dim,
    ):
        raise ValueError(f"{api_name} prev_v shape is incompatible with v")
    return q_heads, groups, group_dim, prev_length


def _validate_ordered_segments(segment_idx: Tensor, name: str) -> None:
    if segment_idx.shape[1] > 1 and bool(
        (segment_idx[:, 1:] < segment_idx[:, :-1]).any().item()
    ):
        raise ValueError(f"{name} must contain nondecreasing contiguous runs")


def _prepare_partition(
    q: Tensor,
    k_length: int,
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    segment_idx: Optional[Tensor],
    bos_mask: Optional[Tensor],
    api_name: str,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if bos_mask is not None:
        if (
            segment_idx is not None
            or q_segment_idx is not None
            or k_segment_idx is not None
        ):
            raise ValueError(
                f"{api_name} BOS masks and segment indices are mutually exclusive"
            )
        expected_shape = (q.shape[0], k_length)
        if (
            bos_mask.device != q.device
            or bos_mask.dtype != torch.bool
            or tuple(bos_mask.shape) != expected_shape
        ):
            raise ValueError(
                f"{api_name} bos_mask must be a bool tensor on the input device "
                f"with shape {list(expected_shape)}"
            )
        if not bool(bos_mask[:, 0].all().item()):
            raise ValueError(f"{api_name} bos_mask must mark position zero")
        k_segment_idx = bos_mask.cumsum(dim=1, dtype=torch.int64)
        q_segment_idx = k_segment_idx[:, -q.shape[1] :]
    else:
        q_segment_idx, k_segment_idx = resolve_segment_indices(
            segment_idx,
            q_segment_idx,
            k_segment_idx,
            batch_size=q.shape[0],
            q_length=q.shape[1],
            k_length=k_length,
        )
        validate_segment_indices(
            q_segment_idx,
            k_segment_idx,
            batch_size=q.shape[0],
            q_length=q.shape[1],
            k_length=k_length,
            device=q.device,
            api_name=api_name,
        )

    if q_segment_idx is not None:
        assert k_segment_idx is not None
        _validate_ordered_segments(q_segment_idx, f"{api_name} q_segment_idx")
        _validate_ordered_segments(k_segment_idx, f"{api_name} k_segment_idx")
    return q_segment_idx, k_segment_idx


def _prepare_flash_bos_plan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    plan: _SoftDeltaPlan,
    k_length: int,
    bos_mask: Tensor,
    deterministic: bool,
    needs_backward: bool,
    api_name: str,
    high_precision_output: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[_SoftDeltaBosPlan]]:
    validate_bos_mask(bos_mask, q, k_length)
    if not bool(bos_mask[:, 0].all().item()):
        raise ValueError(f"{api_name} bos_mask must mark position zero")
    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_q_runs,
        num_k_runs,
        metadata_is_dense,
    ) = flash_softdelta_bos_metadata(q, bos_mask, k_length)
    if metadata_is_dense:
        return None, None, None

    span = (
        k_length - 1
        if plan.primary.visibility is AttentionVisibility.CAUSAL_FULL
        else plan.primary.span
    )
    assert span is not None
    sliding_chunk = (
        plan.primary.visibility is AttentionVisibility.SLIDING_CHUNK
    )
    if not sliding_chunk and requires_row_aligned_k_runs(
        q, cu_seqlens_q, cu_seqlens_k
    ):
        fwd_use_varlen = bwd_use_varlen = False
    else:
        fwd_use_varlen, bwd_use_varlen = select_flash_softdelta_bos_routes(
            q[:, :, 0::2],
            k,
            v.flatten(3),
            span,
            num_q_runs,
            max_seqlen_q,
            deterministic,
            needs_backward,
            sliding_chunk=sliding_chunk,
            full_visibility=(
                plan.primary.visibility
                is AttentionVisibility.CAUSAL_FULL
            ),
            reset_chunk_pos_per_seq=plan.primary.reset_position_per_segment,
            has_previous=k_length > k.shape[1],
            high_precision_output=high_precision_output,
            previous_length=k_length - k.shape[1],
            num_k_runs=num_k_runs,
            max_seqlen_k=max_seqlen_k,
        )
    q_segment_idx = None
    k_segment_idx = None
    if not fwd_use_varlen or not bwd_use_varlen:
        k_segment_idx = flash_softdelta_bos_segment_idx(bos_mask)
        q_segment_idx = k_segment_idx[:, -q.shape[1] :]
    return (
        q_segment_idx,
        k_segment_idx,
        _SoftDeltaBosPlan(
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            fwd_use_varlen,
            bwd_use_varlen,
        ),
    )


def _combine_softdelta_reads(
    read: Tensor,
    past_prediction: Tensor,
    g: Tensor,
    q_heads: int,
    groups: int,
    group_dim: int,
) -> Tensor:
    output_dtype = g.dtype
    output_shape = (
        read.shape[0],
        read.shape[1],
        q_heads,
        groups,
        group_dim,
    )
    accumulation_dtype = (
        torch.float64 if read.dtype == torch.float64 else torch.float32
    )
    read = read.reshape(output_shape).to(accumulation_dtype)
    past_prediction = past_prediction.reshape(output_shape).to(
        accumulation_dtype
    )
    gate = torch.sigmoid(g.to(accumulation_dtype))
    return (read - gate * past_prediction).to(output_dtype)


def _combine_flash_softdelta_reads(
    read: Tensor,
    past_prediction: Tensor,
    g: Tensor,
    q_heads: int,
    groups: int,
    group_dim: int,
) -> Tensor:
    output_shape = (
        read.shape[0],
        read.shape[1],
        q_heads,
        groups,
        group_dim,
    )
    return flash_softdelta_gate(
        read.reshape(output_shape),
        past_prediction.reshape(output_shape),
        g,
    )


def _flash_window_strict_past_read(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    deterministic: bool,
    high_precision_output: bool,
) -> Tensor:
    if window_size == 0:
        return torch.zeros(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            v.shape[-1],
            dtype=v.dtype,
            device=v.device,
        )
    return flash_window_strict_past_read(
        q,
        k,
        v,
        window_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        deterministic,
        high_precision_output,
    )


def _execute_flash_softdelta_reads(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    plan: _SoftDeltaPlan,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    deterministic: bool,
    bos_plan: Optional[_SoftDeltaBosPlan] = None,
    high_precision_output: bool = False,
) -> Tuple[Tensor, Tensor]:
    flat_v = v.flatten(3)
    flat_prev_v = None if prev_v is None else prev_v.flatten(3)
    if bos_plan is not None:
        if plan.primary.visibility is AttentionVisibility.SLIDING_CHUNK:
            assert plan.primary.span is not None
            read = flash_chunk_bos_read(
                q[:, :, 0::2],
                k,
                flat_v,
                plan.primary.span,
                scale,
                prev_k,
                flat_prev_v,
                q_segment_idx,
                k_segment_idx,
                bos_plan.cu_seqlens_q,
                bos_plan.cu_seqlens_k,
                bos_plan.max_seqlen_q,
                bos_plan.max_seqlen_k,
                plan.primary.reset_position_per_segment,
                bos_plan.fwd_use_varlen,
                bos_plan.bwd_use_varlen,
                False,
                deterministic,
                high_precision_output,
            )
            past_prediction = flash_chunk_bos_read(
                q[:, :, 1::2],
                k,
                flat_v,
                plan.primary.span,
                scale,
                prev_k,
                flat_prev_v,
                q_segment_idx,
                k_segment_idx,
                bos_plan.cu_seqlens_q,
                bos_plan.cu_seqlens_k,
                bos_plan.max_seqlen_q,
                bos_plan.max_seqlen_k,
                plan.primary.reset_position_per_segment,
                bos_plan.fwd_use_varlen,
                bos_plan.bwd_use_varlen,
                True,
                deterministic,
                high_precision_output,
            )
            return read, past_prediction

        primary_window = (
            None
            if plan.primary.visibility is AttentionVisibility.CAUSAL_FULL
            else plan.primary.span
        )
        read = flash_window_bos_read(
            q[:, :, 0::2],
            k,
            flat_v,
            primary_window,
            scale,
            prev_k,
            flat_prev_v,
            q_segment_idx,
            k_segment_idx,
            bos_plan.cu_seqlens_q,
            bos_plan.cu_seqlens_k,
            bos_plan.max_seqlen_q,
            bos_plan.max_seqlen_k,
            bos_plan.fwd_use_varlen,
            bos_plan.bwd_use_varlen,
            False,
            deterministic,
            high_precision_output,
        )
        if primary_window == 0:
            past_prediction = torch.zeros_like(read)
        else:
            past_prediction = flash_window_bos_read(
                q[:, :, 1::2],
                k,
                flat_v,
                primary_window,
                scale,
                prev_k,
                flat_prev_v,
                q_segment_idx,
                k_segment_idx,
                bos_plan.cu_seqlens_q,
                bos_plan.cu_seqlens_k,
                bos_plan.max_seqlen_q,
                bos_plan.max_seqlen_k,
                bos_plan.fwd_use_varlen,
                bos_plan.bwd_use_varlen,
                True,
                deterministic,
                high_precision_output,
            )
        return read, past_prediction

    if plan.primary.visibility is AttentionVisibility.SLIDING_CHUNK:
        assert plan.primary.span is not None
        read = flash_chunk_inclusive_read(
            q[:, :, 0::2],
            k,
            flat_v,
            plan.primary.span,
            scale,
            prev_k,
            flat_prev_v,
            q_segment_idx,
            k_segment_idx,
            plan.primary.reset_position_per_segment,
            deterministic,
            high_precision_output,
        )
        past_prediction = flash_chunk_strict_past_read(
            q[:, :, 1::2],
            k,
            flat_v,
            plan.primary.span,
            scale,
            prev_k,
            flat_prev_v,
            q_segment_idx,
            k_segment_idx,
            plan.primary.reset_position_per_segment,
            deterministic,
            high_precision_output,
        )
        return read, past_prediction

    if plan.primary.visibility is AttentionVisibility.CAUSAL_FULL:
        primary_window = None
    else:
        assert plan.primary.visibility is AttentionVisibility.SLIDING_WINDOW
        assert plan.primary.span is not None
        primary_window = plan.primary.span

    read = flash_window_inclusive_read(
        q[:, :, 0::2],
        k,
        flat_v,
        primary_window,
        scale,
        prev_k,
        flat_prev_v,
        q_segment_idx,
        k_segment_idx,
        deterministic,
        high_precision_output,
    )
    past_prediction = _flash_window_strict_past_read(
        q[:, :, 1::2],
        k,
        flat_v,
        primary_window,
        scale,
        prev_k,
        flat_prev_v,
        q_segment_idx,
        k_segment_idx,
        deterministic,
        high_precision_output,
    )
    return read, past_prediction


def _paired_training_route(
    q, k, v, plan, span, prev_k, prev_v, q_segment_idx, k_segment_idx,
    bos_plan, deterministic, high_precision_output,
):
    """Share the production paired-training predicates with standalone APIs."""
    supports_dense_fused_forward = (
        span >= 0
        and (q.shape[1] == k.shape[1] or plan.primary.visibility is AttentionVisibility.CAUSAL_FULL)
        and bos_plan is None
        and prev_k is None
        and prev_v is None
        and q_segment_idx is None
        and k_segment_idx is None
        and not plan.primary.reset_position_per_segment
        and q.shape[3] in _FUSED_HEAD_DIMS
        and v.shape[3] * v.shape[4] in _FUSED_HEAD_DIMS
    )
    use_head_pair_parallel = (
        supports_dense_fused_forward
        and _use_head_pair_parallel(
            q,
            k,
            v,
            plan.primary.visibility,
            span,
            deterministic,
        )
    )
    supports_cta_reader_pair_training = (
        supports_dense_fused_forward
        and plan.primary.visibility is not AttentionVisibility.CAUSAL_FULL
    )
    use_cta_reader_pair_kv_reuse = (
        supports_cta_reader_pair_training
        and _use_cta_reader_pair_kv_reuse(
            q,
            k,
            v,
            plan.primary.visibility,
            span,
            deterministic,
            high_precision_output,
        )
    )
    use_window_reader_pair_training = (
        supports_dense_fused_forward
        and _use_window_reader_pair_training(
            q,
            k,
            v,
            plan.primary.visibility,
            span,
            deterministic,
            high_precision_output,
        )
    )
    return supports_dense_fused_forward, (
        use_head_pair_parallel or use_cta_reader_pair_kv_reuse
        or use_window_reader_pair_training
    )


def _execute_flash_softdelta(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    plan: _SoftDeltaPlan,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    bos_plan: Optional[_SoftDeltaBosPlan],
    deterministic: bool,
    q_heads: int,
    groups: int,
    group_dim: int,
    high_precision_output: bool,
) -> Tensor:
    total_k_length = k.shape[1] + (
        0 if prev_k is None else prev_k.shape[1]
    )
    if plan.primary.visibility is AttentionVisibility.CAUSAL_FULL:
        span = total_k_length - 1
    else:
        assert plan.primary.span is not None
        span = plan.primary.span
    visibility = {
        AttentionVisibility.CAUSAL_FULL: 0,
        AttentionVisibility.SLIDING_WINDOW: 1,
        AttentionVisibility.SLIDING_CHUNK: 2,
    }[plan.primary.visibility]
    matched_reset = visibility == 2 and needs_matched_reset(
        q, q_segment_idx, k_segment_idx, plan.primary.reset_position_per_segment)
    if matched_reset:
        from ._softdelta_attention import _StandaloneSoftDelta
        return _StandaloneSoftDelta.apply(q, k, v, g, prev_k, prev_v, dict(
            visibility=plan.primary.visibility, span=span, scale=scale,
            q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
            segment_idx=None, bos_mask=None, backend="sm90",
            reset=plan.primary.reset_position_per_segment, deterministic=deterministic,
            high_precision_output=high_precision_output, api_name="softdelta_attention"))
    use_composed_backward = (
        bos_plan is None
        or (
            not bos_plan.fwd_use_varlen
            and not bos_plan.bwd_use_varlen
        )
    )
    supports_dense_fused_forward, paired_training = _paired_training_route(
        q, k, v, plan, span, prev_k, prev_v, q_segment_idx, k_segment_idx,
        bos_plan, deterministic, high_precision_output,
    )
    if torch.is_grad_enabled() and paired_training:
        return flash_softdelta_paired_training(
            q,
            k,
            v,
            g,
            span,
            scale,
            visibility,
            deterministic,
            high_precision_output,
        )
    if (
        torch.is_grad_enabled()
        and span > 0
        and use_composed_backward
        and not high_precision_output
    ):
        return flash_softdelta_composed(
            q,
            k,
            v,
            g,
            span,
            scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            visibility,
            plan.primary.reset_position_per_segment,
            deterministic,
        )
    use_fused_forward = (
        supports_dense_fused_forward
        and span > 0
        and not high_precision_output
    )
    if use_fused_forward:
        return flash_softdelta_fwd(
            q,
            k,
            v,
            g,
            span,
            scale,
            visibility,
        )
    read, past_prediction = _execute_flash_softdelta_reads(
        q,
        k,
        v,
        plan,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        deterministic,
        bos_plan,
        high_precision_output,
    )
    return _combine_flash_softdelta_reads(
        read,
        past_prediction,
        g,
        q_heads,
        groups,
        group_dim,
    )


def _execute_softdelta_plan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    plan: _SoftDeltaPlan,
    scale: Optional[float],
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    segment_idx: Optional[Tensor],
    bos_mask: Optional[Tensor],
    backend: str,
    deterministic: bool,
    high_precision_output: bool,
    api_name: str,
) -> Tensor:
    normalized_backend = _normalize_backend(backend)
    q_heads, groups, group_dim, prev_length = _validate_inputs(
        q,
        k,
        v,
        g,
        prev_k,
        prev_v,
        deterministic,
        api_name,
        allow_short_q=True,
    )
    use_fp32_state = use_high_precision_output(
        high_precision_output,
        (q, k, v, prev_k, prev_v),
        api_name=api_name,
    )
    k_length = prev_length + k.shape[1]
    use_sm90 = _use_sm90_backend(normalized_backend, q)
    if use_sm90 and plan.primary.visibility is not AttentionVisibility.CAUSAL_FULL and q.shape[1] != k.shape[1]:
        from ._softdelta_attention import _StandaloneSoftDelta
        return _StandaloneSoftDelta.apply(q, k, v, g, prev_k, prev_v, dict(
            visibility=plan.primary.visibility, span=plan.primary.span, scale=scale,
            q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
            segment_idx=segment_idx, bos_mask=bos_mask, backend="sm90",
            reset=plan.primary.reset_position_per_segment, deterministic=deterministic,
            high_precision_output=use_fp32_state, api_name=api_name))
    bos_plan = None
    if bos_mask is not None and use_sm90:
        if (
            segment_idx is not None
            or q_segment_idx is not None
            or k_segment_idx is not None
        ):
            raise ValueError(
                f"{api_name} BOS masks and segment indices are mutually exclusive"
            )
        needs_backward = torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad
            for tensor in (q, k, v, g, prev_k, prev_v)
        )
        q_segment_idx, k_segment_idx, bos_plan = _prepare_flash_bos_plan(
            q,
            k,
            v,
            plan,
            k_length,
            bos_mask,
            deterministic,
            needs_backward,
            api_name,
            high_precision_output=use_fp32_state,
        )
    else:
        q_segment_idx, k_segment_idx = _prepare_partition(
            q,
            k_length,
            q_segment_idx,
            k_segment_idx,
            segment_idx,
            bos_mask,
            api_name,
        )

    resolved_scale = resolve_scale(q, scale)
    if use_sm90:
        return _execute_flash_softdelta(
            q,
            k,
            v,
            g,
            plan,
            resolved_scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            bos_plan,
            deterministic,
            q_heads,
            groups,
            group_dim,
            use_fp32_state,
        )

    if prev_k is not None:
        assert prev_v is not None
        k = torch.cat((prev_k, k), dim=1)
        v = torch.cat((prev_v, v), dim=1)
    flat_v = v.reshape(v.shape[0], v.shape[1], v.shape[2], -1)
    q1 = q[:, :, 0::2]
    q2 = q[:, :, 1::2]

    mask_kwargs = {
        "batch": q.shape[0],
        "q_length": q.shape[1],
        "k_length": k.shape[1],
        "device": q.device,
        "q_segment_idx": q_segment_idx,
        "k_segment_idx": k_segment_idx,
    }
    primary_mask = build_attention_mask(
        semantics=plan.primary,
        **mask_kwargs,
    )
    correction_mask = build_attention_mask(
        semantics=plan.correction,
        **mask_kwargs,
    )
    read = attention_read(q1, k, flat_v, primary_mask, resolved_scale)
    past_prediction = attention_read(
        q2,
        k,
        flat_v,
        correction_mask,
        resolved_scale,
    )

    return _combine_softdelta_reads(
        read,
        past_prediction,
        g,
        q_heads,
        groups,
        group_dim,
    )


def softdelta_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
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
    """Run causal full-visibility SoftDelta attention.

    ``q`` stores interleaved q1/q2 heads as ``[B, Lq, 2H, D]``. ``k`` is
    ``[B, Lkv, Hkv, D]``, ``v`` is ``[B, Lkv, Hkv, C, Dg]``, and gate
    logits ``g`` are ``[B, Lq, H, C, 1]``, with ``0 < Lq <= Lkv``.
    Query ``i`` aligns to ``i + Lkv - Lq``; an optional previous KV prefix
    increases this offset by its length. The first reader includes that
    position; the second is strictly before it. Segment IDs only isolate
    matching tokens, without resetting
    positions. ``segment_idx`` covers KV and its suffix labels Q; independent
    Q/K segment indices may instead be supplied. No previous KV is required
    for unequal current lengths.

    Set ``high_precision_output=True`` to retain FP32 attention output
    state for backward. The public output dtype is unchanged.
    """
    plan = _make_plan(AttentionVisibility.CAUSAL_FULL, None, False)
    return _execute_softdelta_plan(
        q,
        k,
        v,
        g,
        plan,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        segment_idx,
        bos_mask,
        backend,
        deterministic,
        high_precision_output,
        "softdelta_attention",
    )


def sliding_window_softdelta_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
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
    """Run SoftDelta with an inclusive left-radius sliding window.

    Q may be shorter than current K/V and aligns to total K/V at the right.
    Both readers return zero for rows with no visible keys.

    ``high_precision_output=True`` retains FP32 attention output state for
    backward while leaving the public output dtype unchanged.
    """
    plan = _make_plan(AttentionVisibility.SLIDING_WINDOW, window_size, False)
    return _execute_softdelta_plan(
        q,
        k,
        v,
        g,
        plan,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        segment_idx,
        bos_mask,
        backend,
        deterministic,
        high_precision_output,
        "sliding_window_softdelta_attention",
    )


def sliding_chunk_softdelta_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    deterministic: bool = False,
    high_precision_output: bool = False,
) -> Tensor:
    """Run SoftDelta over the previous and current logical chunks.

    Q may be shorter than current K/V. Positions align globally unless reset
    is enabled, in which case matching segment IDs align at their ends.
    Both readers return zero for rows with no visible keys.

    ``high_precision_output=True`` retains FP32 attention output state for
    backward while leaving the public output dtype unchanged.
    """
    if not isinstance(reset_chunk_pos_per_seq, bool):
        raise TypeError(
            "sliding_chunk_softdelta_attention reset_chunk_pos_per_seq "
            "must be a bool"
        )
    plan = _make_plan(
        AttentionVisibility.SLIDING_CHUNK,
        chunk_size,
        reset_chunk_pos_per_seq,
    )
    if prev_k is not None and prev_k.shape[1] != chunk_size:
        raise ValueError(
            "sliding_chunk_softdelta_attention prev_k/prev_v sequence "
            "length must equal chunk_size"
        )
    return _execute_softdelta_plan(
        q,
        k,
        v,
        g,
        plan,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        segment_idx,
        bos_mask,
        backend,
        deterministic,
        high_precision_output,
        "sliding_chunk_softdelta_attention",
    )


def softdelta_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
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
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(y, readers, lse)`` without an autograd graph.

    Reader states are interleaved ``[B, L, 2H, C, Dv]``; LSE is
    ``[B, 2H, L]``. ``readers`` is FP32 when high precision is requested,
    otherwise the input dtype, including under no_grad/inference mode. Reuse forward
    options for backward.
    """
    from ._softdelta_attention import _softdelta_fwd

    return _softdelta_fwd(
        q, k, v, g,
        visibility=AttentionVisibility.CAUSAL_FULL, span=None,
        scale=scale, prev_k=prev_k, prev_v=prev_v,
        q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
        segment_idx=segment_idx, bos_mask=bos_mask, backend=backend,
        reset=False, deterministic=deterministic,
        high_precision_output=high_precision_output, api_name="softdelta_attention_fwd",
    )


def softdelta_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    readers: Tensor,
    lse: Tensor,
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Return ``(dq, dk, dv, dg, dprev_k, dprev_v)`` without recomputation.

    ``readers`` accepts the input dtype or FP32 and automatically selects
    backward precision. ``y_grad`` must retain the input dtype. Pass the
    matching forward LSE and options; absent previous gradients are None.
    """
    from ._softdelta_attention import _softdelta_bwd

    return _softdelta_bwd(
        y_grad, q, k, v, g, readers, lse,
        visibility=AttentionVisibility.CAUSAL_FULL, span=None,
        scale=scale, prev_k=prev_k, prev_v=prev_v,
        q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
        segment_idx=segment_idx, bos_mask=bos_mask, backend=backend,
        reset=False, deterministic=deterministic,
        high_precision_output=readers.dtype == torch.float32, api_name="softdelta_attention_bwd",
    )


def sliding_window_softdelta_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
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
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(y, readers, lse)`` without an autograd graph.

    Reader states are interleaved ``[B, L, 2H, C, Dv]``; LSE is
    ``[B, 2H, L]``. ``readers`` is FP32 when high precision is requested,
    otherwise the input dtype, including under no_grad/inference mode. Reuse forward
    options for backward.
    """
    from ._softdelta_attention import _softdelta_fwd

    return _softdelta_fwd(
        q, k, v, g,
        visibility=AttentionVisibility.SLIDING_WINDOW, span=window_size,
        scale=scale, prev_k=prev_k, prev_v=prev_v,
        q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
        segment_idx=segment_idx, bos_mask=bos_mask, backend=backend,
        reset=False, deterministic=deterministic,
        high_precision_output=high_precision_output, api_name="sliding_window_softdelta_attention_fwd",
    )


def sliding_window_softdelta_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    readers: Tensor,
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Return ``(dq, dk, dv, dg, dprev_k, dprev_v)`` without recomputation.

    ``readers`` accepts the input dtype or FP32 and automatically selects
    backward precision. ``y_grad`` must retain the input dtype. Pass the
    matching forward LSE and options; absent previous gradients are None.
    """
    from ._softdelta_attention import _softdelta_bwd

    return _softdelta_bwd(
        y_grad, q, k, v, g, readers, lse,
        visibility=AttentionVisibility.SLIDING_WINDOW, span=window_size,
        scale=scale, prev_k=prev_k, prev_v=prev_v,
        q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
        segment_idx=segment_idx, bos_mask=bos_mask, backend=backend,
        reset=False, deterministic=deterministic,
        high_precision_output=readers.dtype == torch.float32, api_name="sliding_window_softdelta_attention_bwd",
    )


def sliding_chunk_softdelta_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    deterministic: bool = False,
    high_precision_output: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(y, readers, lse)`` without an autograd graph.

    Reader states are interleaved ``[B, L, 2H, C, Dv]``; LSE is
    ``[B, 2H, L]``. ``readers`` is FP32 when high precision is requested,
    otherwise the input dtype, including under no_grad/inference mode. Reuse forward
    options for backward.
    """
    from ._softdelta_attention import _softdelta_fwd

    return _softdelta_fwd(
        q, k, v, g,
        visibility=AttentionVisibility.SLIDING_CHUNK, span=chunk_size,
        scale=scale, prev_k=prev_k, prev_v=prev_v,
        q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
        segment_idx=segment_idx, bos_mask=bos_mask, backend=backend,
        reset=reset_chunk_pos_per_seq, deterministic=deterministic,
        high_precision_output=high_precision_output, api_name="sliding_chunk_softdelta_attention_fwd",
    )


def sliding_chunk_softdelta_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    readers: Tensor,
    lse: Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    deterministic: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Return ``(dq, dk, dv, dg, dprev_k, dprev_v)`` without recomputation.

    ``readers`` accepts the input dtype or FP32 and automatically selects
    backward precision. ``y_grad`` must retain the input dtype. Pass the
    matching forward LSE and options; absent previous gradients are None.
    """
    from ._softdelta_attention import _softdelta_bwd

    return _softdelta_bwd(
        y_grad, q, k, v, g, readers, lse,
        visibility=AttentionVisibility.SLIDING_CHUNK, span=chunk_size,
        scale=scale, prev_k=prev_k, prev_v=prev_v,
        q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
        segment_idx=segment_idx, bos_mask=bos_mask, backend=backend,
        reset=reset_chunk_pos_per_seq, deterministic=deterministic,
        high_precision_output=readers.dtype == torch.float32, api_name="sliding_chunk_softdelta_attention_bwd",
    )

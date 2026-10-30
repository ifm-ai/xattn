# Author: Shicheng Wen

"""SoftDelta attention: a negative read shared by each KV group.

Per query head ``h`` with KV head ``kv(h)``::

    read_h      = Attention(q_h,      K, V; inclusive causal support)
    neg_read_kv = Attention(k_read_kv, K, V; strict-past causal support)
    y_h         = read_h - sigmoid(g_h) * neg_read_kv(h)

The negative read runs once per KV head and is shared by the query heads of its
group. Its query ``k_read`` is whatever post-RoPE ``[B, Lq, Hkv, D]`` tensor the
caller passes (the keys, a transformed key, a second query); by default the
aligned keys themselves, whose query-side gradient then folds into ``k``'s
gradient. The forward state is ``y``, ``neg_read`` and the LSE of both reads
(``[B, Hkv + H, Lq]``, the negative reads first). At level 0 the fused forward
writes the reads class-major into one ``[B, Lq, Hkv + H, V]`` buffer and the gate
overwrites the query reads with ``y``: ``y`` and ``neg_read`` are its two head
slices. Level 1 keeps ``neg_read`` in FP32.

The sliding window and sliding chunk variants restrict both reads to the keys of
a causal sliding window or of the query's chunk and the chunk before it; the
negative read keeps only its strictly earlier keys.

The public API takes scalar gates: ``v`` ``[B, Lkv, Hkv, V]`` and one gate logit
per query head and token, ``g`` ``[B, Lq, H]``. The implementation keeps value
groups (``v`` ``[B, Lkv, Hkv, C, Dg]``, ``g`` ``[B, Lq, H, C, 1]``, a gate per
group of ``Dg`` channels); public calls run it with ``C = 1``.
"""

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ._attention_core import (
    AttentionBoundary,
    AttentionSemantics,
    AttentionVisibility,
    attention_read,
    build_attention_mask,
    resolve_scale,
    validate_attention_scale,
    validate_sm90_dimensions,
)
from ._backward_precision import allocate_backward_output_state, validate_high_precision_level
from ._bos_metadata import _bos_plan
from ._bos_routing_heuristics import select_flash_swa_bos_routes
from ._causal_attention import _dense_bwd, _dense_fwd, _extension_ops
from ._sca_segment_alignment import matched_backward, matched_forward, needs_matched_reset
from ._segment_metadata import resolve_segment_indices, validate_segment_indices
from .sliding_chunk_attention import _flash_sca_bwd, _flash_sca_fwd

__all__ = [
    "softdelta_attention",
    "softdelta_attention_fwd",
    "softdelta_attention_bwd",
    "sliding_window_softdelta_attention",
    "sliding_window_softdelta_attention_fwd",
    "sliding_window_softdelta_attention_bwd",
    "sliding_chunk_softdelta_attention",
    "sliding_chunk_softdelta_attention_fwd",
    "sliding_chunk_softdelta_attention_bwd",
]

_API = "softdelta_attention"


@dataclass(frozen=True)
class _Local:
    """Local support of both reads: a sliding window (``span`` the inclusive left radius) or
    sliding chunks (``span`` the chunk size, ``reset`` chunk positions per segment). Causal
    calls pass None instead."""
    kind: AttentionVisibility
    span: int
    reset: bool = False

    @property
    def visibility(self) -> int:
        """The kernels' visibility code: 1 sliding window, 2 sliding chunk."""
        return 1 if self.kind is AttentionVisibility.SLIDING_WINDOW else 2


def _local_support(function, kind: AttentionVisibility, arguments) -> Optional[_Local]:
    """The call's local support (None for causal calls)."""
    if kind is AttentionVisibility.CAUSAL_FULL:
        return None
    if kind is AttentionVisibility.SLIDING_WINDOW:
        span, reset = arguments["window_size"], False
    else:
        span, reset = arguments["chunk_size"], arguments["reset_chunk_pos_per_seq"]
        if not isinstance(reset, bool):
            raise TypeError(f"{function.__name__} reset_chunk_pos_per_seq must be a bool")
    AttentionSemantics(visibility=kind, boundary=AttentionBoundary.INCLUSIVE, span=span)
    return _Local(kind, span, reset)


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    k_read: Optional[Tensor],
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    deterministic: bool,
    api_name: str = _API,
) -> int:
    """Return ``prev_length`` after validating the scalar-gate layouts."""
    if not isinstance(deterministic, bool):
        raise TypeError(f"{api_name} deterministic must be a bool")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"{api_name} expects 4D q/k/v tensors")
    tensors = (k, v, g) if k_read is None else (k, v, g, k_read)
    device, dtype = q.device, q.dtype
    if any(t.device != device for t in tensors):
        raise ValueError(f"{api_name} q/k/v/g/k_read must share a device")
    if any(t.dtype != dtype for t in tensors):
        raise ValueError(f"{api_name} q/k/v/g/k_read must share a dtype")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{api_name} supports only fp16 and bf16 inputs")

    batch, q_length, q_heads, qk_dim = q.shape
    if batch <= 0 or q_length <= 0 or q_heads <= 0 or qk_dim <= 0:
        raise ValueError(f"{api_name} q dimensions must be positive")
    if k.shape[0] != batch or k.shape[:2] != v.shape[:2]:
        raise ValueError(f"{api_name} batch dimensions and k/v lengths must match")
    if q_length > k.shape[1]:
        raise ValueError(f"{api_name} requires q length <= current k/v length")
    kv_heads = k.shape[2]
    if kv_heads <= 0 or v.shape[2] != kv_heads:
        raise ValueError(f"{api_name} k/v head counts must match and be positive")
    if q_heads % kv_heads != 0:
        raise ValueError(f"{api_name} q head count must be divisible by KV heads")
    if k.shape[-1] != qk_dim:
        raise ValueError(f"{api_name} q/k head dimensions must match")
    if k_read is not None and tuple(k_read.shape) != (batch, q_length, kv_heads, qk_dim):
        raise ValueError(
            f"{api_name} k_read must have shape {[batch, q_length, kv_heads, qk_dim]}"
        )

    value_dim = v.shape[3]
    if value_dim <= 0:
        raise ValueError(f"{api_name} value dimension must be positive")
    if tuple(g.shape) != (batch, q_length, q_heads):
        raise ValueError(f"{api_name} g must have shape {[batch, q_length, q_heads]}")

    if (prev_k is None) != (prev_v is None):
        raise ValueError(f"{api_name} prev_k and prev_v must be provided together")
    if prev_k is None:
        return 0
    assert prev_v is not None
    if prev_k.device != q.device or prev_v.device != q.device:
        raise ValueError(f"{api_name} previous and current tensors must share a device")
    if prev_k.dtype != q.dtype or prev_v.dtype != q.dtype:
        raise ValueError(f"{api_name} previous and current tensors must share a dtype")
    prev_length = prev_k.shape[1] if prev_k.dim() == 4 else 0
    if prev_length <= 0:
        raise ValueError(f"{api_name} previous sequence length must be positive")
    if tuple(prev_k.shape) != (batch, prev_length, kv_heads, qk_dim):
        raise ValueError(f"{api_name} prev_k shape is incompatible with q/k")
    if tuple(prev_v.shape) != (batch, prev_length, kv_heads, value_dim):
        raise ValueError(f"{api_name} prev_v shape is incompatible with v")
    return prev_length


def _grouped(v: Tensor, g: Tensor, prev_v: Optional[Tensor]):
    """The scalar-gate tensors in the internal value-group layout, as views: one group of V
    channels (``v`` ``[B, L, Hkv, 1, V]``, ``g`` ``[B, Lq, H, 1, 1]``)."""
    return v.unsqueeze(3), g[..., None, None], None if prev_v is None else prev_v.unsqueeze(3)


def _prepare_segments(
    q: Tensor,
    k_length: int,
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    segment_idx: Optional[Tensor],
    bos_mask: Optional[Tensor],
    api_name: str = _API,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Resolve segment metadata to Q/K segment ids without reading device values."""
    if bos_mask is None and segment_idx is None and q_segment_idx is None and k_segment_idx is None:
        return None, None
    if bos_mask is not None:
        if segment_idx is not None or q_segment_idx is not None or k_segment_idx is not None:
            raise ValueError(f"{api_name} BOS masks and segment indices are mutually exclusive")
        if (
            bos_mask.device != q.device
            or bos_mask.dtype != torch.bool
            or tuple(bos_mask.shape) != (q.shape[0], k_length)
        ):
            raise ValueError(
                f"{api_name} bos_mask must be a bool tensor on the input device "
                f"with shape {[q.shape[0], k_length]}"
            )
        k_segment_idx = bos_mask.cumsum(dim=1, dtype=torch.int64)
        return k_segment_idx[:, k_length - q.shape[1]:].contiguous(), k_segment_idx
    q_segment_idx, k_segment_idx = resolve_segment_indices(
        segment_idx, q_segment_idx, k_segment_idx,
        batch_size=q.shape[0], q_length=q.shape[1], k_length=k_length,
    )
    validate_segment_indices(
        q_segment_idx, k_segment_idx,
        batch_size=q.shape[0], q_length=q.shape[1], k_length=k_length,
        device=q.device, api_name=api_name,
    )
    # The fused launches read contiguous ids (segment_idx gives a Q suffix view when Lq < Lk).
    return q_segment_idx.contiguous(), k_segment_idx.contiguous()


def _aligned_keys(k: Tensor, q_length: int) -> Tensor:
    """The keys at the query positions: the last ``q_length`` current keys."""
    k_length = k.shape[1]
    return k if k_length == q_length else k[:, k_length - q_length:]


def _tma_readable(t: Tensor) -> Tensor:
    """``t`` itself when TMA can address its rows (unit last-dim stride, 16-byte aligned base and
    strides; the fused launches read such views in place), otherwise a contiguous copy."""
    size = t.element_size()
    if (t.stride(-1) == 1 and t.data_ptr() % 16 == 0
            and all(stride * size % 16 == 0 for stride in t.stride()[:-1])):
        return t
    return t.contiguous()


@functools.lru_cache(maxsize=None)
def _sm90_device(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (9, 0)


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("SoftDelta backend must be a string")
    normalized = backend.strip().lower().replace("-", "_")
    if normalized not in ("", "auto", "torch", "sm90"):
        raise ValueError("SoftDelta backend must be one of: auto, torch, sm90")
    return "auto" if normalized == "" else normalized


def _use_sm90_backend(backend: str, q: Tensor, api_name: str = _API) -> bool:
    """The backend rule with the device capability cached per device."""
    if backend == "torch":
        return False
    available = q.is_cuda and _sm90_device(q.get_device())
    if backend == "sm90":
        if not available:
            raise RuntimeError(f"{api_name} requires an SM90 CUDA tensor")
        return True
    return available


def _gate_combine(read: Tensor, neg_read: Tensor, g: Tensor) -> Tensor:
    """``read - sigmoid(g) * rep(neg_read)`` in FP32 (FP64 for FP64 reads).

    read ``[B, L, H, C, Dg]``, neg_read ``[B, L, Hkv, C, Dg]``, g ``[B, L, H, C, 1]``.
    """
    batch, length, q_heads, groups, group_dim = read.shape
    kv_heads = neg_read.shape[2]
    rep = q_heads // kv_heads
    accumulation = torch.float64 if read.dtype == torch.float64 else torch.float32
    gate = torch.sigmoid(g.to(accumulation)).view(batch, length, kv_heads, rep, groups, 1)
    y = read.to(accumulation).view(batch, length, kv_heads, rep, groups, group_dim)
    y = y - gate * neg_read.to(accumulation).unsqueeze(3)
    return y.view(batch, length, q_heads, groups, group_dim).to(g.dtype)


def _fused_ops(q: Tensor, v: Tensor, prev_k: Optional[Tensor], local: Optional[_Local] = None,
               k_length: int = 0):
    """Extension ops with the fused reads for this call, or None (composition).

    The fused SM90 instances cover every D, V <= 256 (multiples of 8, rounded up to
    the causal_flash_attn buckets; csrc/softdelta/instantiations) with gate
    groups of a multiple of 8 channels; sliding windows / chunks need equal Q / K lengths.
    """
    dim, width = q.shape[-1], v.shape[3] * v.shape[4]
    if (
        prev_k is not None
        or not (0 < dim <= 256 and 0 < width <= 256)
        or dim % 8 != 0
        or v.shape[4] % 8 != 0
        or (local is not None and q.shape[1] != k_length)
    ):
        return None
    ops = _extension_ops()
    if not hasattr(ops, "_softdelta_fwd") or (local is not None and not _fused_local(ops)):
        return None
    return ops


@functools.lru_cache(maxsize=None)
def _fused_local(ops) -> bool:
    """Whether the extension's fused reads take a visibility (sliding window / chunk)."""
    return "visibility" in (ops._softdelta_fwd.__doc__ or "")


def _fused_local_args(local: Optional[_Local]) -> tuple:
    """The fused ops' trailing (visibility, span, reset) arguments; none for causal calls."""
    return () if local is None else (local.visibility, local.span, local.reset)


def _effective_local(local: Optional[_Local], k_length: int) -> Optional[_Local]:
    """``local``, or None (causal) when it covers every earlier key: a window of at least
    ``k_length - 1`` keys or a chunk of at least ``k_length`` (total K/V length)."""
    if local is None or local.span < (k_length - 1 if local.visibility == 1 else k_length):
        return local
    return None


def _add_optional(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
    if a is None:
        return b
    if b is None:
        return a
    return (a.float() + b.float()).to(a.dtype)


def _matched_reset(q, q_segment_idx, k_segment_idx, local: Optional[_Local]) -> bool:
    return (local is not None and local.visibility == 2
            and needs_matched_reset(q, q_segment_idx, k_segment_idx, local.reset))


def _local_reader(q, k, v, local, scale, prev_k, prev_v, q_segment_idx, k_segment_idx, strict, deterministic,
                  *, state=None, dy=None, y=None, lse=None, fp32_kv_grads=False):
    """One read over ``local`` support through the sliding window / chunk SM90 routes: the forward
    ``(output, lse)`` (``state`` receives the FP32 output), or with ``dy`` / ``y`` / ``lse`` the
    backward ``(dq, dk, dv, dprev_k, dprev_v)``, dK / dV unrounded in FP32 with ``fp32_kv_grads``
    (grouped heads, not on matched reset chunks)."""
    backward = dy is not None
    prefix = (dy, q, k, v, y, lse) if backward else (q, k, v)
    if local.visibility == 2:
        if _matched_reset(q, q_segment_idx, k_segment_idx, local):
            if fp32_kv_grads:
                raise ValueError("matched reset chunks have no FP32 dK / dV output")
            if backward:
                return matched_backward(
                    dy, q, k, v, y, lse, local.span, scale, prev_k, prev_v, q_segment_idx,
                    k_segment_idx, "sm90", "default", deterministic, strict_past=strict,
                )
            output, saved, out_lse = matched_forward(
                q, k, v, local.span, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
                "sm90", "default", strict_past=strict, fp32_state=state is not None,
            )
            if state is not None:
                state.copy_(saved)
            return output, out_lse
        args = (*prefix, local.span, scale, prev_k, prev_v, q_segment_idx, k_segment_idx)
        if backward:
            return _flash_sca_bwd(*args, deterministic, "sm90", local.reset, strict_past=strict,
                                  fp32_kv_grads=fp32_kv_grads)
        return _flash_sca_fwd(*args, "sm90", local.reset, strict_past=strict, output_state=state)
    args = (*prefix, local.span, scale, prev_k, prev_v, q_segment_idx, k_segment_idx)
    if backward:
        return _dense_bwd(*args, deterministic, "sm90", strict_past=strict, fp32_kv_grads=fp32_kv_grads)
    return _dense_fwd(*args, "sm90", strict_past=strict, use_fast_reciprocal=False, output_state=state)


def _forward(q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx, scale, level, varlen_plan,
             local=None):
    """Both reads and the gate: ``(y, neg_read, lse)``.

    ``y`` ``[B, Lq, H, C, Dg]`` (the input dtype), ``neg_read`` ``[B, Lq, Hkv, C, Dg]`` (the input dtype,
    FP32 at level 1) and ``lse`` ``[B, Hkv + H, Lq]`` (the negative reads first). At level 0 the fused
    launch writes both reads class-major into one ``[B, Lq, Hkv + H, C * Dg]`` buffer and the gate
    overwrites the query reads with ``y`` in place; ``y`` and ``neg_read`` are head slices of that
    buffer. Level 1 reads in FP32 and gates once into ``y``; the FP32 query reads are dropped.
    """
    batch, length, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    groups, group_dim = v.shape[3:]
    flat_v = v.flatten(3)
    if k_read is None:
        k_read = _aligned_keys(k, length)
    ops = _extension_ops()
    g = g.contiguous()
    fused = _fused_ops(q, v, prev_k, local, k.shape[1])
    if fused is not None:
        q, k, flat_v, k_read = (_tma_readable(t) for t in (q, k, flat_v, k_read))
        state = None
        if level:
            state = torch.empty(
                batch, length, kv_heads + q_heads, groups * group_dim,
                dtype=torch.float32, device=q.device,
            )
        if varlen_plan is not None and state is None:
            # Packed sequences: both reads as varlen launches into the readers.
            readers, lse = fused._softdelta_fwd_varlen(
                q, k, flat_v, k_read, varlen_plan[0], varlen_plan[1], scale,
            )
        else:
            readers, lse = fused._softdelta_fwd(
                q, k, flat_v, k_read, scale, q_segment_idx, k_segment_idx, state, *_fused_local_args(local),
            )
        if state is None:
            readers = readers.view(batch, length, kv_heads + q_heads, groups, group_dim)
            y, neg_read = readers[:, :, kv_heads:], readers[:, :, :kv_heads]
            ops._softdelta_gate_fwd(y, neg_read, g, y)
            return y, neg_read, lse
        state = state.view(batch, length, kv_heads + q_heads, groups, group_dim)
        y = ops._softdelta_gate_fwd(state[:, :, kv_heads:], state[:, :, :kv_heads], g)
        return y, state[:, :, :kv_heads].contiguous(), lse
    flat_prev_v = None if prev_v is None else prev_v.flatten(3)
    read_state = neg_read_state = None
    if level:
        read_state = allocate_backward_output_state(q, flat_v)
        neg_read_state = allocate_backward_output_state(k_read, flat_v)
    if local is None:
        common = (scale, prev_k, flat_prev_v, q_segment_idx, k_segment_idx, "sm90")
        read, read_lse = _dense_fwd(q, k, flat_v, None, *common, output_state=read_state)
        neg_read, neg_read_lse = _dense_fwd(
            k_read, k, flat_v, None, *common, strict_past=True, output_state=neg_read_state,
        )
    else:
        common = (local, scale, prev_k, flat_prev_v, q_segment_idx, k_segment_idx)
        read, read_lse = _local_reader(q, k, flat_v, *common, False, False, state=read_state)
        neg_read, neg_read_lse = _local_reader(k_read, k, flat_v, *common, True, False, state=neg_read_state)
    read = read if read_state is None else read_state
    neg_read = (neg_read if neg_read_state is None else neg_read_state).view(
        batch, length, kv_heads, groups, group_dim)
    y = ops._softdelta_gate_fwd(read.view(batch, length, q_heads, groups, group_dim), neg_read, g)
    return y, neg_read, torch.cat((neg_read_lse, read_lse), dim=1)


def _backward(y_grad, q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx,
              y, neg_read, lse, scale, deterministic, level, local=None):
    """``(dq, dk, dv, dg, dk_read, dprev_k, dprev_v)`` from the forward's ``y``, ``neg_read`` and LSE.

    ``dk_read`` is None when ``k_read`` is (the aligned keys' query-side gradient is in ``dk``).
    Level 0 nondeterministic calls with equal Q / K lengths and no previous K/V take the fused
    launch, which reads the query reads' dPsum from ``y`` and ``neg_read``. The others run the gate
    backward, then each read's backward, summed in FP32 at level 1 (the precision-repair backward,
    which reads neither ``y`` nor the LSE); level 0 recovers the query reads from ``y``.
    """
    batch, length, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    groups, group_dim = v.shape[3:]
    flat_v = v.flatten(3)
    flat_prev_v = None if prev_v is None else prev_v.flatten(3)
    fold_k_read = k_read is None
    if fold_k_read:
        k_read = _aligned_keys(k, length)
    y_grad, g = y_grad.contiguous(), g.contiguous()
    ops = _extension_ops()
    fused = _fused_ops(q, v, prev_k, local, k.shape[1])
    if not level and not deterministic and fused is not None and length == k.shape[1]:
        q, k, flat_v, k_read = (_tma_readable(t) for t in (q, k, flat_v, k_read))
        dq, dk, dv, dk_read, dg = fused._softdelta_bwd(
            y_grad, g, q, k, flat_v, k_read, y.flatten(3), neg_read.flatten(3), lse, scale,
            q_segment_idx, k_segment_idx, fold_k_read, _BWD_READER_GROUPS, *_fused_local_args(local),
        )
        return dq, dk, dv.view(v.shape), dg, None if fold_k_read else dk_read, None, None
    neg_read_grad, dg = ops._softdelta_gate_bwd(y_grad, neg_read, g)
    neg_read_grad = neg_read_grad.view(batch, length, kv_heads, groups * group_dim)
    read_grad = y_grad.view(batch, length, q_heads, groups * group_dim)
    if level:
        from ._precision_backend import backward as precision_backward
        common = dict(
            visibility=0 if local is None else local.visibility, span=None if local is None else local.span,
            reset=local is not None and local.reset, scale=scale, prev_k=prev_k, prev_v=flat_prev_v,
            q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
            deterministic=deterministic, return_fp32=True,
        )
        dq, dk, dv, dprev_k, dprev_v = precision_backward(read_grad, q, k, flat_v, **common)
        dk_read, dk_neg, dv_neg, dprev_k_neg, dprev_v_neg = precision_backward(
            neg_read_grad, k_read, k, flat_v, strict=True, **common,
        )
        dk += dk_neg
        dv += dv_neg
        if fold_k_read:
            dk[:, k.shape[1] - length:] += dk_read
            dk_read = None
        dprev_k = _add_optional(dprev_k, dprev_k_neg)
        dprev_v = _add_optional(dprev_v, dprev_v_neg)
        dq, dk, dv, dk_read, dprev_k, dprev_v = (
            None if x is None else x.to(q.dtype) for x in (dq, dk, dv, dk_read, dprev_k, dprev_v)
        )
    else:
        read = ops._softdelta_gate_restore(y, neg_read, g).view(batch, length, q_heads, groups * group_dim)
        neg_read = neg_read.reshape(batch, length, kv_heads, groups * group_dim).contiguous()
        read_lse, neg_read_lse = lse[:, kv_heads:].contiguous(), lse[:, :kv_heads].contiguous()
        # Grouped heads: the query reads' dK / dV stay in their FP32 accumulators, so each merged
        # gradient is rounded once.
        fp32_kv_grads = q_heads > kv_heads and not _matched_reset(q, q_segment_idx, k_segment_idx, local)
        if local is not None:
            common = (local, scale, prev_k, flat_prev_v, q_segment_idx, k_segment_idx)
            dq, dk, dv, dprev_k, dprev_v = _local_reader(
                q, k, flat_v, *common, False, deterministic, dy=read_grad, y=read, lse=read_lse,
                fp32_kv_grads=fp32_kv_grads,
            )
            dk_read, dk_neg, dv_neg, dprev_k_neg, dprev_v_neg = _local_reader(
                k_read, k, flat_v, *common, True, deterministic, dy=neg_read_grad, y=neg_read,
                lse=neg_read_lse,
            )
        else:
            common = (scale, prev_k, flat_prev_v, q_segment_idx, k_segment_idx, deterministic, "sm90")
            dq, dk, dv, dprev_k, dprev_v = _dense_bwd(
                read_grad, q, k, flat_v, read, read_lse, None, *common, fp32_kv_grads=fp32_kv_grads)
            dk_read, dk_neg, dv_neg, dprev_k_neg, dprev_v_neg = _dense_bwd(
                neg_read_grad, k_read, k, flat_v, neg_read, neg_read_lse, None,
                *common, strict_past=True,
            )
        dk, dv = ops._softdelta_merge_gradients(
            dk.contiguous(), dk_neg.contiguous(), dk_read.contiguous() if fold_k_read else None,
            dv.contiguous(), dv_neg.contiguous(),
        )
        if fold_k_read:
            dk_read = None
        if dprev_k is not None and dprev_k_neg is not None:
            dprev_k, dprev_v = ops._softdelta_merge_gradients(
                dprev_k.contiguous(), dprev_k_neg.contiguous(), None,
                dprev_v.contiguous(), dprev_v_neg.contiguous(),
            )
        else:
            dprev_k = _add_optional(dprev_k, dprev_k_neg)
            dprev_v = _add_optional(dprev_v, dprev_v_neg)
    dv = dv.view(v.shape)
    if dprev_v is not None:
        dprev_v = dprev_v.view(prev_v.shape)
    return dq, dk, dv, dg, dk_read, dprev_k, dprev_v


class _SoftDeltaFunc(torch.autograd.Function):
    """SM90 reads (fused launch or composition), gate, and their backward. Level 0 saves ``y``
    (the output itself), ``neg_read`` and the LSE; level 1 saves only the FP32 ``neg_read``."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        k_read: Optional[Tensor],
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        scale: float,
        deterministic: bool,
        high_precision_level: int,
        varlen_plan: Optional[Tuple[Tensor, int]] = None,
        local: Optional[_Local] = None,
    ) -> Tensor:
        y, neg_read, lse = _forward(
            q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx, scale,
            high_precision_level, varlen_plan, local,
        )
        state = (neg_read,) if high_precision_level else (y, neg_read, lse)
        ctx.save_for_backward(q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx, *state)
        ctx.scale = scale
        ctx.deterministic = deterministic
        ctx.high_precision_level = high_precision_level
        ctx.local = local
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y_grad: Tensor) -> tuple:
        q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx, *state = ctx.saved_tensors
        y, neg_read, lse = (None, *state, None) if ctx.high_precision_level else state
        gradients = _backward(
            y_grad, q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx,
            y, neg_read, lse, ctx.scale, ctx.deterministic, ctx.high_precision_level, ctx.local,
        )
        return (*gradients, None, None, None, None, None, None, None)


def _torch_softdelta(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    k_read: Optional[Tensor],
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    scale: float,
    local: Optional[_Local] = None,
) -> Tensor:
    """Explicit correctness plan: materialized masks, FP32 reads and gate."""
    length = q.shape[1]
    if k_read is None:
        k_read = _aligned_keys(k, length)
    if prev_k is not None:
        assert prev_v is not None
        k = torch.cat((prev_k, k), dim=1)
        v = torch.cat((prev_v, v), dim=1)
    batch, _, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    groups, group_dim = v.shape[3:]
    flat_v = v.flatten(3)
    masks = {}
    for boundary in (AttentionBoundary.INCLUSIVE, AttentionBoundary.STRICT_PAST):
        masks[boundary] = build_attention_mask(
            batch=batch,
            q_length=length,
            k_length=k.shape[1],
            device=q.device,
            semantics=AttentionSemantics(
                visibility=AttentionVisibility.CAUSAL_FULL, boundary=boundary
            ) if local is None else AttentionSemantics(
                visibility=local.kind, boundary=boundary, span=local.span,
                reset_position_per_segment=local.reset,
            ),
            q_segment_idx=q_segment_idx,
            k_segment_idx=k_segment_idx,
        )
    read = attention_read(q, k, flat_v, masks[AttentionBoundary.INCLUSIVE], scale)
    neg_read = attention_read(
        k_read, k, flat_v, masks[AttentionBoundary.STRICT_PAST], scale
    )
    return _gate_combine(
        read.reshape(batch, length, q_heads, groups, group_dim),
        neg_read.reshape(batch, length, kv_heads, groups, group_dim),
        g,
    )


def _dispatch_nonpositive_scale(function, arguments):
    """Run a finite nonpositive scale as a positive one, like the other attention APIs.

    The masked softmax kernels need a positive scale. Negating (scale < 0) or zeroing (scale = 0)
    both queries is exact for finite FP16/BF16 inputs; ``k_read=None`` becomes the aligned keys so
    that the keys keep their sign. Autograd sees the transforms; standalone backward applies the
    chain rule and folds the aligned keys' query gradient back into ``dk``.
    """
    options = dict(arguments)
    scale = float(options["scale"])
    q, k, k_read = options["q"], options["k"], options["k_read"]
    fold_k_read = k_read is None
    if fold_k_read and q.dim() == 4 and k.dim() == 4 and q.shape[1] <= k.shape[1]:
        k_read = _aligned_keys(k, q.shape[1])
    else:
        fold_k_read = False
    transform = (lambda t: -t) if scale < 0 else (lambda t: t * 0)
    options.update(
        q=transform(q), k_read=None if k_read is None else transform(k_read),
        scale=-scale if scale < 0 else 1.0,
    )
    result = function(**options)
    if not function.__name__.endswith("_bwd"):
        return result
    dq, dk, dv, dg, dk_read, *previous = result
    dq, dk_read = (-x if scale < 0 else torch.zeros_like(x) for x in (dq, dk_read))
    if fold_k_read:
        if scale < 0:
            dk = dk.float()
            dk[:, k.shape[1] - q.shape[1]:] += dk_read.float()
            dk = dk.to(q.dtype)
        dk_read = None
    return (dq, dk, dv, dg, dk_read, *previous)


def _prepare(api_name, q, k, v, g, k_read, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
             segment_idx, bos_mask, backend, deterministic, high_precision_level, standalone):
    """Validate a call: ``(q_segment_idx, k_segment_idx, resolved scale, SM90)``.

    The level, scale and SM90 widths are checked before any tensor shape.
    """
    validate_high_precision_level(high_precision_level, api_name=api_name)
    validate_attention_scale(scale)
    backend = _normalize_backend(backend)
    validate_sm90_dimensions(q, v, backend, api_name=api_name)
    prev_length = _validate_inputs(q, k, v, g, k_read, prev_k, prev_v, deterministic, api_name)
    q_segment_idx, k_segment_idx = _prepare_segments(
        q, prev_length + k.shape[1], q_segment_idx, k_segment_idx, segment_idx, bos_mask, api_name,
    )
    use_sm90 = _use_sm90_backend(backend, q, api_name)
    if not use_sm90 and (standalone or high_precision_level):
        raise ValueError(
            f"{api_name} requires the SM90 CUDA backend"
            + ("" if standalone else " for high_precision_level=1")
        )
    if use_sm90 and high_precision_level:
        from ._precision_backend import _ops as precision_ops
        precision_ops()
    return q_segment_idx, k_segment_idx, resolve_scale(q, scale), use_sm90


_TENSORS = ("q", "k", "v", "g", "k_read", "prev_k", "prev_v")
_OPTIONS = ("q_segment_idx", "k_segment_idx", "segment_idx", "bos_mask", "backend", "deterministic",
            "high_precision_level")


def _prepare_call(function, kind, arguments, standalone):
    """``_prepare`` over a public call's arguments, then its local support (validated after the
    level, scale, widths and shapes): ``(tensors, local,
    q_segment_idx, k_segment_idx, resolved scale, SM90)``. ``tensors`` holds ``v`` / ``g`` /
    ``prev_v`` in the internal value-group layout (``_grouped``); only the sliding chunk calls take
    previous K/V. ``local`` is None for causal calls and for windows / chunks covering every earlier key."""
    q, k, v, g, k_read, prev_k, prev_v = tuple(arguments.get(n) for n in _TENSORS)
    q_segment_idx, k_segment_idx, segment_idx, bos_mask, backend, deterministic, level = (
        arguments[n] for n in _OPTIONS
    )
    prepared = _prepare(
        function.__name__, q, k, v, g, k_read, arguments["scale"], prev_k, prev_v, q_segment_idx,
        k_segment_idx, segment_idx, bos_mask, backend, deterministic, level, standalone,
    )
    local = _local_support(function, kind, arguments)
    prev_length = 0 if prev_k is None else prev_k.shape[1]
    if local is not None and local.visibility == 2 and prev_k is not None and prev_length != local.span:
        raise ValueError(f"{function.__name__} prev_k/prev_v sequence length must equal chunk_size")
    v, g, prev_v = _grouped(v, g, prev_v)
    tensors = (q, k, v, g, k_read, prev_k, prev_v)
    return (tensors, _effective_local(local, k.shape[1] + prev_length), *prepared)


def _call(function, kind, arguments):
    """The autograd call of ``function`` over ``kind`` support."""
    if arguments["scale"] is not None and float(arguments["scale"]) <= 0:
        return _dispatch_nonpositive_scale(function, arguments)
    tensors, local, q_segment_idx, k_segment_idx, resolved_scale, use_sm90 = _prepare_call(
        function, kind, arguments, standalone=False,
    )
    q, k, v, g, k_read, prev_k, prev_v = tensors
    deterministic, level = arguments["deterministic"], arguments["high_precision_level"]
    if not use_sm90:
        return _torch_softdelta(
            q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx, resolved_scale, local,
        ).flatten(3)
    needs_backward = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)
    varlen_plan, q_segment_idx, k_segment_idx = _varlen_route(
        q, k, v, g, k_read, prev_k, arguments["bos_mask"], q_segment_idx, k_segment_idx, deterministic,
        level, needs_backward, local,
    )
    return _SoftDeltaFunc.apply(
        q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx,
        resolved_scale, deterministic, level, varlen_plan, local,
    ).flatten(3)


def _state_rows(t: Tensor) -> Tensor:
    """``t`` ``[B, L, X, V]`` as is when its heads are packed rows over merged batch and length (head
    slices of the forward's buffer qualify), otherwise contiguous."""
    batch, length, heads, width = t.shape
    if t.stride(-1) == 1 and (heads == 1 or t.stride(2) == width) and (
            batch == 1 or length == 1 or t.stride(0) == length * t.stride(1)):
        return t
    return t.contiguous()


def _call_fwd(function, kind, arguments):
    """The standalone forward of ``function``: ``(y, neg_read, lse)``."""
    if arguments["scale"] is not None and float(arguments["scale"]) <= 0:
        return _dispatch_nonpositive_scale(function, arguments)
    tensors, local, q_segment_idx, k_segment_idx, resolved_scale, _ = _prepare_call(
        function, kind, arguments, standalone=True,
    )
    q, k, v, g, k_read, prev_k, prev_v = tensors
    deterministic, level = arguments["deterministic"], arguments["high_precision_level"]
    varlen_plan, q_segment_idx, k_segment_idx = _varlen_route(
        q, k, v, g, k_read, prev_k, arguments["bos_mask"], q_segment_idx, k_segment_idx, deterministic,
        level, True, local,
    )
    y, neg_read, lse = _forward(
        q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx, resolved_scale,
        level, varlen_plan, local,
    )
    return y.flatten(3), neg_read.flatten(3), lse


def _call_bwd(function, kind, arguments):
    """The standalone backward of ``function``: ``(dq, dk, dv, dg, dk_read)``, then ``(dprev_k, dprev_v)``
    when it takes previous K/V."""
    if arguments["scale"] is not None and float(arguments["scale"]) <= 0:
        return _dispatch_nonpositive_scale(function, arguments)
    tensors, local, q_segment_idx, k_segment_idx, resolved_scale, _ = _prepare_call(
        function, kind, arguments, standalone=True,
    )
    q, k, v, g, k_read, prev_k, prev_v = tensors
    y_grad, y, neg_read, lse = (arguments[n] for n in ("y_grad", "y", "neg_read", "lse"))
    deterministic, level = arguments["deterministic"], arguments["high_precision_level"]
    batch, length, q_heads, _ = q.shape
    kv_heads, value_dim = k.shape[2], arguments["v"].shape[-1]
    for name, tensor, shape, dtype in (
        ("y_grad", y_grad, (batch, length, q_heads, value_dim), q.dtype),
        ("y", y, (batch, length, q_heads, value_dim), q.dtype),
        ("neg_read", neg_read, (batch, length, kv_heads, value_dim), torch.float32 if level else q.dtype),
        ("lse", lse, (batch, kv_heads + q_heads, length), torch.float32),
    ):
        if tuple(tensor.shape) != shape or tensor.device != q.device or tensor.dtype != dtype:
            raise ValueError(f"{function.__name__} {name} has incompatible shape, device, or dtype")
    dq, dk, dv, dg, dk_read, dprev_k, dprev_v = _backward(
        y_grad.unsqueeze(3), q, k, v, g, k_read, prev_k, prev_v, q_segment_idx, k_segment_idx,
        _state_rows(y).unsqueeze(3), _state_rows(neg_read).unsqueeze(3), lse.contiguous(),
        resolved_scale, deterministic, level, local,
    )
    grads = (dq, dk, dv.flatten(3), dg.reshape(g.shape[:3]), dk_read)
    if "prev_k" not in arguments:
        return grads
    return (*grads, dprev_k, None if dprev_v is None else dprev_v.flatten(3))


def softdelta_attention(
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
    v: Tensor,
    g: Tensor,
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
    """Run causal SoftDelta attention with autograd.

    ``q`` is ``[B, Lq, H, D]``, ``k`` ``[B, Lkv, Hkv, D]``, ``v``
    ``[B, Lkv, Hkv, V]`` and ``g`` ``[B, Lq, H]`` the gate logits (before the
    sigmoid; one scalar gate per query head and token), with ``Lq <= Lkv`` and
    ``H % Hkv == 0``. Query ``i`` aligns to key ``i + Lkv - Lq``.

    The output ``y`` ``[B, Lq, H, V]`` is ``read - sigmoid(g) * neg_read``:
    the query read ``read`` covers every visible key up to and including the
    aligned position with ``q``; the negative read ``neg_read`` covers only
    strictly earlier visible keys, once per KV head, with the negative query
    ``k_read`` ``[B, Lq, Hkv, D]``: any post-RoPE tensor, such as the keys, a
    transformed key or a second query. ``k_read=None`` uses the aligned keys, and
    their query-side gradient is added to ``k``'s gradient. ``neg_read`` is zero
    on rows without an earlier visible key.

    Segment metadata (``segment_idx``, ``q_segment_idx``/``k_segment_idx`` or
    ``bos_mask``) isolates tokens without resetting positions. ``scale`` is the
    softmax scale of both reads (``softmax(scale * q k^T)``); ``None`` uses
    ``1/sqrt(D)``. ``high_precision_level=1`` runs the reads' softmax in high
    precision and the precision-repair backward (SM90).
    """
    validate_attention_scale(scale)
    return _call(softdelta_attention, AttentionVisibility.CAUSAL_FULL, locals())


@torch.no_grad()
def softdelta_attention_fwd(
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
    v: Tensor,
    g: Tensor,
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
    """Return ``(y, neg_read, lse)`` without an autograd graph (SM90).

    ``y`` ``[B, Lq, H, V]`` keeps the input dtype. ``neg_read``
    ``[B, Lq, Hkv, V]`` holds the negative reads, in the input dtype at level 0
    and in FP32 at level 1. ``lse`` ``[B, Hkv + H, Lq]`` (FP32) holds the
    negative reads' log-sum-exp, then the query reads'. At level 0 ``y`` and
    ``neg_read`` may be head slices of one buffer. Pass all three, with the
    forward's other arguments, to ``softdelta_attention_bwd``.
    """
    validate_attention_scale(scale)
    return _call_fwd(softdelta_attention_fwd, AttentionVisibility.CAUSAL_FULL, locals())


@torch.no_grad()
def softdelta_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
    v: Tensor,
    g: Tensor,
    y: Tensor,
    neg_read: Tensor,
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Return ``(dq, dk, dv, dg, dk_read)`` (SM90).

    ``y``, ``neg_read`` and ``lse`` come from ``softdelta_attention_fwd`` with
    the same arguments; ``y_grad`` keeps the input dtype. Level 1 recomputes
    the softmax statistics and reads only ``neg_read``. ``dk_read`` is None when
    ``k_read`` is (the aligned keys' gradient is included in ``dk``).
    """
    validate_attention_scale(scale)
    return _call_bwd(softdelta_attention_bwd, AttentionVisibility.CAUSAL_FULL, locals())


def sliding_window_softdelta_attention(
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
    v: Tensor,
    g: Tensor,
    window_size: int,
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
    """Run SoftDelta within a causal sliding window, with autograd.

    ``softdelta_attention`` with both reads limited to keys at most
    ``window_size`` positions before the aligned query position (an inclusive
    left radius): the query read sees ``[i - window_size, i]``, the negative
    read ``[i - window_size, i - 1]``.
    """
    validate_attention_scale(scale)
    return _call(sliding_window_softdelta_attention, AttentionVisibility.SLIDING_WINDOW, locals())


@torch.no_grad()
def sliding_window_softdelta_attention_fwd(
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
    v: Tensor,
    g: Tensor,
    window_size: int,
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
    """``softdelta_attention_fwd`` within a sliding window: ``(y, neg_read, lse)``."""
    validate_attention_scale(scale)
    return _call_fwd(sliding_window_softdelta_attention_fwd, AttentionVisibility.SLIDING_WINDOW, locals())


@torch.no_grad()
def sliding_window_softdelta_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
    v: Tensor,
    g: Tensor,
    y: Tensor,
    neg_read: Tensor,
    lse: Tensor,
    window_size: int,
    scale: Optional[float] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    deterministic: bool = False,
    high_precision_level: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """``softdelta_attention_bwd`` within a sliding window: ``(dq, dk, dv, dg, dk_read)``."""
    validate_attention_scale(scale)
    return _call_bwd(sliding_window_softdelta_attention_bwd, AttentionVisibility.SLIDING_WINDOW, locals())


def sliding_chunk_softdelta_attention(
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
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
    high_precision_level: int = 0,
) -> Tensor:
    """Run SoftDelta over the previous and current logical chunks, with autograd.

    ``softdelta_attention`` with both reads limited to the keys of the query's
    chunk and the chunk before it; the negative read keeps only strictly earlier
    keys. Previous K/V, when given, holds exactly ``chunk_size`` tokens.
    ``reset_chunk_pos_per_seq`` restarts chunk positions in every segment.
    """
    validate_attention_scale(scale)
    return _call(sliding_chunk_softdelta_attention, AttentionVisibility.SLIDING_CHUNK, locals())


@torch.no_grad()
def sliding_chunk_softdelta_attention_fwd(
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
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
    high_precision_level: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """``softdelta_attention_fwd`` over sliding chunks: ``(y, neg_read, lse)``."""
    validate_attention_scale(scale)
    return _call_fwd(sliding_chunk_softdelta_attention_fwd, AttentionVisibility.SLIDING_CHUNK, locals())


@torch.no_grad()
def sliding_chunk_softdelta_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    k_read: Optional[Tensor],
    v: Tensor,
    g: Tensor,
    y: Tensor,
    neg_read: Tensor,
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
    high_precision_level: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """``softdelta_attention_bwd`` over sliding chunks:
    ``(dq, dk, dv, dg, dk_read, dprev_k, dprev_v)``."""
    validate_attention_scale(scale)
    return _call_bwd(sliding_chunk_softdelta_attention_bwd, AttentionVisibility.SLIDING_CHUNK, locals())


_VARLEN_MIN_TOKENS = 65536
# Fused backward launch layout: -1 lets the extension choose (per logical head, or
# reader groups for MHA / groups of two with enough work); 0 / 1 force a layout.
_BWD_READER_GROUPS = -1


def _varlen_route(q, k, v, g, k_read, prev_k, bos_mask, q_segment_idx, k_segment_idx, deterministic,
                  high_precision_level, needs_backward, local=None):
    """``(varlen plan or None, q_segment_idx, k_segment_idx)`` for the fused causal forward.

    Packed varlen forward from 64K tokens; below, the plan's host synchronization and extra launches
    cost more than the segment forward loses. Local support reads BOS masks as segment ids.
    """
    if (local is not None or bos_mask is None or prev_k is not None or q.shape[1] != k.shape[1]
            or q.shape[0] * q.shape[1] < _VARLEN_MIN_TOKENS or _fused_ops(q, v, prev_k) is None):
        return None, q_segment_idx, k_segment_idx
    varlen_plan, dense = _bos_varlen_plan(
        q, k, v, bos_mask, deterministic, high_precision_level, needs_backward,
    )
    if dense:
        return varlen_plan, None, None
    return varlen_plan, q_segment_idx, k_segment_idx


def _bos_varlen_plan(q, k, v, bos_mask, deterministic, high_precision_level, needs_backward):
    """(varlen plan or None, dense) for a BOS mask over equal-length rows: one sequence per row reads
    densely; otherwise the BOS router's forward choice selects the packed varlen forward (level 0)."""
    cu_seqlens, max_seqlen, num_runs = _bos_plan(_extension_ops(), bos_mask)
    if num_runs == q.shape[0]:
        return None, True
    if high_precision_level:
        return None, False
    fwd_varlen, _ = select_flash_swa_bos_routes(
        q, k, v.flatten(3), k.shape[1] - 1, num_runs, max_seqlen, deterministic, needs_backward,
        full_visibility=True,
    )
    return ((cu_seqlens, max_seqlen) if fwd_varlen else None), False

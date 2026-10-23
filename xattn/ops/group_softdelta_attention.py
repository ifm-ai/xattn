# Author: Shicheng Wen

"""Group SoftDelta attention: a SoftDelta negative read shared by each KV group.

Per query head ``h`` with KV head ``kv(h)``::

    read_h      = Attention(q_h,      K, V; inclusive causal support)
    neg_read_kv = Attention(neg_q_kv, K, V; strict-past causal support)
    y_h         = read_h - sigmoid(g_h) * neg_read_kv(h)

The negative read runs once per KV head and is shared by the query heads of its
group. Its query ``neg_q`` is whatever post-RoPE ``[B, Lq, Hkv, D]`` tensor the
caller passes (the keys, a transformed key, a second query); by default the
aligned keys themselves, whose query-side gradient then folds into ``k``'s
gradient. ``readers`` ``[B, Lq, Hkv + H, C, Dg]`` stacks the negative reads,
then the query reads.
"""

import functools
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
from ._backward_precision import (
    allocate_backward_output_state,
    validate_backward_precision,
    validate_high_precision_level,
)
from ._bos_metadata import _bos_plan
from ._bos_routing_heuristics import select_flash_swa_bos_routes
from ._causal_attention import _dense_bwd, _dense_fwd, _extension_ops
from ._segment_metadata import resolve_segment_indices, validate_segment_indices
from .softdelta_attention import _normalize_backend

__all__ = [
    "group_softdelta_attention",
    "group_softdelta_attention_fwd",
    "group_softdelta_attention_bwd",
]

_API = "group_softdelta_attention"


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    neg_q: Optional[Tensor],
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    deterministic: bool,
    api_name: str = _API,
) -> Tuple[int, int, int]:
    """Return ``(groups, group_dim, prev_length)`` after shape validation."""
    if not isinstance(deterministic, bool):
        raise TypeError(f"{api_name} deterministic must be a bool")
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(f"{api_name} expects 4D q/k tensors")
    if v.dim() != 5 or g.dim() != 5:
        raise ValueError(f"{api_name} expects 5D v/g tensors")
    tensors = (k, v, g) if neg_q is None else (k, v, g, neg_q)
    device, dtype = q.device, q.dtype
    if any(t.device != device for t in tensors):
        raise ValueError(f"{api_name} q/k/v/g/neg_q must share a device")
    if any(t.dtype != dtype for t in tensors):
        raise ValueError(f"{api_name} q/k/v/g/neg_q must share a dtype")
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
    if neg_q is not None and tuple(neg_q.shape) != (batch, q_length, kv_heads, qk_dim):
        raise ValueError(
            f"{api_name} neg_q must have shape {[batch, q_length, kv_heads, qk_dim]}"
        )

    groups, group_dim = v.shape[3:]
    if groups <= 0 or group_dim <= 0:
        raise ValueError(f"{api_name} value group dimensions must be positive")
    expected_gate = (batch, q_length, q_heads, groups, 1)
    if tuple(g.shape) != expected_gate:
        raise ValueError(f"{api_name} g must have shape {list(expected_gate)}")

    if (prev_k is None) != (prev_v is None):
        raise ValueError(f"{api_name} prev_k and prev_v must be provided together")
    if prev_k is None:
        return groups, group_dim, 0
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
    if tuple(prev_v.shape) != (batch, prev_length, kv_heads, groups, group_dim):
        raise ValueError(f"{api_name} prev_v shape is incompatible with v")
    return groups, group_dim, prev_length


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


def _use_sm90_backend(backend: str, q: Tensor, api_name: str = _API) -> bool:
    """SoftDelta's backend rule with the device capability cached per device."""
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


def _fused_ops(q: Tensor, v: Tensor, prev_k: Optional[Tensor]):
    """Extension ops with the fused reads for this call, or None (composition).

    The fused SM90 instances cover every D, V <= 256 (multiples of 8, rounded up to
    the causal_flash_attn buckets; csrc/group_softdelta/instantiations) with gate
    groups of a multiple of 8 channels.
    """
    dim, width = q.shape[-1], v.shape[3] * v.shape[4]
    if (
        prev_k is not None
        or not (0 < dim <= 256 and 0 < width <= 256)
        or dim % 8 != 0
        or v.shape[4] % 8 != 0
    ):
        return None
    ops = _extension_ops()
    return ops if hasattr(ops, "_group_softdelta_fwd") else None


def _split_readers(readers: Tensor, lse: Tensor, q_heads: int):
    """Contiguous (read, read_lse, neg_read, neg_read_lse) of the class-major readers:
    heads [0, Hkv) the negative reads, [Hkv, Hkv + H) the query reads."""
    kv_heads = readers.shape[2] - q_heads
    return (
        readers[:, :, kv_heads:].contiguous(),
        lse[:, kv_heads:].contiguous(),
        readers[:, :, :kv_heads].contiguous(),
        lse[:, :kv_heads].contiguous(),
    )


def _add_optional(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
    if a is None:
        return b
    if b is None:
        return a
    return (a.float() + b.float()).to(a.dtype)


def _forward(q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx, scale, level, varlen_plan):
    """Both reads and the gate: ``(y, readers, lse)``.

    ``readers`` ``[B, Lq, Hkv + H, C * Dg]`` holds the KV heads' negative reads, then the query heads'
    reads (FP32 state for level >= 1); ``lse`` ``[B, Hkv + H, Lq]`` matches. The fused SM90 launch
    writes this layout directly; the composition (previous K/V or other gate groups) concatenates it.
    """
    batch, length, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    groups, group_dim = v.shape[3:]
    flat_v = v.flatten(3)
    if neg_q is None:
        neg_q = _aligned_keys(k, length)
    fused = _fused_ops(q, v, prev_k)
    if fused is not None:
        q, k, flat_v, neg_q = (_tma_readable(t) for t in (q, k, flat_v, neg_q))
        state = None
        if level:
            state = torch.empty(
                batch, length, kv_heads + q_heads, groups * group_dim,
                dtype=torch.float32, device=q.device,
            )
        if varlen_plan is not None and state is None:
            # Packed sequences: both reads as varlen launches into the readers.
            readers, lse = fused._group_softdelta_fwd_varlen(
                q, k, flat_v, neg_q, varlen_plan[0], varlen_plan[1], scale,
            )
        else:
            readers, lse = fused._group_softdelta_fwd(
                q, k, flat_v, neg_q, scale, q_segment_idx, k_segment_idx, state,
            )
        readers = readers if state is None else state
        y = fused._group_softdelta_gate_fwd_readers(
            readers.view(batch, length, kv_heads + q_heads, groups, group_dim), g.contiguous(),
        )
        return y, readers, lse
    flat_prev_v = None if prev_v is None else prev_v.flatten(3)
    read_state = neg_read_state = None
    if level:
        read_state = allocate_backward_output_state(q, flat_v)
        neg_read_state = allocate_backward_output_state(neg_q, flat_v)
    common = (scale, prev_k, flat_prev_v, q_segment_idx, k_segment_idx, "sm90")
    read, read_lse = _dense_fwd(q, k, flat_v, None, *common, output_state=read_state)
    neg_read, neg_read_lse = _dense_fwd(
        neg_q, k, flat_v, None, *common, strict_past=True, output_state=neg_read_state,
    )
    read = read if read_state is None else read_state
    neg_read = neg_read if neg_read_state is None else neg_read_state
    y = _extension_ops()._group_softdelta_gate_fwd(
        read.view(batch, length, q_heads, groups, group_dim),
        neg_read.view(batch, length, kv_heads, groups, group_dim),
        g.contiguous(),
    )
    return y, torch.cat((neg_read, read), dim=2), torch.cat((neg_read_lse, read_lse), dim=1)


def _backward(y_grad, q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx,
              readers, lse, scale, deterministic, level):
    """``(dq, dk, dv, dg, dneg_q, dprev_k, dprev_v)`` from the forward's readers and LSE.

    ``dneg_q`` is None when ``neg_q`` is (the aligned keys' query-side gradient is in ``dk``).
    Level 0/1 nondeterministic calls with equal Q / K lengths and no previous K/V take the fused
    launch; the others run the gate backward, then each read's backward (levels 2-4: the
    advanced precision backward), summed in FP32.
    """
    batch, length, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    groups, group_dim = v.shape[3:]
    flat_v = v.flatten(3)
    flat_prev_v = None if prev_v is None else prev_v.flatten(3)
    fold_neg_q = neg_q is None
    if fold_neg_q:
        neg_q = _aligned_keys(k, length)
    y_grad, g = y_grad.contiguous(), g.contiguous()
    fused = _fused_ops(q, v, prev_k)
    if level < 2 and not deterministic and fused is not None and length == k.shape[1]:
        q, k, flat_v, neg_q = (_tma_readable(t) for t in (q, k, flat_v, neg_q))
        dq, dk, dv, dneg_q, dg = fused._group_softdelta_bwd(
            y_grad, g, q, k, flat_v, neg_q, readers, lse, scale, q_segment_idx, k_segment_idx,
            fold_neg_q, _BWD_READER_GROUPS,
        )
        return dq, dk, dv.view(v.shape), dg, None if fold_neg_q else dneg_q, None, None
    read, read_lse, neg_read, neg_read_lse = _split_readers(readers, lse, q_heads)
    ops = _extension_ops()
    neg_read_grad, dg = ops._group_softdelta_gate_bwd(
        y_grad,
        neg_read.view(batch, length, kv_heads, groups, group_dim),
        g,
    )
    neg_read_grad = neg_read_grad.view(batch, length, kv_heads, groups * group_dim)
    read_grad = y_grad.view(batch, length, q_heads, groups * group_dim)
    if level >= 2:
        from ._precision_backend import backward as precision_backward
        common = dict(
            level=level, visibility=0, scale=scale, prev_k=prev_k, prev_v=flat_prev_v,
            q_segment_idx=q_segment_idx, k_segment_idx=k_segment_idx,
            deterministic=deterministic, return_fp32=True,
        )
        dq, dk, dv, dprev_k, dprev_v = precision_backward(read_grad, q, k, flat_v, **common)
        dneg_q, dk_neg, dv_neg, dprev_k_neg, dprev_v_neg = precision_backward(
            neg_read_grad, neg_q, k, flat_v, strict=True, **common,
        )
        dk += dk_neg
        dv += dv_neg
        if fold_neg_q:
            dk[:, k.shape[1] - length:] += dneg_q
            dneg_q = None
        dprev_k = _add_optional(dprev_k, dprev_k_neg)
        dprev_v = _add_optional(dprev_v, dprev_v_neg)
        dq, dk, dv, dneg_q, dprev_k, dprev_v = (
            None if x is None else x.to(q.dtype) for x in (dq, dk, dv, dneg_q, dprev_k, dprev_v)
        )
    else:
        common = (scale, prev_k, flat_prev_v, q_segment_idx, k_segment_idx, deterministic, "sm90")
        dq, dk, dv, dprev_k, dprev_v = _dense_bwd(read_grad, q, k, flat_v, read, read_lse, None, *common)
        dneg_q, dk_neg, dv_neg, dprev_k_neg, dprev_v_neg = _dense_bwd(
            neg_read_grad, neg_q, k, flat_v, neg_read, neg_read_lse, None,
            *common, strict_past=True,
        )
        dk, dv = ops._group_softdelta_merge_gradients(
            dk.contiguous(), dk_neg.contiguous(), dneg_q.contiguous() if fold_neg_q else None,
            dv.contiguous(), dv_neg.contiguous(),
        )
        if fold_neg_q:
            dneg_q = None
        if dprev_k is not None and dprev_k_neg is not None:
            dprev_k, dprev_v = ops._group_softdelta_merge_gradients(
                dprev_k.contiguous(), dprev_k_neg.contiguous(), None,
                dprev_v.contiguous(), dprev_v_neg.contiguous(),
            )
        else:
            dprev_k = _add_optional(dprev_k, dprev_k_neg)
            dprev_v = _add_optional(dprev_v, dprev_v_neg)
    dv = dv.view(v.shape)
    if dprev_v is not None:
        dprev_v = dprev_v.view(prev_v.shape)
    return dq, dk, dv, dg, dneg_q, dprev_k, dprev_v


class _GroupSoftDeltaFunc(torch.autograd.Function):
    """SM90 reads (fused launch or composition), gate, and their backward."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        neg_q: Optional[Tensor],
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        scale: float,
        deterministic: bool,
        high_precision_level: int,
        varlen_plan: Optional[Tuple[Tensor, int]] = None,
    ) -> Tensor:
        y, readers, lse = _forward(
            q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx, scale,
            high_precision_level, varlen_plan,
        )
        ctx.save_for_backward(q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx, readers, lse)
        ctx.scale = scale
        ctx.deterministic = deterministic
        ctx.high_precision_level = high_precision_level
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y_grad: Tensor) -> tuple:
        q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx, readers, lse = ctx.saved_tensors
        gradients = _backward(
            y_grad, q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx,
            readers, lse, ctx.scale, ctx.deterministic, ctx.high_precision_level,
        )
        return (*gradients, None, None, None, None, None, None)


def _torch_group_softdelta(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    neg_q: Optional[Tensor],
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    scale: float,
) -> Tensor:
    """Explicit correctness plan: materialized masks, FP32 reads and gate."""
    length = q.shape[1]
    if neg_q is None:
        neg_q = _aligned_keys(k, length)
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
            ),
            q_segment_idx=q_segment_idx,
            k_segment_idx=k_segment_idx,
        )
    read = attention_read(q, k, flat_v, masks[AttentionBoundary.INCLUSIVE], scale)
    neg_read = attention_read(
        neg_q, k, flat_v, masks[AttentionBoundary.STRICT_PAST], scale
    )
    return _gate_combine(
        read.reshape(batch, length, q_heads, groups, group_dim),
        neg_read.reshape(batch, length, kv_heads, groups, group_dim),
        g,
    )


def _dispatch_nonpositive_scale(function, arguments):
    """Run a finite nonpositive scale as a positive one, like the other attention APIs.

    The masked softmax kernels need a positive scale. Negating (scale < 0) or zeroing (scale = 0)
    both queries is exact for finite FP16/BF16 inputs; ``neg_q=None`` becomes the aligned keys so
    that the keys keep their sign. Autograd sees the transforms; standalone backward applies the
    chain rule and folds the aligned keys' query gradient back into ``dk``.
    """
    options = dict(arguments)
    scale = float(options["scale"])
    q, k, neg_q = options["q"], options["k"], options["neg_q"]
    fold_neg_q = neg_q is None
    if fold_neg_q and q.dim() == 4 and k.dim() == 4 and q.shape[1] <= k.shape[1]:
        neg_q = _aligned_keys(k, q.shape[1])
    else:
        fold_neg_q = False
    transform = (lambda t: -t) if scale < 0 else (lambda t: t * 0)
    options.update(
        q=transform(q), neg_q=None if neg_q is None else transform(neg_q),
        scale=-scale if scale < 0 else 1.0,
    )
    result = function(**options)
    if not function.__name__.endswith("_bwd"):
        return result
    dq, dk, dv, dg, dneg_q, dprev_k, dprev_v = result
    dq, dneg_q = (-x if scale < 0 else torch.zeros_like(x) for x in (dq, dneg_q))
    if fold_neg_q:
        if scale < 0:
            dk = dk.float()
            dk[:, k.shape[1] - q.shape[1]:] += dneg_q.float()
            dk = dk.to(q.dtype)
        dneg_q = None
    return dq, dk, dv, dg, dneg_q, dprev_k, dprev_v


def _prepare(api_name, q, k, v, g, neg_q, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
             segment_idx, bos_mask, backend, deterministic, high_precision_level, standalone):
    """Validate a call: ``(q_segment_idx, k_segment_idx, resolved scale, SM90)``.

    The level, scale and SM90 widths are checked before any tensor shape.
    """
    validate_high_precision_level(high_precision_level, api_name=api_name)
    validate_attention_scale(scale)
    backend = _normalize_backend(backend)
    validate_sm90_dimensions(q, v, backend, api_name=api_name)
    _, _, prev_length = _validate_inputs(q, k, v, g, neg_q, prev_k, prev_v, deterministic, api_name)
    q_segment_idx, k_segment_idx = _prepare_segments(
        q, prev_length + k.shape[1], q_segment_idx, k_segment_idx, segment_idx, bos_mask, api_name,
    )
    use_sm90 = _use_sm90_backend(backend, q, api_name)
    if not use_sm90 and (standalone or high_precision_level >= 2):
        raise ValueError(
            f"{api_name} requires the SM90 CUDA backend"
            + ("" if standalone else " for high_precision_level >= 2")
        )
    if use_sm90 and high_precision_level >= 2:
        from ._precision_backend import _ops as precision_ops
        precision_ops()
    return q_segment_idx, k_segment_idx, resolve_scale(q, scale), use_sm90


def group_softdelta_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    neg_q: Optional[Tensor] = None,
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
    """Run causal group SoftDelta attention with autograd.

    ``q`` is ``[B, Lq, H, D]``, ``k`` ``[B, Lkv, Hkv, D]``, ``v``
    ``[B, Lkv, Hkv, C, Dg]`` and the gate logits ``g`` ``[B, Lq, H, C, 1]``,
    with ``Lq <= Lkv`` and ``H % Hkv == 0``. Query ``i`` aligns to key
    ``i + P + Lkv - Lq``, where ``P`` is the optional previous KV length.

    The output ``y`` ``[B, Lq, H, C, Dg]`` is ``read - sigmoid(g) * neg_read``:
    the query read ``read`` covers every visible key up to and including the
    aligned position with ``q``; the negative read ``neg_read`` covers only
    strictly earlier visible keys, once per KV head, with the negative query
    ``neg_q`` ``[B, Lq, Hkv, D]``: any post-RoPE tensor, such as the keys, a
    transformed key or a second query. ``neg_q=None`` uses the aligned keys, and
    their query-side gradient is added to ``k``'s gradient. ``neg_read`` is zero
    on rows without an earlier visible key.

    Segment metadata (``segment_idx``, ``q_segment_idx``/``k_segment_idx`` or
    ``bos_mask``) isolates tokens without resetting positions. ``scale`` is the
    softmax scale of both reads (``softmax(scale * q k^T)``); ``None`` uses
    ``1/sqrt(D)``. ``high_precision_level`` 1 keeps FP32 reader state for
    backward; 2-4 add the advanced precision backward (SM90).
    """
    validate_attention_scale(scale)
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(group_softdelta_attention, locals())
    q_segment_idx, k_segment_idx, resolved_scale, use_sm90 = _prepare(
        _API, q, k, v, g, neg_q, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
        segment_idx, bos_mask, backend, deterministic, high_precision_level, standalone=False,
    )
    if not use_sm90:
        return _torch_group_softdelta(
            q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx,
            resolved_scale,
        )
    needs_backward = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, g, neg_q, prev_k, prev_v)
    )
    varlen_plan, q_segment_idx, k_segment_idx = _varlen_route(
        q, k, v, g, neg_q, prev_k, bos_mask, q_segment_idx, k_segment_idx, deterministic,
        high_precision_level, needs_backward,
    )
    return _GroupSoftDeltaFunc.apply(
        q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx,
        resolved_scale, deterministic, high_precision_level, varlen_plan,
    )


@torch.no_grad()
def group_softdelta_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    neg_q: Optional[Tensor] = None,
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
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(y, readers, lse)`` without an autograd graph (SM90).

    ``readers`` ``[B, Lq, Hkv + H, C, Dg]`` holds the KV heads' negative reads,
    then the query heads' reads: FP32 when ``high_precision_level >= 1``,
    otherwise the input dtype, including under no_grad/inference mode. ``lse``
    ``[B, Hkv + H, Lq]`` (FP32) uses the same head order. Pass both, with the
    forward's other arguments, to ``group_softdelta_attention_bwd``.
    """
    validate_attention_scale(scale)
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(group_softdelta_attention_fwd, locals())
    api = "group_softdelta_attention_fwd"
    q_segment_idx, k_segment_idx, resolved_scale, _ = _prepare(
        api, q, k, v, g, neg_q, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
        segment_idx, bos_mask, backend, deterministic, high_precision_level, standalone=True,
    )
    varlen_plan, q_segment_idx, k_segment_idx = _varlen_route(
        q, k, v, g, neg_q, prev_k, bos_mask, q_segment_idx, k_segment_idx, deterministic,
        high_precision_level, True,
    )
    y, readers, lse = _forward(
        q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx, resolved_scale,
        high_precision_level, varlen_plan,
    )
    batch, length, q_heads, _ = q.shape
    return y, readers.view(batch, length, k.shape[2] + q_heads, *v.shape[3:]), lse


@torch.no_grad()
def group_softdelta_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    readers: Tensor,
    lse: Tensor,
    neg_q: Optional[Tensor] = None,
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Return ``(dq, dk, dv, dg, dneg_q, dprev_k, dprev_v)`` (SM90).

    ``readers`` and ``lse`` come from ``group_softdelta_attention_fwd`` with the
    same arguments. ``high_precision_level`` selects the backward algorithm, and
    ``readers`` must have the matching state dtype; ``y_grad`` keeps the input
    dtype. ``dneg_q`` is None when ``neg_q`` is (the aligned keys' gradient is
    included in ``dk``); absent previous gradients are None.
    """
    validate_attention_scale(scale)
    if scale is not None and float(scale) <= 0:
        return _dispatch_nonpositive_scale(group_softdelta_attention_bwd, locals())
    api = "group_softdelta_attention_bwd"
    q_segment_idx, k_segment_idx, resolved_scale, _ = _prepare(
        api, q, k, v, g, neg_q, scale, prev_k, prev_v, q_segment_idx, k_segment_idx,
        segment_idx, bos_mask, backend, deterministic, high_precision_level, standalone=True,
    )
    validate_backward_precision(high_precision_level, readers, q)
    batch, length, q_heads, _ = q.shape
    reader_heads = k.shape[2] + q_heads
    for name, tensor, shape, dtypes in (
        ("y_grad", y_grad, (batch, length, q_heads, *v.shape[3:]), (q.dtype,)),
        ("readers", readers, (batch, length, reader_heads, *v.shape[3:]), (q.dtype, torch.float32)),
        ("lse", lse, (batch, reader_heads, length), (torch.float32,)),
    ):
        if tuple(tensor.shape) != shape or tensor.device != q.device or tensor.dtype not in dtypes:
            raise ValueError(f"{api} {name} has incompatible shape, device, or dtype")
    return _backward(
        y_grad, q, k, v, g, neg_q, prev_k, prev_v, q_segment_idx, k_segment_idx,
        readers.reshape(batch, length, reader_heads, -1).contiguous(), lse.contiguous(),
        resolved_scale, deterministic, high_precision_level,
    )


_VARLEN_MIN_TOKENS = 65536
# Fused backward launch layout: -1 lets the extension choose (per logical head, or
# reader groups for MHA / groups of two with enough work); 0 / 1 force a layout.
_BWD_READER_GROUPS = -1


def _varlen_route(q, k, v, g, neg_q, prev_k, bos_mask, q_segment_idx, k_segment_idx, deterministic,
                  high_precision_level, needs_backward):
    """``(varlen plan or None, q_segment_idx, k_segment_idx)`` for the fused forward.

    Packed varlen forward from 64K tokens; below, the plan's host sync and extra launches cost more
    than the segment forward loses (T4096 B8: segment 1.098x vs varlen 1.085x the Stage 3 FA3 path,
    fwd+bwd).
    """
    if (bos_mask is None or prev_k is not None or q.shape[1] != k.shape[1]
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

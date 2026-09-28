# Author: Shicheng Wen

"""Shared metadata, autograd, and dispatch for causal attention APIs."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ._backward_precision import (
    allocate_backward_output_state,
    use_high_precision_output,
)
from ._bos_metadata import (
    bos_to_segment_idx,
    build_bos_metadata,
    pack_batched_backward,
    pack_batched_qkv,
    requires_row_aligned_k_runs,
    restore_batched_lse,
    restore_batched_output,
    unpack_batched_grads,
    validate_bos_mask,
)
from ._bos_routing_heuristics import select_flash_swa_bos_routes
from ._segment_metadata import (
    resolve_segment_indices,
    validate_segment_indices,
)
from ._varlen_metadata import (
    validate_packed_qkv,
    validate_varlen_metadata,
)


def _extension_ops():
    try:
        import xattn_cuda
    except ImportError as exc:
        raise ImportError(
            "xattn CUDA extension is not installed. Install xattn with "
            "`pip install --no-build-isolation .` with a CUDA compiler "
            "available."
        ) from exc
    return xattn_cuda.ops


def _effective_window(
    k: Tensor, prev_k: Optional[Tensor], window_size: Optional[int]
) -> int:
    if window_size is not None:
        return window_size
    return k.shape[1] + (0 if prev_k is None else prev_k.shape[1]) - 1


def _dense_fwd(
    q, k, v, window_size, *args, strict_past=False, ops=None,
    use_fast_reciprocal=True, output_fp32=False, output_state=None,
):
    ops = _extension_ops() if ops is None else ops
    output_options = {"output_fp32": True} if output_fp32 else {}
    if output_state is not None:
        output_options["output_state"] = output_state
    if window_size is None:
        scale, prev_k, prev_v, *options = args
        if not strict_past and prev_k is None and prev_v is None:
            precision = {} if use_fast_reciprocal else {"use_fast_reciprocal": False}
            return ops.causal_flash_attn_fwd(
                q, k, v, scale, *options, **precision, **output_options
            )
        precision = {} if use_fast_reciprocal else {"use_fast_reciprocal": False}
        return ops._causal_flash_attn_component_fwd(
            q, k, v, *args, strict_past=strict_past, **precision, **output_options
        )
    operation = (
        ops._flash_swa_sm90_strict_past_fwd
        if strict_past
        else ops.flash_swa_fwd
    )
    return operation(q, k, v, window_size, *args, **output_options)


def _dense_bwd(
    y_grad, q, k, v, y, lse, window_size, *args, strict_past=False, ops=None
):
    ops = _extension_ops() if ops is None else ops
    if window_size is None:
        scale, prev_k, prev_v, *options = args
        if not strict_past and prev_k is None and prev_v is None:
            grads = ops.causal_flash_attn_bwd(
                y_grad, q, k, v, y, lse, scale, *options
            )
            return (*grads, None, None)
        return ops._causal_flash_attn_component_bwd(
            y_grad, q, k, v, y, lse, *args, strict_past=strict_past
        )
    operation = (
        ops._flash_swa_sm90_strict_past_bwd
        if strict_past
        else ops.flash_swa_bwd
    )
    return operation(y_grad, q, k, v, y, lse, window_size, *args)


def _resolve_scale(q: Tensor, scale: Optional[float]) -> float:
    return 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)


def _validate_window_size(
    window_size: int, api_name: str = "flash_swa"
) -> None:
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError(f"{api_name} window_size must be an integer")
    if window_size < 0:
        raise ValueError(f"{api_name} window_size must be nonnegative")


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    *,
    api_name: str = "flash_swa",
    allow_short_q: bool = False,
) -> int:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"{api_name} expects 4D [B, L, H, D] q/k/v")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"{api_name} q/k/v must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"{api_name} q/k/v must share a dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{api_name} supports only fp16 and bf16 inputs")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError(
            f"{api_name} batch dimensions and k/v lengths must match" if allow_short_q else
            f"{api_name} q/k/v batch and current sequence dimensions must match"
        )
    if q.shape[1] != k.shape[1] and not (allow_short_q and q.shape[1] <= k.shape[1]):
        raise ValueError(
            f"{api_name} requires q length <= k/v length" if allow_short_q else
            f"{api_name} q/k/v batch and current sequence dimensions must match"
        )
    if q.shape[1] <= 0:
        raise ValueError(
            f"{api_name} current sequence length must be positive"
        )
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"{api_name} k/v head counts must match")
    if q.shape[2] <= 0 or k.shape[2] <= 0:
        raise ValueError(f"{api_name} q and KV head counts must be positive")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(
            f"{api_name} q head count must be divisible by the KV head count"
        )
    if q.shape[-1] != k.shape[-1] or q.shape[-1] <= 0:
        raise ValueError(
            f"{api_name} q/k head dims must match and be positive"
        )
    if v.shape[-1] <= 0:
        raise ValueError(f"{api_name} value head dim must be positive")
    if (prev_k is None) != (prev_v is None):
        raise ValueError(
            f"{api_name} prev_k and prev_v must be provided together"
        )
    if prev_k is None:
        return 0
    assert prev_v is not None
    if prev_k.dim() != 4 or prev_v.dim() != 4:
        raise ValueError(f"{api_name} prev_k/prev_v must be 4D tensors")
    if prev_k.device != q.device or prev_v.device != q.device:
        raise ValueError(
            f"{api_name} previous and current K/V must share a device"
        )
    if prev_k.dtype != q.dtype or prev_v.dtype != q.dtype:
        raise ValueError(
            f"{api_name} previous and current K/V must share a dtype"
        )
    if (
        prev_k.shape[0] != q.shape[0]
        or prev_v.shape[0] != q.shape[0]
        or prev_k.shape[1] != prev_v.shape[1]
        or prev_k.shape[1] <= 0
        or prev_k.shape[2:] != k.shape[2:]
        or prev_v.shape[2:] != v.shape[2:]
    ):
        raise ValueError(
            f"{api_name} previous K/V shapes are incompatible with Q/K/V"
        )
    return prev_k.shape[1]


def _prepare_segment_indices(
    q: Tensor,
    k: Tensor,
    prev_length: int,
    segment_idx: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    *,
    api_name: str = "flash_swa",
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    q_segment_idx, k_segment_idx = resolve_segment_indices(
        segment_idx,
        q_segment_idx,
        k_segment_idx,
        batch_size=q.shape[0],
        q_length=q.shape[1],
        k_length=k.shape[1] + prev_length,
    )
    validate_segment_indices(
        q_segment_idx,
        k_segment_idx,
        batch_size=q.shape[0],
        q_length=q.shape[1],
        k_length=k.shape[1] + prev_length,
        device=q.device,
        api_name=api_name,
    )
    return q_segment_idx, k_segment_idx


def _prepare_bos_metadata(
    q: Tensor,
    k: Tensor,
    prev_length: int,
    bos_mask: Tensor,
    backend: str,
):
    if backend.strip().lower() not in ("", "auto", "sm90"):
        raise ValueError("BOS-mask FlashSWA requires backend='auto'/'sm90'")
    k_length = k.shape[1] + prev_length
    validate_bos_mask(bos_mask, q, k_length)
    if q.shape[1] > k_length:
        raise ValueError("bos_mask cannot right-align Q beyond its K layout")
    return build_bos_metadata(_extension_ops(), q, bos_mask, k_length)


def _check_bos_exclusive(
    bos_mask: Optional[Tensor],
    segment_idx: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
) -> None:
    if bos_mask is not None and (
        segment_idx is not None
        or q_segment_idx is not None
        or k_segment_idx is not None
    ):
        raise ValueError(
            "BOS masks and segment indices are mutually exclusive"
        )


def _bos_segment_indices(
    q: Tensor,
    bos_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    k_segment_idx = bos_to_segment_idx(_extension_ops(), bos_mask)
    return k_segment_idx[:, -q.shape[1] :], k_segment_idx


def _batched_varlen_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: Optional[int],
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    backend: str,
    strict_past: bool = False,
    output_state: Optional[Tensor] = None,
    output_fp32: bool = False,
) -> Tuple[Tensor, Tensor]:
    q_packed, k_packed, v_packed = pack_batched_qkv(q, k, v, prev_k, prev_v)
    args = (
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        _effective_window(k, prev_k, window_size),
        scale,
        backend,
        strict_past,
    )
    if output_state is not None:
        args = (*args, output_state.flatten(0, 1))
    options = {"output_fp32": True} if output_fp32 else {}
    y, lse = _extension_ops()._flash_swa_sm90_varlen_fwd(*args, **options)
    return restore_batched_output(y, q, v), restore_batched_lse(lse, q)


def _batched_varlen_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    window_size: Optional[int],
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    deterministic: bool,
    backend: str,
    strict_past: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    q_packed, k_packed, v_packed = pack_batched_qkv(q, k, v, prev_k, prev_v)
    dy_packed, y_packed, lse_packed = pack_batched_backward(y_grad, y, lse, q)
    dq, dk, dv = _extension_ops()._flash_swa_sm90_varlen_bwd(
        dy_packed,
        q_packed,
        k_packed,
        v_packed,
        y_packed,
        lse_packed,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        _effective_window(k, prev_k, window_size),
        scale,
        deterministic,
        backend,
        strict_past,
    )
    return unpack_batched_grads(dq, dk, dv, q, k, v, prev_k, prev_v)


class _CausalAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        window_size: Optional[int],
        scale: float,
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        backend: str,
        deterministic: bool,
        high_precision_output: bool,
    ) -> Tensor:
        output_state = (
            allocate_backward_output_state(q, v)
            if high_precision_output
            else None
        )
        args = (
            q,
            k,
            v,
            window_size,
            scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            backend,
        )
        if output_state is not None:
            args = (*args, output_state)
        y, lse = _dense_fwd(*args)
        ctx.save_for_backward(
            q,
            k,
            v,
            output_state if output_state is not None else y,
            lse,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
        )
        ctx.window_size = window_size
        ctx.scale = scale
        ctx.backend = backend
        ctx.deterministic = deterministic
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y_grad: Tensor) -> tuple:
        (
            q,
            k,
            v,
            y,
            lse,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
        ) = ctx.saved_tensors
        dq, dk, dv, dprev_k, dprev_v = _dense_bwd(
            y_grad,
            q,
            k,
            v,
            y,
            lse,
            ctx.window_size,
            ctx.scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            ctx.deterministic,
            ctx.backend,
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            dprev_k,
            dprev_v,
            None,
            None,
            None,
            None,
            None,
        )


class _CausalAttentionBosFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        window_size: Optional[int],
        scale: float,
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        backend: str,
        deterministic: bool,
        fwd_use_varlen: bool,
        bwd_use_varlen: bool,
        high_precision_output: bool,
    ) -> Tensor:
        output_state = (
            allocate_backward_output_state(q, v)
            if high_precision_output
            else None
        )
        if fwd_use_varlen:
            y, lse = _batched_varlen_fwd(
                q,
                k,
                v,
                window_size,
                scale,
                prev_k,
                prev_v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                backend,
                output_state=output_state,
            )
        else:
            args = (
                q,
                k,
                v,
                window_size,
                scale,
                prev_k,
                prev_v,
                q_segment_idx,
                k_segment_idx,
                backend,
            )
            if output_state is not None:
                args = (*args, output_state)
            y, lse = _dense_fwd(*args)
        ctx.save_for_backward(
            q,
            k,
            v,
            output_state if output_state is not None else y,
            lse,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        ctx.window_size = window_size
        ctx.scale = scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.backend = backend
        ctx.deterministic = deterministic
        ctx.bwd_use_varlen = bwd_use_varlen
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y_grad: Tensor) -> tuple:
        (
            q,
            k,
            v,
            y,
            lse,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = ctx.saved_tensors
        if ctx.bwd_use_varlen:
            grads = _batched_varlen_bwd(
                y_grad,
                q,
                k,
                v,
                y,
                lse,
                ctx.window_size,
                ctx.scale,
                prev_k,
                prev_v,
                cu_seqlens_q,
                cu_seqlens_k,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                ctx.deterministic,
                ctx.backend,
            )
        else:
            grads = _dense_bwd(
                y_grad,
                q,
                k,
                v,
                y,
                lse,
                ctx.window_size,
                ctx.scale,
                prev_k,
                prev_v,
                q_segment_idx,
                k_segment_idx,
                ctx.deterministic,
                ctx.backend,
            )
        dq, dk, dv, dprev_k, dprev_v = grads
        return (
            dq,
            dk,
            dv,
            None,
            None,
            dprev_k,
            dprev_v,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FlashSWAVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        window_size: Optional[int],
        scale: float,
        backend: str,
        deterministic: bool,
        high_precision_output: bool,
    ) -> Tensor:
        output_state = (
            allocate_backward_output_state(q, v)
            if high_precision_output
            else None
        )
        args = (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            window_size,
            scale,
            backend,
        )
        if output_state is not None:
            # Append extension arguments only when FP32 state is allocated.
            args = (*args, False, output_state)
        y, lse = _extension_ops()._flash_swa_sm90_varlen_fwd(*args)
        ctx.save_for_backward(
            q,
            k,
            v,
            output_state if output_state is not None else y,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.window_size = window_size
        ctx.scale = scale
        ctx.backend = backend
        ctx.deterministic = deterministic
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y_grad: Tensor) -> tuple:
        (
            q,
            k,
            v,
            y,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = ctx.saved_tensors
        dq, dk, dv = _extension_ops()._flash_swa_sm90_varlen_bwd(
            y_grad,
            q,
            k,
            v,
            y,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.window_size,
            ctx.scale,
            ctx.deterministic,
            ctx.backend,
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _validate_varlen_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: Optional[int],
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    api_name: str,
) -> None:
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError(f"{api_name} window_size must be an integer")
    if window_size < 0:
        raise ValueError(f"{api_name} window_size must be nonnegative")
    validate_packed_qkv(q, k, v, api_name=api_name)
    validate_varlen_metadata(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        api_name=api_name,
    )


def _causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: Optional[int],
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
    api_name: str = "flash_swa",
) -> Tensor:
    """Shared causal full/window attention dispatch."""
    prev_length = _validate_inputs(
        q, k, v, prev_k, prev_v, api_name=api_name,
        allow_short_q=True,
    )
    use_fp32_state = use_high_precision_output(
        high_precision_output,
        (q, k, v, prev_k, prev_v),
        api_name=api_name,
    )
    _check_bos_exclusive(bos_mask, segment_idx, q_segment_idx, k_segment_idx)
    if bos_mask is not None:
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            num_q_runs,
            num_k_runs,
            metadata_is_dense,
        ) = _prepare_bos_metadata(q, k, prev_length, bos_mask, backend)
        if metadata_is_dense:
            return _CausalAttentionFunc.apply(
                q,
                k,
                v,
                window_size,
                _resolve_scale(q, scale),
                prev_k,
                prev_v,
                None,
                None,
                backend,
                deterministic,
                use_fp32_state,
            )
        needs_backward = torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad
            for tensor in (q, k, v, prev_k, prev_v)
        )
        if requires_row_aligned_k_runs(q, cu_seqlens_q, cu_seqlens_k):
            fwd_use_varlen = bwd_use_varlen = False
        else:
            fwd_use_varlen, bwd_use_varlen = select_flash_swa_bos_routes(
                q,
                k,
                v,
                _effective_window(k, prev_k, window_size),
                num_q_runs,
                max_seqlen_q,
                deterministic,
                needs_backward,
                full_visibility=window_size is None,
                has_previous=prev_k is not None,
                high_precision_output=use_fp32_state,
                previous_length=prev_length,
                num_k_runs=num_k_runs,
                max_seqlen_k=max_seqlen_k,
            )
        if not fwd_use_varlen or not bwd_use_varlen:
            q_segment_idx, k_segment_idx = _bos_segment_indices(q, bos_mask)
        return _CausalAttentionBosFunc.apply(
            q,
            k,
            v,
            window_size,
            _resolve_scale(q, scale),
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            backend,
            deterministic,
            fwd_use_varlen,
            bwd_use_varlen,
            use_fp32_state,
        )
    q_segment_idx, k_segment_idx = _prepare_segment_indices(
        q,
        k,
        prev_length,
        segment_idx,
        q_segment_idx,
        k_segment_idx,
        api_name=api_name,
    )
    return _CausalAttentionFunc.apply(
        q,
        k,
        v,
        window_size,
        _resolve_scale(q, scale),
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        backend,
        deterministic,
        use_fp32_state,
    )


def _causal_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: Optional[int],
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    api_name: str = "flash_swa",
    high_precision_output: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Shared causal full/window attention dispatch."""
    if not isinstance(high_precision_output, bool):
        raise TypeError(f"{api_name} high_precision_output must be a bool")
    prev_length = _validate_inputs(
        q, k, v, prev_k, prev_v, api_name=api_name,
        allow_short_q=True,
    )
    output_state = allocate_backward_output_state(q, v) if high_precision_output else None
    output_options = {"output_state": output_state} if output_state is not None else {}
    _check_bos_exclusive(bos_mask, segment_idx, q_segment_idx, k_segment_idx)
    resolved_scale = _resolve_scale(q, scale)
    if bos_mask is not None:
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            num_q_runs,
            num_k_runs,
            metadata_is_dense,
        ) = _prepare_bos_metadata(q, k, prev_length, bos_mask, backend)
        if metadata_is_dense:
            q_segment_idx = k_segment_idx = None
        else:
            use_varlen, _ = select_flash_swa_bos_routes(
                q,
                k,
                v,
                _effective_window(k, prev_k, window_size),
                num_q_runs,
                max_seqlen_q,
                False,
                False,
                full_visibility=window_size is None,
                has_previous=prev_k is not None,
                high_precision_output=high_precision_output,
                previous_length=prev_length,
                num_k_runs=num_k_runs,
                max_seqlen_k=max_seqlen_k,
            )
            if use_varlen and not requires_row_aligned_k_runs(
                q, cu_seqlens_q, cu_seqlens_k
            ):
                result = _batched_varlen_fwd(
                    q,
                    k,
                    v,
                    window_size,
                    resolved_scale,
                    prev_k,
                    prev_v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    backend,
                    **output_options,
                )
                return result[0], output_state, result[1]
            q_segment_idx, k_segment_idx = _bos_segment_indices(q, bos_mask)
        result = _dense_fwd(
            q,
            k,
            v,
            window_size,
            resolved_scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            backend,
            **output_options,
        )
        return result[0], output_state, result[1]
    q_segment_idx, k_segment_idx = _prepare_segment_indices(
        q,
        k,
        prev_length,
        segment_idx,
        q_segment_idx,
        k_segment_idx,
        api_name=api_name,
    )
    result = _dense_fwd(
        q,
        k,
        v,
        window_size,
        resolved_scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        backend,
        **output_options,
    )

    return result[0], output_state, result[1]


def _causal_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    window_size: Optional[int],
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
    api_name: str = "flash_swa",
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Shared causal full/window attention dispatch."""
    prev_length = _validate_inputs(
        q, k, v, prev_k, prev_v, api_name=api_name,
        allow_short_q=True,
    )
    _check_bos_exclusive(bos_mask, segment_idx, q_segment_idx, k_segment_idx)
    resolved_scale = _resolve_scale(q, scale)
    if bos_mask is not None:
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            num_q_runs,
            num_k_runs,
            metadata_is_dense,
        ) = _prepare_bos_metadata(q, k, prev_length, bos_mask, backend)
        if metadata_is_dense:
            q_segment_idx = k_segment_idx = None
        else:
            _, use_varlen = select_flash_swa_bos_routes(
                q,
                k,
                v,
                _effective_window(k, prev_k, window_size),
                num_q_runs,
                max_seqlen_q,
                deterministic,
                True,
                full_visibility=window_size is None,
                has_previous=prev_k is not None,
                high_precision_output=y.dtype == torch.float32,
                previous_length=prev_length,
                num_k_runs=num_k_runs,
                max_seqlen_k=max_seqlen_k,
            )
            if use_varlen and not requires_row_aligned_k_runs(
                q, cu_seqlens_q, cu_seqlens_k
            ):
                return _batched_varlen_bwd(
                    y_grad,
                    q,
                    k,
                    v,
                    y,
                    lse,
                    window_size,
                    resolved_scale,
                    prev_k,
                    prev_v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    deterministic,
                    backend,
                )
            q_segment_idx, k_segment_idx = _bos_segment_indices(q, bos_mask)
        return _dense_bwd(
            y_grad,
            q,
            k,
            v,
            y,
            lse,
            window_size,
            resolved_scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            deterministic,
            backend,
        )
    q_segment_idx, k_segment_idx = _prepare_segment_indices(
        q,
        k,
        prev_length,
        segment_idx,
        q_segment_idx,
        k_segment_idx,
        api_name=api_name,
    )
    return _dense_bwd(
        y_grad,
        q,
        k,
        v,
        y,
        lse,
        window_size,
        resolved_scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        deterministic,
        backend,
    )


def _flash_swa_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: Optional[int],
    scale: Optional[float] = None,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    backend: str = "auto",
    deterministic: bool = False,
    high_precision_output: bool = False,
) -> Tensor:
    """Run packed varlen FlashSWA with each cu-seqlens entry isolated.

    Q sequence ``r`` is paired with the right-aligned K sequence ``r``.
    The cu-seqlens boundaries are the segment boundaries, so attention
    never crosses from one packed sequence into another.
    """
    _validate_varlen_inputs(
        q,
        k,
        v,
        window_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        "_flash_swa_varlen",
    )
    use_fp32_state = use_high_precision_output(
        high_precision_output,
        (q, k, v),
        api_name="_flash_swa_varlen",
    )
    return _FlashSWAVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_size,
        _resolve_scale(q, scale),
        backend,
        deterministic,
        use_fp32_state,
    )


def _flash_swa_varlen_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: Optional[int],
    scale: Optional[float] = None,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    backend: str = "auto",
    output_state: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the low-level packed varlen FlashSWA forward pass."""
    _validate_varlen_inputs(
        q,
        k,
        v,
        window_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        "_flash_swa_varlen_fwd",
    )
    args = (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_size,
        _resolve_scale(q, scale),
        backend,
    )
    if output_state is not None:
        args = (*args, False, output_state)
    return _extension_ops()._flash_swa_sm90_varlen_fwd(*args)


def _flash_swa_varlen_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    window_size: Optional[int],
    scale: Optional[float] = None,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    backend: str = "auto",
    deterministic: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run the low-level packed varlen FlashSWA backward pass."""
    _validate_varlen_inputs(
        q,
        k,
        v,
        window_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        "_flash_swa_varlen_bwd",
    )
    return _extension_ops()._flash_swa_sm90_varlen_bwd(
        y_grad,
        q,
        k,
        v,
        y,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_size,
        _resolve_scale(q, scale),
        deterministic,
        backend,
    )

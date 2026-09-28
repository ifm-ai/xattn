# Author: Shicheng Wen

from typing import Optional

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ._backward_precision import allocate_backward_output_state
from ._bos_metadata import bos_to_segment_idx, build_bos_metadata
from ._causal_attention import (
    _batched_varlen_bwd,
    _batched_varlen_fwd,
    _dense_bwd,
    _dense_fwd,
)
from .sliding_chunk_attention import _flash_sca_bwd, _flash_sca_fwd


def _extension_ops():
    try:
        import xattn_cuda
    except ImportError as exc:
        raise ImportError(
            "xattn CUDA extension is required by Flash SoftDelta"
        ) from exc
    return xattn_cuda.ops


def _window_fwd(*args):
    # Precise normalization for both saved-state dtypes.
    return _dense_fwd(*args, ops=_extension_ops(), use_fast_reciprocal=False)


def _window_strict_past_fwd(*args):
    return _dense_fwd(
        *args, strict_past=True, ops=_extension_ops(), use_fast_reciprocal=False
    )


def _window_bwd(*args):
    return _dense_bwd(*args, ops=_extension_ops())


def _window_strict_past_bwd(*args):
    return _dense_bwd(*args, strict_past=True, ops=_extension_ops())


def flash_softdelta_bos_metadata(
    q: Tensor,
    bos_mask: Tensor,
    k_length: int,
) -> tuple[Tensor, Tensor, int, int, int, int, bool]:
    return build_bos_metadata(
        _extension_ops(), q, bos_mask, k_length
    )


def flash_softdelta_bos_segment_idx(bos_mask: Tensor) -> Tensor:
    return bos_to_segment_idx(_extension_ops(), bos_mask)


class _FlashSoftDeltaGateFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        read: Tensor,
        correction: Tensor,
        gate: Tensor,
    ) -> Tensor:
        output = _extension_ops()._flash_softdelta_gate_fwd(
            read,
            correction,
            gate,
        )
        ctx.save_for_backward(correction, gate)
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, output_grad: Tensor) -> tuple:
        correction, gate = ctx.saved_tensors
        return _extension_ops()._flash_softdelta_gate_bwd(
            output_grad.contiguous(),
            correction,
            gate,
        )


def flash_softdelta_gate(
    read: Tensor,
    correction: Tensor,
    gate: Tensor,
) -> Tensor:
    read = read.contiguous()
    correction = correction.contiguous()
    gate = gate.contiguous()
    if not torch.is_grad_enabled():
        return _extension_ops()._flash_softdelta_gate_fwd(
            read,
            correction,
            gate,
        )
    return _FlashSoftDeltaGateFunc.apply(
        read,
        correction,
        gate,
    )


def flash_softdelta_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    span: int,
    scale: float,
    visibility: int,
) -> Tensor:
    return _extension_ops()._flash_softdelta_fwd(
        q,
        k,
        v,
        gate,
        span,
        scale,
        visibility,
    )


def _use_window_composed_backward(q, k, v, span, visibility,
                                  deterministic, high_precision_output):
    """Restore the measured ordinary-reader BWD for underfilled KV grids."""
    value_dim = v.shape[3] * v.shape[4]
    restored_shape = (
        k.shape[2] == 1
        and (q.shape[3], value_dim) in ((32, 64), (128, 128), (192, 96), (256, 160))
    ) or (k.shape[2] == 2 and (q.shape[3], value_dim) == (256, 160))
    # Additional rectangular shapes for the 4K composed route.
    restored_shape = restored_shape or (
        k.shape[2] == 1 and q.shape[1] == 4096
        and (q.shape[3], value_dim) in (
            (32, 32),
            (32, 96),
            (32, 128),
            (32, 160),
            (32, 192),
            (32, 256),
            (64, 32),
            (64, 64),
            (64, 96),
            (64, 128),
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
            (128, 160),
            (128, 192),
            (128, 256),
            (160, 32),
            (160, 64),
            (160, 96),
            (160, 192),
            (192, 32),
            (192, 64),
            (192, 160),
            (256, 32),
            (256, 64),
            (256, 96),
            (256, 128),
        )
    )
    return (
        visibility == 1 and span == 255 and not deterministic
        and not high_precision_output and restored_shape
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[0] == 1 and q.shape[2] == 16 and v.shape[3] == 4
        and q.shape[1] * k.shape[2] < 16384
        and torch.cuda.get_device_name(q.device) == "NVIDIA H200"
    )


class _FlashSoftDeltaPairedTrainingFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        gate: Tensor,
        span: int,
        scale: float,
        visibility: int,
        deterministic: bool,
        high_precision_output: bool,
    ) -> Tensor:
        flat_v = v.flatten(3)
        ctx.composed_backward = _use_window_composed_backward(
            q, k, v, span, visibility, deterministic, high_precision_output
        )
        output_state = (
            allocate_backward_output_state(q, flat_v)
            if high_precision_output
            else None
        )
        output, pair_output, pair_lse = (
            _extension_ops()._flash_softdelta_training_fwd(
                q,
                k,
                v,
                gate,
                span,
                scale,
                visibility,
                output_state,
            )
        )
        gate_backward_state = (
            output_state if output_state is not None else pair_output
        )
        ctx.save_for_backward(
            q,
            k,
            flat_v,
            gate,
            gate_backward_state,
            gate_backward_state,
            pair_lse,
        )
        ctx.span = span
        ctx.scale = scale
        ctx.visibility = visibility
        ctx.deterministic = deterministic
        ctx.v_shape = v.shape
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, output_grad: Tensor) -> tuple:
        (
            q,
            k,
            flat_v,
            gate,
            gate_backward_state,
            attention_output_state,
            pair_lse,
        ) = ctx.saved_tensors
        if ctx.composed_backward:
            # Composed backward consumes both saved reader states.
            q_grad, k_grad, v_grad, gate_grad = _extension_ops()._flash_softdelta_bwd(
                output_grad, q, k, flat_v, gate,
                attention_output_state[:, :, 0::2].contiguous(),
                pair_lse[:, 0::2].contiguous(),
                attention_output_state[:, :, 1::2].contiguous(),
                pair_lse[:, 1::2].contiguous(),
                ctx.span, ctx.scale, None, None, None, None,
                ctx.visibility, False, ctx.deterministic,
            )[:4]
        else:
            q_grad, k_grad, v_grad, gate_grad = _extension_ops()._flash_softdelta_paired_bwd(
                output_grad,
                q,
                k,
                flat_v,
                gate,
                gate_backward_state,
                attention_output_state,
                pair_lse,
                ctx.span,
                ctx.scale,
                ctx.visibility,
                ctx.deterministic,
            )
        return (
            q_grad,
            k_grad,
            v_grad.reshape(ctx.v_shape),
            gate_grad,
            None,
            None,
            None,
            None,
            None,
        )


def flash_softdelta_paired_training(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    span: int,
    scale: float,
    visibility: int,
    deterministic: bool,
    high_precision_output: bool,
) -> Tensor:
    return _FlashSoftDeltaPairedTrainingFunc.apply(
        q,
        k,
        v,
        gate,
        span,
        scale,
        visibility,
        deterministic,
        high_precision_output,
    )


class _FlashSoftDeltaFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        gate: Tensor,
        span: int,
        scale: float,
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        visibility: int,
        reset_chunk_pos_per_seq: bool,
        deterministic: bool,
    ) -> Tensor:
        operations = _extension_ops()
        flat_v = v.flatten(3)
        flat_prev_v = None if prev_v is None else prev_v.flatten(3)
        read_q = q[:, :, 0::2]
        correction_q = q[:, :, 1::2]
        if visibility == 2:
            read, read_lse = operations.flash_sca_fwd(
                read_q,
                k,
                flat_v,
                span,
                scale,
                prev_k,
                flat_prev_v,
                q_segment_idx,
                k_segment_idx,
                "sm90",
                reset_chunk_pos_per_seq,
            )
            correction, correction_lse = (
                operations._flash_sca_sm90_strict_past_fwd(
                    correction_q,
                    k,
                    flat_v,
                    span,
                    scale,
                    prev_k,
                    flat_prev_v,
                    q_segment_idx,
                    k_segment_idx,
                    "sm90",
                    reset_chunk_pos_per_seq,
                )
            )
        else:
            read, read_lse = _window_fwd(
                read_q,
                k,
                flat_v,
                None if visibility == 0 else span,
                scale,
                prev_k,
                flat_prev_v,
                q_segment_idx,
                k_segment_idx,
                "sm90",
            )
            correction, correction_lse = (
                _window_strict_past_fwd(
                    correction_q,
                    k,
                    flat_v,
                    None if visibility == 0 else span,
                    scale,
                    prev_k,
                    flat_prev_v,
                    q_segment_idx,
                    k_segment_idx,
                    "sm90",
                )
            )
        gate = gate.contiguous()
        output_shape = (*read.shape[:3], *v.shape[3:])
        output = operations._flash_softdelta_gate_fwd(
            read.reshape(output_shape),
            correction.reshape(output_shape),
            gate,
        )
        ctx.save_for_backward(
            q,
            k,
            flat_v,
            gate,
            read,
            read_lse,
            correction,
            correction_lse,
            prev_k,
            flat_prev_v,
            q_segment_idx,
            k_segment_idx,
        )
        ctx.span = span
        ctx.scale = scale
        ctx.visibility = visibility
        ctx.reset_chunk_pos_per_seq = reset_chunk_pos_per_seq
        ctx.deterministic = deterministic
        ctx.v_shape = v.shape
        ctx.prev_v_shape = None if prev_v is None else prev_v.shape
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, output_grad: Tensor) -> tuple:
        (
            q,
            k,
            flat_v,
            gate,
            read,
            read_lse,
            correction,
            correction_lse,
            prev_k,
            flat_prev_v,
            q_segment_idx,
            k_segment_idx,
        ) = ctx.saved_tensors
        (
            q_grad,
            k_grad,
            v_grad,
            gate_grad,
            prev_k_grad,
            prev_v_grad,
        ) = _extension_ops()._flash_softdelta_bwd(
            output_grad,
            q,
            k,
            flat_v,
            gate,
            read,
            read_lse,
            correction,
            correction_lse,
            ctx.span,
            ctx.scale,
            prev_k,
            flat_prev_v,
            q_segment_idx,
            k_segment_idx,
            ctx.visibility,
            ctx.reset_chunk_pos_per_seq,
            ctx.deterministic,
        )
        v_grad = v_grad.reshape(ctx.v_shape)
        if prev_v_grad is not None:
            prev_v_grad = prev_v_grad.reshape(ctx.prev_v_shape)
        return (
            q_grad,
            k_grad,
            v_grad,
            gate_grad,
            None,
            None,
            prev_k_grad,
            prev_v_grad,
            None,
            None,
            None,
            None,
            None,
        )


def flash_softdelta_composed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    span: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    visibility: int,
    reset_chunk_pos_per_seq: bool,
    deterministic: bool,
) -> Tensor:
    return _FlashSoftDeltaFunc.apply(
        q,
        k,
        v,
        gate,
        span,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        visibility,
        reset_chunk_pos_per_seq,
        deterministic,
    )


class _FlashWindowReadFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        window_size: int,
        scale: float,
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        strict_past: bool,
        deterministic: bool,
        high_precision_output: bool,
    ) -> Tensor:
        operation = (
            _window_strict_past_fwd
            if strict_past
            else _window_fwd
        )
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
            "sm90",
        )
        if output_state is not None:
            args = (*args, output_state)
        y, lse = operation(*args)
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
        ctx.strict_past = strict_past
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
        operation = (
            _window_strict_past_bwd
            if ctx.strict_past
            else _window_bwd
        )
        dq, dk, dv, dprev_k, dprev_v = (
            operation(
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
                "sm90",
            )
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


class _FlashWindowBosReadFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        window_size: int,
        scale: float,
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        fwd_use_varlen: bool,
        bwd_use_varlen: bool,
        strict_past: bool,
        deterministic: bool,
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
                "sm90",
                strict_past,
                output_state,
            )
        else:
            operation = (
                _window_strict_past_fwd
                if strict_past
                else _window_fwd
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
                "sm90",
            )
            if output_state is not None:
                args = (*args, output_state)
            y, lse = operation(*args)
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
        ctx.bwd_use_varlen = bwd_use_varlen
        ctx.strict_past = strict_past
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
                "sm90",
                ctx.strict_past,
            )
        else:
            operation = (
                _window_strict_past_bwd
                if ctx.strict_past
                else _window_bwd
            )
            grads = operation(
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
                "sm90",
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


def flash_window_bos_read(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    fwd_use_varlen: bool,
    bwd_use_varlen: bool,
    strict_past: bool,
    deterministic: bool,
    high_precision_output: bool = False,
) -> Tensor:
    return _FlashWindowBosReadFunc.apply(
        q,
        k,
        v,
        window_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        fwd_use_varlen,
        bwd_use_varlen,
        strict_past,
        deterministic,
        high_precision_output,
    )


def flash_window_inclusive_read(
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
    high_precision_output: bool = False,
) -> Tensor:
    if not torch.is_grad_enabled():
        return _window_fwd(
            q,
            k,
            v,
            window_size,
            scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            "sm90",
        )[0]
    return _FlashWindowReadFunc.apply(
        q,
        k,
        v,
        window_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        False,
        deterministic,
        high_precision_output,
    )


def flash_window_strict_past_read(
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
    high_precision_output: bool = False,
) -> Tensor:
    if not torch.is_grad_enabled():
        return _window_strict_past_fwd(
            q,
            k,
            v,
            window_size,
            scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            "sm90",
        )[0]
    return _FlashWindowReadFunc.apply(
        q,
        k,
        v,
        window_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        True,
        deterministic,
        high_precision_output,
    )


class _FlashChunkReadFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        scale: float,
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        reset_chunk_pos_per_seq: bool,
        strict_past: bool,
        deterministic: bool,
        high_precision_output: bool,
    ) -> Tensor:
        operation = (
            _extension_ops()._flash_sca_sm90_strict_past_fwd
            if strict_past
            else _extension_ops().flash_sca_fwd
        )
        output_state = (
            allocate_backward_output_state(q, v)
            if high_precision_output
            else None
        )
        args = (
            q,
            k,
            v,
            chunk_size,
            scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            "sm90",
            reset_chunk_pos_per_seq,
        )
        if output_state is not None:
            args = (*args, output_state)
        y, lse = operation(*args)
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
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.reset_chunk_pos_per_seq = reset_chunk_pos_per_seq
        ctx.strict_past = strict_past
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
        operation = (
            _extension_ops()._flash_sca_sm90_strict_past_bwd
            if ctx.strict_past
            else _extension_ops().flash_sca_bwd
        )
        dq, dk, dv, dprev_k, dprev_v = (
            operation(
                y_grad,
                q,
                k,
                v,
                y,
                lse,
                ctx.chunk_size,
                ctx.scale,
                prev_k,
                prev_v,
                q_segment_idx,
                k_segment_idx,
                ctx.deterministic,
                "sm90",
                ctx.reset_chunk_pos_per_seq,
            )
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
            None,
        )


class _FlashChunkBosReadFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        scale: float,
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        reset_chunk_pos_per_seq: bool,
        fwd_use_varlen: bool,
        bwd_use_varlen: bool,
        strict_past: bool,
        deterministic: bool,
        high_precision_output: bool,
    ) -> Tensor:
        output_state = (
            allocate_backward_output_state(q, v)
            if high_precision_output
            else None
        )
        y, lse = _flash_sca_fwd(
            q,
            k,
            v,
            chunk_size,
            scale,
            prev_k,
            prev_v,
            None if fwd_use_varlen else q_segment_idx,
            None if fwd_use_varlen else k_segment_idx,
            "sm90",
            reset_chunk_pos_per_seq,
            cu_seqlens_q=cu_seqlens_q if fwd_use_varlen else None,
            cu_seqlens_k=cu_seqlens_k if fwd_use_varlen else None,
            max_seqlen_q=max_seqlen_q if fwd_use_varlen else None,
            max_seqlen_k=max_seqlen_k if fwd_use_varlen else None,
            strict_past=strict_past,
            output_state=output_state,
        )
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
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.reset_chunk_pos_per_seq = reset_chunk_pos_per_seq
        ctx.bwd_use_varlen = bwd_use_varlen
        ctx.strict_past = strict_past
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
            cu_seqlens_q,
            cu_seqlens_k,
        ) = ctx.saved_tensors
        grads = _flash_sca_bwd(
            y_grad,
            q,
            k,
            v,
            y,
            lse,
            ctx.chunk_size,
            ctx.scale,
            prev_k,
            prev_v,
            None if ctx.bwd_use_varlen else q_segment_idx,
            None if ctx.bwd_use_varlen else k_segment_idx,
            ctx.deterministic,
            "sm90",
            ctx.reset_chunk_pos_per_seq,
            cu_seqlens_q=cu_seqlens_q if ctx.bwd_use_varlen else None,
            cu_seqlens_k=cu_seqlens_k if ctx.bwd_use_varlen else None,
            max_seqlen_q=ctx.max_seqlen_q if ctx.bwd_use_varlen else None,
            max_seqlen_k=ctx.max_seqlen_k if ctx.bwd_use_varlen else None,
            strict_past=ctx.strict_past,
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
            None,
        )


def flash_chunk_bos_read(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    reset_chunk_pos_per_seq: bool,
    fwd_use_varlen: bool,
    bwd_use_varlen: bool,
    strict_past: bool,
    deterministic: bool,
    high_precision_output: bool = False,
) -> Tensor:
    return _FlashChunkBosReadFunc.apply(
        q,
        k,
        v,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        reset_chunk_pos_per_seq,
        fwd_use_varlen,
        bwd_use_varlen,
        strict_past,
        deterministic,
        high_precision_output,
    )


def flash_chunk_inclusive_read(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    reset_chunk_pos_per_seq: bool,
    deterministic: bool,
    high_precision_output: bool = False,
) -> Tensor:
    if not torch.is_grad_enabled():
        return _extension_ops().flash_sca_fwd(
            q,
            k,
            v,
            chunk_size,
            scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            "sm90",
            reset_chunk_pos_per_seq,
        )[0]
    return _FlashChunkReadFunc.apply(
        q,
        k,
        v,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        reset_chunk_pos_per_seq,
        False,
        deterministic,
        high_precision_output,
    )


def flash_chunk_strict_past_read(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    reset_chunk_pos_per_seq: bool,
    deterministic: bool,
    high_precision_output: bool = False,
) -> Tensor:
    if not torch.is_grad_enabled():
        return _extension_ops()._flash_sca_sm90_strict_past_fwd(
            q,
            k,
            v,
            chunk_size,
            scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            "sm90",
            reset_chunk_pos_per_seq,
        )[0]
    return _FlashChunkReadFunc.apply(
        q,
        k,
        v,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        reset_chunk_pos_per_seq,
        True,
        deterministic,
        high_precision_output,
    )

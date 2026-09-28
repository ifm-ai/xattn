# Author: Shicheng Wen

"""Standalone SoftDelta state and gradients, using the production SM90 routes."""

import torch

from . import _flash_softdelta as flash
from ._sca_segment_alignment import needs_matched_reset, matched_forward, matched_backward
from ._attention_core import AttentionVisibility, resolve_scale
from ._backward_precision import allocate_backward_output_state
from ._causal_attention import _batched_varlen_bwd, _batched_varlen_fwd
from .sliding_chunk_attention import _flash_sca_bwd, _flash_sca_fwd
from .softdelta_attention import (
    _make_plan, _normalize_backend, _paired_training_route,
    _prepare_flash_bos_plan, _prepare_partition, _use_sm90_backend,
    _validate_inputs,
)


def _prepare(q, k, v, g, *, visibility, span, scale, prev_k, prev_v,
             q_segment_idx, k_segment_idx, segment_idx, bos_mask, backend,
             reset, deterministic, high_precision_output, api_name):
    backend = _normalize_backend(backend)
    _, _, _, prev_length = _validate_inputs(
        q, k, v, g, prev_k, prev_v, deterministic, api_name,
        allow_short_q=True,
    )
    if not isinstance(high_precision_output, bool):
        raise TypeError(f"{api_name} high_precision_output must be a bool")
    if not isinstance(reset, bool):
        raise TypeError(f"{api_name} reset_chunk_pos_per_seq must be a bool")
    plan = _make_plan(visibility, span, reset)
    if visibility is AttentionVisibility.SLIDING_CHUNK and prev_k is not None:
        if prev_length != span:
            raise ValueError(f"{api_name} previous sequence length must equal chunk_size")
    if not _use_sm90_backend(backend, q):
        raise ValueError(f"{api_name} requires the SM90 CUDA backend")
    total_length = k.shape[1] + prev_length
    bos = None
    if bos_mask is not None:
        if any(x is not None for x in (segment_idx, q_segment_idx, k_segment_idx)):
            raise ValueError(f"{api_name} BOS masks and segment indices are mutually exclusive")
        q_segment_idx, k_segment_idx, bos = _prepare_flash_bos_plan(
            q, k, v, plan, total_length, bos_mask, deterministic, True,
            api_name, high_precision_output=high_precision_output,
        )
    else:
        q_segment_idx, k_segment_idx = _prepare_partition(
            q, total_length, q_segment_idx, k_segment_idx, segment_idx, None,
            api_name,
        )
    span = total_length - 1 if visibility is AttentionVisibility.CAUSAL_FULL else span
    _, paired = _paired_training_route(
        q, k, v, plan, span, prev_k, prev_v, q_segment_idx, k_segment_idx,
        bos, deterministic, high_precision_output,
    )
    visibility = {
        AttentionVisibility.CAUSAL_FULL: 0,
        AttentionVisibility.SLIDING_WINDOW: 1,
        AttentionVisibility.SLIDING_CHUNK: 2,
    }[visibility]
    return span, resolve_scale(q, scale), q_segment_idx, k_segment_idx, bos, paired, visibility


def _reader(q, k, v, *, span, scale, prev_k, prev_v, qi, ki, bos,
            visibility, reset, strict, deterministic, state=None,
            dy=None, y=None, lse=None):
    backward = dy is not None
    varlen = bos is not None and (bos.bwd_use_varlen if backward else bos.fwd_use_varlen)
    prefix = (dy, q, k, v, y, lse) if backward else (q, k, v)
    if visibility == 2:
        if needs_matched_reset(q, qi, ki, reset):
            if backward:
                return matched_backward(dy, q, k, v, y, lse, span, scale,
                    prev_k, prev_v, qi, ki, "sm90", "default", deterministic,
                    strict_past=strict)
            output, saved, lp = matched_forward(q, k, v, span, scale,
                prev_k, prev_v, qi, ki, "sm90", "default", state is not None,
                strict_past=strict)
            if state is not None:
                state.copy_(saved)
            return output, lp
        metadata = {} if not varlen else dict(
            cu_seqlens_q=bos.cu_seqlens_q, cu_seqlens_k=bos.cu_seqlens_k,
            max_seqlen_q=bos.max_seqlen_q, max_seqlen_k=bos.max_seqlen_k,
        )
        args = (*prefix, span, scale, prev_k, prev_v,
                None if varlen else qi, None if varlen else ki)
        if backward:
            return _flash_sca_bwd(*args, deterministic, "sm90", reset,
                                  strict_past=strict, **metadata)
        return _flash_sca_fwd(*args, "sm90", reset, strict_past=strict,
                              output_state=state, **metadata)
    window = None if visibility == 0 else span
    if varlen:
        args = (*prefix, window, scale, prev_k, prev_v,
                bos.cu_seqlens_q, bos.cu_seqlens_k,
                bos.max_seqlen_q, bos.max_seqlen_k)
        if backward:
            return _batched_varlen_bwd(*args, deterministic, "sm90", strict)
        return _batched_varlen_fwd(*args, "sm90", strict, state)
    args = (*prefix, window, scale, prev_k, prev_v, qi, ki)
    if backward:
        op = flash._window_strict_past_bwd if strict else flash._window_bwd
        return op(*args, deterministic, "sm90")
    op = flash._window_strict_past_fwd if strict else flash._window_fwd
    return op(*args, "sm90", *(() if state is None else (state,)))


def _pack_readers(first, second):
    return torch.stack((first, second), dim=3).flatten(2, 3)


@torch.no_grad()
def _softdelta_fwd(q, k, v, g, **options):
    span, scale, qi, ki, bos, paired, visibility = _prepare(q, k, v, g, **options)
    flat_v = v.flatten(3)
    hp = options['high_precision_output']
    ops = flash._extension_ops()
    shape = (*q.shape[:3], *v.shape[3:])
    if paired:
        state = allocate_backward_output_state(q, flat_v) if hp else None
        y, readers, lse = ops._flash_softdelta_training_fwd(
            q, k, v, g, span, scale, visibility, state,
        )
        return y, (state if hp else readers).reshape(shape), lse
    outputs, states, lses = [], [], []
    for i in range(2):
        reader_q = q[:, :, i::2]
        state = allocate_backward_output_state(reader_q, flat_v) if hp else None
        output, lse = _reader(
            reader_q, k, flat_v, span=span, scale=scale,
            prev_k=options['prev_k'],
            prev_v=None if options['prev_v'] is None else options['prev_v'].flatten(3),
            qi=qi, ki=ki, bos=bos, visibility=visibility, reset=options['reset'],
            strict=bool(i), deterministic=options['deterministic'], state=state,
        )
        outputs.append(output.reshape(*output.shape[:3], *v.shape[3:]))
        states.append(state)
        lses.append(lse)
    y = ops._flash_softdelta_gate_fwd(outputs[0], outputs[1], g.contiguous())
    readers = _pack_readers(*(states if hp else outputs)).reshape(shape)
    lse = torch.stack(lses, dim=2).flatten(1, 2)
    return y, readers, lse


@torch.no_grad()
def _softdelta_bwd(dy, q, k, v, g, readers, lse, **options):
    # Validate saved-state tensors before any native pointer access.
    _validate_inputs(q, k, v, g, options['prev_k'], options['prev_v'],
                     options['deterministic'], options['api_name'],
                     allow_short_q=True)
    for name, tensor, shape, dtypes in (
        ('y_grad', dy, (*g.shape[:-1], v.shape[-1]), (q.dtype,)),
        ('readers', readers, (*q.shape[:3], *v.shape[3:]), (q.dtype, torch.float32)),
        ('lse', lse, (q.shape[0], q.shape[2], q.shape[1]), (torch.float32,)),
    ):
        if tensor.shape != shape or tensor.device != q.device or tensor.dtype not in dtypes:
            raise ValueError(f"{options['api_name']} {name} has incompatible shape, device, or dtype")
    span, scale, qi, ki, bos, paired, visibility = _prepare(q, k, v, g, **options)
    flat_v = v.flatten(3)
    flat_prev_v = None if options['prev_v'] is None else options['prev_v'].flatten(3)
    readers = readers.flatten(3).contiguous()
    lse = lse.contiguous()
    dy, g = dy.contiguous(), g.contiguous()
    ops = flash._extension_ops()
    det = options['deterministic']
    hp = readers.dtype == torch.float32
    if paired and not flash._use_window_composed_backward(q, k, v, span, visibility, det, hp):
        dq, dk, dv, dg = ops._flash_softdelta_paired_bwd(
            dy, q, k, flat_v, g, readers, readers, lse, span, scale, visibility, det,
        )
        return dq, dk, dv.reshape(v.shape), dg, None, None
    matched_reset = visibility == 2 and needs_matched_reset(q, qi, ki, options['reset'])
    short_local = visibility != 0 and q.shape[1] != k.shape[1]
    if not hp and not matched_reset and not short_local and (bos is None or not bos.bwd_use_varlen):
        dq, dk, dv, dg, dpk, dpv = ops._flash_softdelta_bwd(
            dy, q, k, flat_v, g, readers[:, :, 0::2].contiguous(),
            lse[:, 0::2].contiguous(), readers[:, :, 1::2].contiguous(),
            lse[:, 1::2].contiguous(), span, scale, options['prev_k'], flat_prev_v,
            qi, ki, visibility, options['reset'], det,
        )
    else:
        # Pair gate BWD supports both FP32 and input-precision reader state.
        pair_grad, dg = ops._flash_softdelta_pair_gate_bwd(dy, readers, g)
        gradients = []
        for i in range(2):
            gradients.append(_reader(
                q[:, :, i::2], k, flat_v, span=span, scale=scale,
                prev_k=options['prev_k'], prev_v=flat_prev_v, qi=qi, ki=ki,
                bos=bos, visibility=visibility, reset=options['reset'],
                strict=bool(i), deterministic=det,
                dy=pair_grad[:, :, i::2].contiguous(),
                y=readers[:, :, i::2].contiguous(), lse=lse[:, i::2].contiguous(),
            ))
        dq = _pack_readers(gradients[0][0], gradients[1][0])
        dk, dv, dpk, dpv = (
            None if a is None else a + b
            for a, b in zip(gradients[0][1:], gradients[1][1:])
        )
    return dq, dk, dv.reshape(v.shape), dg, dpk, None if dpv is None else dpv.reshape(options['prev_v'].shape)


class _StandaloneSoftDelta(torch.autograd.Function):
    """Autograd bridge for rectangular local and independently reset readers."""
    @staticmethod
    def forward(ctx, q, k, v, g, prev_k, prev_v, options):
        options = dict(options, prev_k=prev_k, prev_v=prev_v)
        y, readers, lse = _softdelta_fwd(q, k, v, g, **options)
        ctx.save_for_backward(q, k, v, g, readers, lse, prev_k, prev_v)
        ctx.options = {k: v for k, v in options.items() if k not in ('prev_k', 'prev_v')}
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, g, readers, lse, prev_k, prev_v = ctx.saved_tensors
        grads = _softdelta_bwd(dy, q, k, v, g, readers, lse,
                               **ctx.options, prev_k=prev_k, prev_v=prev_v)
        return (*grads, None)

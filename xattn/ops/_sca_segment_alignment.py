"""Match independent reset-position segments before calling packed FlashSCA."""
from collections import OrderedDict
from dataclasses import dataclass
import weakref

import torch
from torch import Tensor

from ._segment_metadata import validate_segment_indices


@dataclass(frozen=True)
class _MatchedSegments:
    q_indices: Tensor
    k_indices: Tensor
    cu_q: Tensor
    cu_k: Tensor
    max_q: int
    max_k: int


_plans = OrderedDict()
_suffix_equivalence = OrderedDict()


def needs_matched_reset(q, q_idx, k_idx, reset, segment_idx=None, bos_mask=None):
    if (not reset or not q.is_cuda or q_idx is None or k_idx is None
            or segment_idx is not None or bos_mask is not None):
        return False
    # Common indices and their right-aligned Q views already use the fast path.
    if q_idx.dim() == k_idx.dim() == 2 and q_idx.shape[1] <= k_idx.shape[1]:
        suffix = k_idx[:, k_idx.shape[1] - q_idx.shape[1]:]
        if (q_idx.shape == suffix.shape and q_idx.stride() == suffix.stride()
                and q_idx.data_ptr() == suffix.data_ptr()):
            return False
        # Version-keyed cache for separately allocated common metadata.
        if (q_idx.device == k_idx.device == q.device
                and q_idx.dtype == k_idx.dtype == torch.int64
                and not torch.is_inference(q_idx) and not torch.is_inference(k_idx)):
            key = (id(q_idx), id(k_idx), q_idx._version, k_idx._version,
                   torch.cuda.current_stream(q.device).cuda_stream)
            cached = _suffix_equivalence.get(key)
            if cached is not None and cached[0]() is q_idx and cached[1]() is k_idx:
                _suffix_equivalence.move_to_end(key)
                return not cached[2]
            same = torch.equal(q_idx, suffix)
            _suffix_equivalence[key] = (weakref.ref(q_idx), weakref.ref(k_idx), same)
            while len(_suffix_equivalence) > 8:
                _suffix_equivalence.popitem(last=False)
            return not same
    return True


def _build_plan(q_idx: Tensor, k_idx: Tensor) -> _MatchedSegments:
    cacheable = not torch.is_inference(q_idx) and not torch.is_inference(k_idx)
    key = None
    if cacheable:
        key = (id(q_idx), id(k_idx), q_idx._version, k_idx._version,
               torch.cuda.current_stream(q_idx.device).cuda_stream)
        cached = _plans.get(key)
        if cached is not None and cached[0]() is q_idx and cached[1]() is k_idx:
            _plans.move_to_end(key)
            return cached[2]
    # Plans created during inference must remain usable by a later backward.
    with torch.inference_mode(False), torch.no_grad():
        qi, ki = q_idx.contiguous(), k_idx.contiguous()
        lq, lk = qi.shape[1], ki.shape[1]
        lower = torch.searchsorted(ki, qi)
        upper = torch.searchsorted(ki, qi, right=True)
        q_keep = upper > lower
        k_lower = torch.searchsorted(qi, ki)
        k_keep = (k_lower < lq) & (qi.gather(1, k_lower.clamp(max=lq - 1)) == ki)
        q_indices = q_keep.flatten().nonzero().flatten()
        k_indices = k_keep.flatten().nonzero().flatten()
        starts = torch.ones_like(qi, dtype=torch.bool)
        starts[:, 1:] = qi[:, 1:] != qi[:, :-1]
        start_indices = starts.flatten().nonzero().flatten()
        ends = torch.cat((start_indices[1:], start_indices.new_tensor([qi.numel()])))
        q_lengths = ends - start_indices
        k_lengths = (upper - lower).flatten().index_select(0, start_indices)
        matched = k_lengths > 0
        q_lengths, k_lengths = q_lengths[matched], k_lengths[matched]
        zero = torch.zeros(1, device=qi.device, dtype=torch.int32)
        cu_q = torch.cat((zero, q_lengths.cumsum(0, dtype=torch.int32)))
        cu_k = torch.cat((zero, k_lengths.cumsum(0, dtype=torch.int32)))
        plan = _MatchedSegments(q_indices, k_indices, cu_q, cu_k,
                                int(q_lengths.max().item()) if q_lengths.numel() else 0,
                                int(k_lengths.max().item()) if k_lengths.numel() else 0)
    if key is not None:
        _plans[key] = (weakref.ref(q_idx), weakref.ref(k_idx), plan)
        _plans.move_to_end(key)
        while len(_plans) > 8:
            _plans.popitem(last=False)
    return plan


def _prepare(q, k, v, chunk_size, scale, prev_k, prev_v, q_idx, k_idx,
             backend, attn_method):
    from . import sliding_chunk_attention as sca
    sca._check_attn_method_supported(attn_method)
    sca._validate_qkv_heads(q, k, v, packed=False, api_name="flash_sca")
    if (prev_k is None) != (prev_v is None):
        raise ValueError("prev_k and prev_v must be provided together")
    sca._validate_prev_kv_heads(q, k, v, prev_k, prev_v, chunk_size, api_name="flash_sca")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if backend.strip().lower() not in ("", "auto", "sm90"):
        raise ValueError("reset segment alignment requires the SM90 backend")
    total_k = k.shape[1] + (0 if prev_k is None else prev_k.shape[1])
    validate_segment_indices(q_idx, k_idx, batch_size=q.shape[0],
                             q_length=q.shape[1], k_length=total_k,
                             device=q.device, api_name="flash_sca")
    plan = _build_plan(q_idx, k_idx)
    kt = k if prev_k is None else torch.cat((prev_k, k), 1)
    vt = v if prev_v is None else torch.cat((prev_v, v), 1)
    packed = (q.flatten(0, 1).index_select(0, plan.q_indices),
              kt.flatten(0, 1).index_select(0, plan.k_indices),
              vt.flatten(0, 1).index_select(0, plan.k_indices))
    options = dict(cu_seqlens_q=plan.cu_q, cu_seqlens_k=plan.cu_k,
                   max_seqlen_q=plan.max_q, max_seqlen_k=plan.max_k,
                   reset_chunk_pos_per_seq=True, backend=backend, attn_method=attn_method)
    return sca, plan, packed, options, sca._resolve_scale(q, scale), kt, vt


def _restore(t, indices, batch, length):
    flat = t.new_zeros((batch * length, *t.shape[1:]))
    return flat.index_copy(0, indices, t).reshape(batch, length, *t.shape[1:])


def matched_forward(q, k, v, chunk_size, scale, prev_k, prev_v, q_idx, k_idx,
                    backend, attn_method, high_precision_output, deterministic=False,
                    *, autograd=False, strict_past=False):
    sca, plan, packed, opts, scale, _, _ = _prepare(
        q, k, v, chunk_size, scale, prev_k, prev_v, q_idx, k_idx, backend, attn_method)
    if not isinstance(high_precision_output, bool):
        raise TypeError("high_precision_output must be a bool")
    batch, length, heads = q.shape[:3]
    if not plan.max_q:
        y = q.new_zeros((batch, length, heads, v.shape[-1]))
        if autograd:
            # Retain zero gradients for every input, including unused previous KV.
            for x in (q, k, v, prev_k, prev_v):
                if x is not None:
                    y = y + (x.sum(dtype=torch.float32) * 0).to(y.dtype)
            return y
        state = y.float() if high_precision_output else None
        return y, state, q.new_full((batch, heads, length), float("inf"), dtype=torch.float32)
    if autograd:
        yp = sca._flash_sca_varlen(*packed, chunk_size, scale, **opts,
                                  deterministic=deterministic,
                                  high_precision_output=high_precision_output)
        return _restore(yp, plan.q_indices, batch, length)
    state_packed = (torch.empty((*packed[0].shape[:2], v.shape[-1]),
                               device=q.device, dtype=torch.float32)
                    if high_precision_output else None)
    yp, lp = sca._flash_sca_varlen_fwd(*packed, chunk_size, scale, **opts,
                                     output_state=state_packed, strict_past=strict_past)
    y = _restore(yp, plan.q_indices, batch, length)
    state = None if state_packed is None else _restore(state_packed, plan.q_indices, batch, length)
    lf = lp.new_full((heads, batch * length), float("inf"))
    lse = lf.index_copy(1, plan.q_indices, lp).reshape(heads, batch, length).permute(1, 0, 2).contiguous()
    return y, state, lse


def matched_backward(dy, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,
                     q_idx, k_idx, backend, attn_method, deterministic, *, strict_past=False):
    sca, plan, packed, opts, scale, kt, vt = _prepare(
        q, k, v, chunk_size, scale, prev_k, prev_v, q_idx, k_idx, backend, attn_method)
    if not plan.max_q:
        return tuple(None if t is None else torch.zeros_like(t) for t in (q, k, v, prev_k, prev_v))
    dyp = dy.flatten(0, 1).index_select(0, plan.q_indices)
    yp = y.flatten(0, 1).index_select(0, plan.q_indices)
    lp = lse.permute(1, 0, 2).reshape(q.shape[2], -1).index_select(1, plan.q_indices).contiguous()
    dq, dk, dv = sca._flash_sca_varlen_bwd(dyp, *packed, yp, lp, chunk_size, scale,
                                         **opts, deterministic=deterministic, strict_past=strict_past)
    dq = _restore(dq, plan.q_indices, q.shape[0], q.shape[1])
    dk = _restore(dk, plan.k_indices, kt.shape[0], kt.shape[1])
    dv = _restore(dv, plan.k_indices, vt.shape[0], vt.shape[1])
    if prev_k is None:
        return dq, dk, dv, None, None
    prev = prev_k.shape[1]
    return dq, dk[:, prev:].contiguous(), dv[:, prev:].contiguous(), dk[:, :prev].contiguous(), dv[:, :prev].contiguous()

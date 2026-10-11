"""Shared advanced-precision reader backward for the six SM90 families.

The visible-support policy is independent of numerical level. SoftDelta calls
this for each reader and combines FP32 accumulators before the public cast.
"""
from typing import Optional
import torch
from ._backward_precision import validate_high_precision_level
from ._segment_metadata import resolve_segment_indices, validate_segment_indices
from ._attention_core import resolve_scale


# Level 3/4 zero-sum repair spreads the non-anchor share of each row's dS
# operand residual over its last ZERO_SUM_WINDOW admitted keys.
ZERO_SUM_WINDOW = 64
_PRESCALE_TARGET = 2.0 ** 13


def _ops():
    import xattn_cuda
    ops = xattn_cuda.ops
    if not all(hasattr(ops, name) for name in (
            '_attention_precision_main_backward',
            '_attention_precision_zero_sum_repair',
            '_attention_precision_ordered_backward',
            '_attention_precision_key_runs',
            '_attention_precision_enforce_constant_keys')):
        raise RuntimeError('high_precision_level >= 2 requires rebuilding the xattn SM90 CUDA extension')
    return ops


def backward(dy, q, k, v, *, level, visibility, span=None, scale=None,
             prev_k=None, prev_v=None, q_segment_idx=None, k_segment_idx=None,
             segment_idx=None, bos_mask=None, reset=False, strict=False,
             deterministic=False, backend='sm90', return_fp32=False):
    validate_high_precision_level(level)
    if level < 2:
        raise ValueError('advanced precision backward requires level 2..4')
    if backend.strip().lower() not in ('', 'auto', 'sm90'):
        raise ValueError('advanced precision requires the SM90 backend')
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] != 9:
        raise ValueError('advanced precision requires an SM90 CUDA device')
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError('precision readers require rank-four Q/K/V')
    if (prev_k is None) != (prev_v is None):
        raise ValueError('prev_k and prev_v must be provided together')
    if dy.shape != (*q.shape[:-1], v.shape[-1]) or dy.dtype != q.dtype or dy.device != q.device:
        raise ValueError('dO must have the reader output shape and Q dtype/device')
    if visibility not in (0, 1, 2):
        raise ValueError('invalid precision visibility')
    if visibility and (isinstance(span, bool) or not isinstance(span, int) or span < (1 if visibility == 2 else 0)):
        raise ValueError('invalid window/chunk span')
    if prev_k is not None:
        if visibility == 2 and prev_k.shape[1] != span:
            raise ValueError('previous chunk length must equal chunk_size')
        for a, b in ((k, prev_k), (v, prev_v)):
            if a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:] or a.dtype != b.dtype or a.device != b.device:
                raise ValueError('previous K/V shapes, dtypes and devices must match current K/V')
    previous = 0 if prev_k is None else prev_k.shape[1]
    kt = k if prev_k is None else torch.cat((prev_k, k), dim=1)
    vt = v if prev_v is None else torch.cat((prev_v, v), dim=1)
    b, nq, h, _ = q.shape
    nk = kt.shape[1]
    if bos_mask is not None:
        if any(x is not None for x in (segment_idx, q_segment_idx, k_segment_idx)):
            raise ValueError('BOS masks and segment indices are mutually exclusive')
        if bos_mask.shape != (b, nk) or bos_mask.dtype != torch.bool or bos_mask.device != q.device:
            raise ValueError('bos_mask must be boolean [B,total_K] on Q device')
        segment_idx = bos_mask.to(torch.int64).cumsum(-1)
    qi, ki = resolve_segment_indices(segment_idx, q_segment_idx, k_segment_idx,
                                    batch_size=b, q_length=nq, k_length=nk)
    validate_segment_indices(qi, ki, batch_size=b, q_length=nq, k_length=nk,
                             device=q.device, api_name='precision backward')
    scale = resolve_scale(q, scale)
    # The kernels use a positive scale so masked -inf logits stay masked.
    # Exact sign/zero transforms preserve the public finite-scale contract.
    original_scale = scale
    if scale < 0:
        q, scale = -q, -scale
    elif scale == 0:
        q, scale = torch.zeros_like(q), 1.0
    if reset and visibility != 2:
        raise ValueError('reset positions apply only to sliding chunks')
    if reset and qi is not None:
        from ._sca_segment_alignment import needs_matched_reset
        if needs_matched_reset(q, qi, ki, True):
            outputs = _matched_backward(dy, q, kt, vt, qi, ki, level, span, scale, strict, deterministic)
            outputs = _restore_scale_gradients(outputs, original_scale)
            return _finish(outputs, q.dtype, previous, return_fp32)
    q, kt, vt, dy = (x.contiguous() for x in (q, kt, vt, dy))
    # A contiguous view can still start at an odd 16-bit storage offset.
    # Statistics uses 16-byte async transfers; repair alignment without casts.
    q, kt, vt, dy = (x.clone() if x.data_ptr() % 16 else x for x in (q, kt, vt, dy))
    qi = None if qi is None else qi.contiguous()
    ki = None if ki is None else ki.contiguous()
    position = torch.arange(nq, device=q.device, dtype=torch.int32)[None].expand(b, -1) + nk - nq
    right = position - int(strict)
    q_positions = None
    if visibility == 0:
        left = torch.zeros_like(position)
    elif visibility == 1:
        left = (position - span).clamp_min(0)
    elif reset and qi is not None:
        # Same formulas as FlashSCABuildDenseResetChunkPositionsKernel. Segment
        # IDs use the existing sorted-run contract, including short Q suffixes.
        qlo = torch.searchsorted(qi, qi)
        qhi = torch.searchsorted(qi, qi, right=True)
        klo = torch.searchsorted(ki, qi)
        khi = torch.searchsorted(ki, qi, right=True)
        local = (torch.arange(nq, device=q.device)[None] - qlo) + (khi - klo) - (qhi - qlo)
        q_positions = local.to(torch.int32).contiguous()
        left = (klo + ((local // span - 1) * span).clamp_min(0)).to(torch.int32)
        right = torch.minimum(klo + local - int(strict), khi - 1).to(torch.int32)
    else:
        left = ((position // span - 1) * span).clamp_min(0)
    exact = None
    if qi is not None and q_positions is None:
        # With nondecreasing key runs (always true for BOS masks) a query's
        # segment support is the key run [first, last]; intersecting it with
        # the visibility bounds makes them exact, so the row scans skip other
        # segments and per-key segment tests. Other layouts keep the
        # visibility bounds and the segment test. No host sync.
        runs = (ki[:, 1:] >= ki[:, :-1]).all(dim=1, keepdim=True)
        first = torch.searchsorted(ki, qi).to(torch.int32)
        last = torch.searchsorted(ki, qi, right=True).to(torch.int32) - 1
        left = torch.where(runs, torch.maximum(left, first), left)
        right = torch.where(runs, torch.minimum(right, last), right)
        exact = runs.to(torch.int32).reshape(b).contiguous()
    left, right = left.contiguous(), right.contiguous()
    ops = _ops()
    kc, _ = ops._attention_precision_center(kt)
    vc, _ = ops._attention_precision_center(vt)
    dy, inverse_prescale = _prescale_fp16(dy, vc)
    delta, logz, metadata = ops._attention_precision_row_statistics(q, kc, vc, dy, left, right, qi, ki, scale, exact)
    if level == 4:
        scaled_dy, factors = ops._attention_precision_scale_do(dy)
        metadata[2].copy_(factors)
    else:
        scaled_dy = dy
    # Row statistics of the dS operand: sum (rho), P mass, sum of squares
    # (level 3 only: the rounding-variance weight of the anchor share) and
    # anchor probability mass.
    stats = torch.zeros((4, b, h, nq), dtype=torch.float64, device=q.device)
    variance = (stats[2], delta) if level == 3 else (None, None)
    if deterministic:
        # dS is in unscaled dO units: level 4's row scale cancels in-kernel.
        exponent = _rho_exponent(dy, vc) if level >= 3 else None
        dq, dk, dv = ops._attention_precision_ordered_backward(
            q, kc, vc, scaled_dy, delta, logz, stats, metadata, qi, ki, q_positions,
            scale, level, visibility, span or 0, strict, reset, exponent)
        if level >= 3:
            _decode_fixed_statistics(stats, exponent, level)
            # The uniform shares need no atomics; the anchor share left in
            # stats[0] goes through the ordered repair below.
            ops._attention_precision_zero_sum_repair(
                q, kc, stats[0], metadata, logz, ki, ZERO_SUM_WINDOW, False, stats[0], dq, dk, scale,
                *variance)
            # Stable order within each KV head makes anchor scatter reductions
            # independent of CTA scheduling without an N² workspace.
            order = torch.argsort(metadata[3].reshape(b, kt.shape[2], -1), dim=-1, stable=True)
            flags = ops._attention_precision_deterministic_repair(
                q, kc, dy, stats, metadata, order, dq, dk, dv, scale, level)
    else:
        dq, dk, dv = ops._attention_precision_main_backward(
            q, kc, vc, scaled_dy, delta, logz, stats, metadata, qi, ki, q_positions,
            scale, level, visibility, span or 0, strict, reset)
        if level >= 3:
            ops._attention_precision_zero_sum_repair(
                q, kc, stats[0], metadata, logz, ki, ZERO_SUM_WINDOW, True,
                torch.empty_like(stats[0]), dq, dk, scale, *variance)
        if level == 4:
            flags = ops._attention_precision_probability_repair(dy, stats, metadata, dv)
    if level == 4:
        ops._attention_precision_fallback_dv(q, kc, scaled_dy, left, right, logz,
                                           metadata, flags, dv, qi, ki, scale)
    # Output-space subtraction can leave a huge FP32 cancellation residual
    # when all visible keys coincide and dO/V have a very large range.
    # Equality runs prove the exact dQ=0 invariant in O(ND), without an
    # extra attention contraction or a guessed small-gradient threshold.
    key_runs = ops._attention_precision_key_runs(kt)
    ops._attention_precision_enforce_constant_keys(key_runs, metadata, dq)
    if inverse_prescale is not None:
        for x in (dq, dk, dv):
            x.mul_(inverse_prescale)
    outputs = _restore_scale_gradients((dq, dk, dv), original_scale)
    return _finish(outputs, q.dtype, previous, return_fp32)


def _power_of_two(exponent):
    """Exact FP64 2**exponent for int64 exponents in the normal range."""
    return ((exponent + 1023) << 52).view(torch.float64)


def _rho_exponent(dy, vc):
    """Per-row int32 [B,H,N] exponent e with sum_j |dS_ij| < 2^(e-3).

    |dS_ij| <= P_ij (|dP_ij - dP_ia| + |D'_i|) <= 4 P_ij |dO_i| max_j |V_j|, and
    2^(e-5) exceeds |dO_i| max|V_j|, so the deterministic kernel's int64 sums on
    the 2^(e-62) grid (2^(2e-68) for sum_j dS_ij^2) cannot overflow. Finite
    16-bit inputs give |e| < 300; the clamp keeps both grids normal FP64.
    """
    g = torch.linalg.vector_norm(dy, dim=-1, dtype=torch.float32).double()
    w = torch.linalg.vector_norm(vc, dim=-1, dtype=torch.float32).amax().double() if vc.numel() else 0.0
    bound = g * w
    e = torch.frexp(bound)[1].long() + 5
    e = torch.where(torch.isfinite(bound), e, torch.full_like(e, 300)).clamp(-300, 300)
    return e.to(torch.int32).transpose(1, 2).contiguous()


def _decode_fixed_statistics(stats, exponent, level):
    """Convert the deterministic kernel's int64 planes back to FP64 in place."""
    fixed = stats.view(torch.int64)
    rho = fixed[0].double() * _power_of_two(exponent.long() - 62)
    if level == 3:
        stats[2].copy_(fixed[2].double() * _power_of_two(2 * exponent.long() - 68))
    if level == 4:
        stats[1].copy_(fixed[1].double().mul_(2.0 ** -60))
        stats[3].copy_(fixed[3].double().mul_(2.0 ** -60))
    stats[0].copy_(rho)


def _prescale_fp16(dy, vc):
    """Exact power-of-two dO scaling for FP16 operands.

    Gradients are linear in dO, so dO * 2^e followed by output * 2^-e is
    exact. The scale keeps dS = P(dP - D), bounded by 2 max|dO_i| max|Vc_j|,
    near 2^13 so small entries stay out of FP16 subnormals; it only scales
    down when that bound risks FP16 overflow. BF16 shares FP32's exponent
    range and is returned unchanged. The scale stays on the device.
    """
    if dy.dtype != torch.float16 or not dy.numel() or not vc.numel():
        return dy, None
    g = torch.linalg.vector_norm(dy, dim=-1, dtype=torch.float32).amax()
    w = torch.linalg.vector_norm(vc, dim=-1, dtype=torch.float32).amax()
    bound = 2 * g * w
    e = torch.floor(torch.log2(_PRESCALE_TARGET / bound))
    # dO itself must stay finite in FP16.
    e = torch.minimum(e, torch.floor(torch.log2(30000.0 / dy.abs().amax().float())))
    e = torch.where(bound > 32768.0, e, e.clamp_min(0))
    e = torch.where(torch.isfinite(e), e, torch.zeros_like(e)).clamp(-60, 60).to(torch.int32)
    # Build 2^e and 2^-e exactly from FP32 exponent bits.
    factor = ((e + 127) << 23).view(torch.float32)
    # Scale in FP32: an FP16 product would round factors above 2^15 to inf.
    return (dy.float() * factor).to(dy.dtype), ((127 - e) << 23).view(torch.float32)


def _restore_scale_gradients(outputs, original_scale):
    dq, dk, dv = outputs
    if original_scale < 0:
        dq = -dq
    elif original_scale == 0:
        dq, dk = torch.zeros_like(dq), torch.zeros_like(dk)
    return dq, dk, dv


def _finish(outputs, dtype, previous, keep_fp32):
    dq, dk, dv = outputs
    if not keep_fp32:
        dq, dk, dv = (x.to(dtype) for x in (dq, dk, dv))
    if not previous:
        return dq, dk, dv, None, None
    return (dq, dk[:, previous:].contiguous(), dv[:, previous:].contiguous(),
            dk[:, :previous].contiguous(), dv[:, :previous].contiguous())


def _matched_backward(dy, q, k, v, qi, ki, level, chunk, scale, strict, deterministic):
    # Irregular independently aligned reset runs use the same exact packing
    # plan as the existing API. No N² mask is allocated. The common aligned
    # BOS path above remains entirely GPU-driven.
    from ._sca_segment_alignment import _build_plan
    plan = _build_plan(qi, ki)
    qp = q.flatten(0, 1).index_select(0, plan.q_indices)
    kp = k.flatten(0, 1).index_select(0, plan.k_indices)
    vp = v.flatten(0, 1).index_select(0, plan.k_indices)
    gp = dy.flatten(0, 1).index_select(0, plan.q_indices)
    oq, ok, ov = (torch.zeros_like(x, dtype=torch.float32) for x in (qp, kp, vp))
    cq, ck = plan.cu_q.cpu().tolist(), plan.cu_k.cpu().tolist()
    for index in range(len(cq) - 1):
        qa, qb, ka, kb = cq[index], cq[index + 1], ck[index], ck[index + 1]
        a, b, c, _, _ = backward(gp[qa:qb][None], qp[qa:qb][None], kp[ka:kb][None], vp[ka:kb][None],
                                level=level, visibility=2, span=chunk, scale=scale,
                                strict=strict, deterministic=deterministic, return_fp32=True)
        oq[qa:qb], ok[ka:kb], ov[ka:kb] = a[0], b[0], c[0]
    result = []
    for tensor, original, indices in ((oq, q, plan.q_indices), (ok, k, plan.k_indices), (ov, v, plan.k_indices)):
        output = torch.zeros_like(original, dtype=torch.float32)
        output.flatten(0, 1).index_copy_(0, indices, tensor)
        result.append(output)
    return tuple(result)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, prev_k, prev_v, visibility, options):
        # Call the existing standalone FWD so all routing and return semantics
        # remain shared with the public API. Only BWD algorithm selection differs.
        from .causal_flash_attn import causal_flash_attn_fwd
        from .sliding_window_attention import flash_swa_fwd
        from .sliding_chunk_attention import flash_sca_fwd
        functions = (causal_flash_attn_fwd, flash_swa_fwd, flash_sca_fwd)
        forward_options = dict(options)
        forward_options.pop('deterministic', None)
        if visibility:
            forward_options.update(prev_k=prev_k, prev_v=prev_v)
        y, saved, lse = functions[visibility](q, k, v, **forward_options)
        ctx.save_for_backward(q, k, v, prev_k, prev_v, saved, lse)
        ctx.visibility, ctx.options = visibility, options
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, prev_k, prev_v, saved, lse = ctx.saved_tensors
        opts = dict(ctx.options)
        span = opts.pop('window_size', opts.pop('chunk_size', None))
        level = opts.pop('high_precision_level')
        reset = opts.pop('reset_chunk_pos_per_seq', False)
        opts.pop('attn_method', None)
        grads = backward(dy, q, k, v, level=level, visibility=ctx.visibility, span=span,
                         prev_k=prev_k, prev_v=prev_v, reset=reset, **opts)
        return (*grads, None, None)


def apply(q, k, v, *, visibility, prev_k=None, prev_v=None, **options):
    # Fail before doing a forward if the installed binary predates this API.
    _ops()
    return _Attention.apply(q, k, v, prev_k, prev_v, visibility, options)

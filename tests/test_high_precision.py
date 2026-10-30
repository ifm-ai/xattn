"""Validate high_precision_level 0 and 1, with level 0 as the baseline.

Covers the public API contract of every attention family, the SBS forward,
and, for causal / window / chunk attention, FP64 numerical parity, padded
gradients, live previous KV, and precision-repair regressions. SoftDelta
numerics live in tests/softdelta.
"""
import inspect
import math

import pytest
import torch
import xattn
from tests._numerics import (
    Case, LEVELS, OPERATIONS, _allowed_masks, _attention_from_mask,
    assert_gradients, assert_output, cases_for, check_case,
    make_inputs, options_and_segments, oracle, require_device,
)

STANDARD_FAMILIES = ('causal_flash_attn', 'flash_sca', 'flash_swa')
FAMILIES = (
    *STANDARD_FAMILIES, 'softdelta_attention',
    'sliding_chunk_softdelta_attention', 'sliding_window_softdelta_attention',
)
ENTRIES = [getattr(xattn, name + suffix) for name in FAMILIES
           for suffix in ('', '_fwd', '_bwd')]
INVALID_LEVELS = ((-1, ValueError), (2, ValueError), (4, ValueError), (100, ValueError),
                  (False, TypeError), (True, TypeError), (1.0, TypeError),
                  ('1', TypeError), (None, TypeError))
ENTRY_LEVELS = [(fn, level) for fn in ENTRIES for level in LEVELS]


def _name(value):
    return getattr(value, '__name__', repr(value))


@pytest.mark.parametrize('fn,value,error',
                         [(fn, *invalid) for fn in ENTRIES for invalid in INVALID_LEVELS],
                         ids=_name)
def test_invalid_level_rejected_before_tensor_access(fn, value, error):
    signature = inspect.signature(fn)
    kwargs = {n: None for n, p in signature.parameters.items()
              if p.default is inspect.Parameter.empty}
    kwargs['high_precision_level'] = value
    with pytest.raises(error, match='high_precision_level'):
        fn(**kwargs)


@pytest.mark.parametrize('name', STANDARD_FAMILIES)
@pytest.mark.parametrize('level', (0, 1))
def test_standard_backward_rejects_fp32_output(name, level):
    device = require_device()
    q = torch.zeros(1, 8, 2, 64, device=device, dtype=torch.bfloat16)
    k = v = torch.zeros(1, 8, 1, 64, device=device, dtype=torch.bfloat16)
    y = torch.zeros(1, 8, 2, 64, device=device, dtype=torch.float32)
    lse = torch.zeros(1, 2, 8, device=device, dtype=torch.float32)
    span = {'flash_sca': dict(chunk_size=8), 'flash_swa': dict(window_size=8)}.get(name, {})
    with pytest.raises(ValueError, match='Q/K/V dtype'):
        getattr(xattn, name + '_bwd')(y.to(q.dtype), q, k, v, y, lse, **span,
                                      backend='sm90', high_precision_level=level)


@pytest.mark.parametrize('fn,level', ENTRY_LEVELS, ids=_name)
@pytest.mark.parametrize('dimensions', ((7, 16), (16, 13), (264, 16), (16, 264)))
def test_sm90_invalid_dimensions_rejected_before_launch(fn, level, dimensions):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    d, width = dimensions
    signature = inspect.signature(fn)
    kwargs = {n: None for n, p in signature.parameters.items()
              if p.default is inspect.Parameter.empty}
    kwargs.update(q=torch.empty(1, 1, 1, d, device='cuda', dtype=torch.bfloat16),
                  v=torch.empty(1, 1, 1, width, device='cuda', dtype=torch.bfloat16),
                  backend='sm90', high_precision_level=level)
    with pytest.raises(ValueError, match='multiples of 8'):
        fn(**kwargs)


@pytest.mark.parametrize('fn,level', ENTRY_LEVELS, ids=_name)
@pytest.mark.parametrize('scale', (float('inf'), float('-inf'), float('nan')))
def test_nonfinite_scale_rejected(fn, level, scale):
    kwargs = {n: None for n, p in inspect.signature(fn).parameters.items()
              if p.default is inspect.Parameter.empty}
    kwargs.update(high_precision_level=level, scale=scale)
    with pytest.raises(ValueError, match='scale must be finite'):
        fn(**kwargs)


@pytest.mark.parametrize("mechanism", ("full", "sca", "swa", "sda", "scsda", "swsda"))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_high_precision_single_key_state(mechanism, dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires FlashSCA SM90")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires SM90")

    # Large, exactly representable QK. With one visible key, the HP output
    # must equal V: SBS makes the unnormalized maximum term exactly exp2(0).
    # A pre-rounded scaled maximum followed by FMA need not have this property.
    q = torch.zeros(1, 1, 4, 128, device="cuda", dtype=dtype)
    if not xattn.flash_sca_sm90_available(q):
        pytest.skip("requires FlashSCA SM90 extension")
    k = torch.zeros(1, 1, 2, 128, device="cuda", dtype=dtype)
    q[..., 0] = 256
    k[..., 0] = 65536 if dtype == torch.bfloat16 else 4096
    v = torch.linspace(-1, 1, 256, device="cuda", dtype=torch.float32)
    v = v.reshape(1, 1, 2, 128).to(dtype)
    options = dict(backend="sm90", high_precision_level=1)
    functions = {
        "full": xattn.causal_flash_attn_fwd,
        "sca": xattn.flash_sca_fwd,
        "swa": xattn.flash_swa_fwd,
        "sda": xattn.softdelta_attention_fwd,
        "scsda": xattn.sliding_chunk_softdelta_attention_fwd,
        "swsda": xattn.sliding_window_softdelta_attention_fwd,
    }
    if mechanism in ("sca", "scsda"):
        options["chunk_size"] = 256
    if mechanism in ("swa", "swsda"):
        options["window_size"] = 256
    expected = v.float().repeat_interleave(2, dim=2)
    args = (q, k, v)
    if "sd" in mechanism:
        args = (q, k, None, v, torch.zeros(1, 1, 4, device="cuda", dtype=dtype))

    # Explicit standalone HP remains active even without an autograd graph.
    with torch.no_grad():
        result = functions[mechanism](*args, **options)
    output = result[0]
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, atol=0, rtol=0)
    if "sd" in mechanism:
        # The strict-past reader has no visible key.
        neg_read = result[1]
        assert len(result) == 3 and neg_read.dtype == torch.float32
        assert torch.count_nonzero(neg_read) == 0
    else:
        assert len(result) == 2


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('level', LEVELS)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float16))
@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('dimensions', ((40, 72), (136, 120), (152, 152), (56, 248)))
def test_padded_gqa_gradients(operation, dtype, deterministic, level, dimensions):
    device = require_device()
    case = Case('padded-gqa', 17, 33, *dimensions, span=16, batch=1)
    inputs = make_inputs(operation, case, dtype, 2, device)
    options, qi, ki = options_and_segments(operation, case, inputs, 'sm90')
    refs, expected = oracle(operation, case, inputs, qi, ki)
    dy = torch.randn_like(expected, dtype=dtype)
    reference = torch.autograd.grad(expected, refs, dy.double())
    name = OPERATIONS[operation]
    y, lse = getattr(xattn, name + '_fwd')(*inputs, **options, high_precision_level=level)
    gradients = getattr(xattn, name + '_bwd')(
        dy, *inputs, y, lse,
        **options, deterministic=deterministic, high_precision_level=level)[:len(inputs)]
    for actual, ref in zip(gradients, reference):
        assert actual.dtype == dtype and actual.shape == ref.shape
        assert torch.isfinite(actual).all()
        error = (actual.double() - ref).norm()
        assert error <= .015 * ref.norm() + 1e-6


@pytest.mark.parametrize('operation,reset,level', [
    (operation, reset, level)
    for operation, reset in (('window', False), ('chunk', False), ('chunk', True))
    for level in LEVELS
])
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float16))
@pytest.mark.parametrize('kv_heads', (1, 2, 4))
@pytest.mark.parametrize('deterministic', (False, True))
def test_live_bos_previous_gradients(operation, reset, dtype, kv_heads, deterministic, level):
    device = require_device()
    case = Case('live-bos-previous', 129, 129, 128, 96, metadata='bos',
                previous=64, span=64, reset=reset, batch=1)
    inputs = make_inputs(operation, case, dtype, kv_heads, device)
    options, qi, ki = options_and_segments(operation, case, inputs, 'sm90')
    refs, expected = oracle(operation, case, inputs, qi, ki)
    actual = getattr(xattn, OPERATIONS[operation])(
        *inputs[:3], **options,
        high_precision_level=level, deterministic=deterministic)
    dy = torch.randn_like(actual)
    reference = torch.autograd.grad(expected, refs, dy.double())
    gradients = torch.autograd.grad(actual, inputs, dy)
    for grad, ref in zip(gradients[-2:], reference[-2:]):
        assert ref.norm() > 0, 'fixture must exercise live previous KV'
        assert grad.dtype == dtype and grad.shape == ref.shape
        assert torch.isfinite(grad).all()
        assert (grad.double() - ref).norm() <= .02 * ref.norm() + 1e-6


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float16))
@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('case_name', ('dense-uneven', 'segment', 'bos', 'short-independent', 'empty-rows'))
def test_precision_kernel(operation, dtype, case_name, deterministic):
    case = next(c for c in cases_for(operation) if c.name == case_name)
    check_case(operation, case, dtype, 1, deterministic, 1)


def test_deterministic_long_segments():
    """Deterministic segment backward with more key tiles than resident CTAs."""
    case = Case('long-segment', 4096, 4096, metadata='segment', segment_length=2048, batch=4)
    check_case('full', case, torch.bfloat16, 4, True, 1)


@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float16))
@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('case_name', ('reset', 'short-reset', 'reset-long-q-run', 'chunk-wide'))
def test_precision_chunk_boundaries(dtype, case_name, deterministic):
    case = next(c for c in cases_for('chunk') if c.name == case_name)
    check_case('chunk', case, dtype, 1, deterministic, 1)


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('level', LEVELS)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float16))
@pytest.mark.parametrize('scale_sign', (0, -1))
def test_precision_nonpositive_scale(operation, dtype, level, scale_sign):
    case = Case('scale-edge', 65, 65, 64, 64, span=32, batch=1)
    inputs = make_inputs(operation, case, dtype, 2, require_device())
    options, qi, ki = options_and_segments(operation, case, inputs, 'sm90')
    scale = scale_sign * case.dim ** -0.5
    options['scale'] = scale
    refs = [x.detach().double().requires_grad_() for x in inputs]
    masks = _allowed_masks(65, 65, 1, inputs[0].device, operation, 32, qi, ki, False)
    expected = _attention_from_mask(*refs, masks[0], scale)
    fn = OPERATIONS[operation]
    y = getattr(xattn, fn)(*inputs, **options, high_precision_level=level)
    assert_output(y, expected, dtype)
    dy = torch.randn_like(y) / y.numel() ** 0.5
    reference = torch.autograd.grad(expected, refs, dy.double())
    assert_gradients(torch.autograd.grad(y, inputs, dy), reference, dtype)
    explicit, lse = getattr(xattn, fn + '_fwd')(*inputs, **options, high_precision_level=level)
    assert_output(explicit, expected, dtype)
    gradients = getattr(xattn, fn + '_bwd')(
        dy, *inputs, explicit, lse, **options, high_precision_level=level)[:len(inputs)]
    assert_gradients(gradients, reference, dtype)
    if scale == 0:
        assert torch.count_nonzero(gradients[0]) == 0
        assert torch.count_nonzero(gradients[1]) == 0


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float16))
@pytest.mark.parametrize('deterministic', (False, True))
def test_precision_constant_key_support(operation, dtype, deterministic):
    device = require_device()
    torch.manual_seed(927)
    n, d = 384, 64
    q = torch.zeros(1, n, 4, d, device=device, dtype=dtype)
    q[..., 0] = 1
    k = torch.zeros(1, n, 2, d, device=device, dtype=dtype)
    if operation == 'full':
        k[..., 0] = 800  # Every supported K is identical: dQ is exactly zero.
        start = 0
    else:
        k[:, 0, :, 0] = 800
        start = 256  # Outlier has left the SCA/SWA support completely.
    v = (torch.randn(1, n, 2, d, device=device) * (1e30 if dtype == torch.bfloat16 else 1024)).to(dtype)
    q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
    inputs = [q, k, v]
    options = dict(backend='sm90', high_precision_level=1, deterministic=deterministic)
    if operation == 'chunk':
        options['chunk_size'] = 128
    elif operation == 'window':
        options['window_size'] = 127
    out = getattr(xattn, OPERATIONS[operation])(*inputs, **options)
    gradients = torch.autograd.grad(out, inputs, torch.ones_like(out))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert gradients[0].dtype == dtype
    assert torch.count_nonzero(gradients[0][:, start:]) == 0


@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('operation', ('full', 'chunk'))
def test_precision_tiny_probability_large_value(deterministic, operation):
    """A tiny P times a huge dP is representable and must not vanish early."""
    device = require_device()
    torch.manual_seed(31415)
    n, d = 257, 64
    q = torch.zeros(1, n, 2, d, device=device, dtype=torch.bfloat16)
    k = torch.zeros(1, n, 1, d, device=device, dtype=torch.bfloat16)
    q[..., 0] = 1
    k[:, 0, :, 0] = 100 * d ** .5
    v = (torch.randn_like(k, dtype=torch.float32) * 1e35).to(k.dtype)
    g = torch.randn_like(q)
    Q, K, V, G = [t.double().transpose(1, 2) for t in (q, k, v, g)]
    K, V = K.repeat_interleave(2, 1), V.repeat_interleave(2, 1)
    i = torch.arange(n, device=device)
    mask = i[None, :] <= i[:, None]
    if operation == 'chunk':
        mask &= i[None, :] >= ((i[:, None] // 128 - 1) * 128).clamp_min(0)
    p = (Q @ K.transpose(-1, -2) * d ** -.5).masked_fill(~mask, -torch.inf).softmax(-1)
    dp = G @ V.transpose(-1, -2)
    centered = dp - dp.gather(-1, p.argmax(-1, keepdim=True))
    ds = p * (centered - (p * centered).sum(-1, keepdim=True))
    refs = ((ds @ K * d ** -.5).transpose(1, 2),
            (ds.transpose(-1, -2) @ Q * d ** -.5).sum(1, keepdim=True).transpose(1, 2),
            (p.transpose(-1, -2) @ G).sum(1, keepdim=True).transpose(1, 2))
    inputs = [t.requires_grad_() for t in (q, k, v)]
    options = dict(chunk_size=128) if operation == 'chunk' else {}
    y = getattr(xattn, OPERATIONS[operation])(
        *inputs, **options, backend='sm90', deterministic=deterministic, high_precision_level=1)
    for actual, ref in zip(torch.autograd.grad(y, inputs, g), refs):
        assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
        assert (actual.double() - ref).norm() <= .01 * ref.norm()


@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('operation,reset', (
    ('full', False), ('window', False), ('chunk', False), ('chunk', True)))
def test_precision_segment_constant_keys(deterministic, operation, reset):
    device = require_device()
    torch.manual_seed(927)
    n, d = 257, 64
    q = torch.ones(1, n, 2, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, n, 1, d, device=device, dtype=q.dtype)
    k[:, :128] = 1
    v = (torch.randn_like(k, dtype=torch.float32) * 1e30).to(q.dtype)
    g = torch.randn_like(q)
    ids = (torch.arange(n, device=device) // 128)[None]
    options = {'window': dict(window_size=64),
               'chunk': dict(chunk_size=64, reset_chunk_pos_per_seq=reset)}.get(operation, {})
    y = getattr(xattn, OPERATIONS[operation])(
        q, k, v, **options, segment_idx=ids, backend='sm90',
        deterministic=deterministic, high_precision_level=1)
    dq, = torch.autograd.grad(y, q, g)
    assert torch.count_nonzero(dq) == 0


def _llm_like_inputs(operation, case, dtype, kv_heads, device):
    """Shared query direction and a sink key at position 0 (attention sink rows)."""
    inputs = make_inputs(operation, case, dtype, kv_heads, device)
    gen = torch.Generator(device=device).manual_seed(7)
    q, keys = inputs[0], [inputs[1]] + ([inputs[-2]] if case.previous else [])
    d = q.shape[-1]
    u = torch.nn.functional.normalize(torch.randn(d, device=device, generator=gen), dim=-1)
    amp = math.sqrt(15.0 * math.sqrt(d))
    with torch.no_grad():
        q.copy_((torch.nn.functional.normalize(
            torch.randn(q.shape, device=device, generator=gen) / math.sqrt(d) + 0.5 * u, dim=-1) * amp).to(dtype))
        for t in keys:
            t.copy_((torch.nn.functional.normalize(torch.randn(t.shape, device=device, generator=gen), dim=-1) * amp).to(dtype))
        keys[-1][:, 0] = (u * amp).to(dtype)
    return inputs


@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float16))
def test_deterministic_sink_anchor_repair(dtype):
    """Thousands of rows anchored at one key span many repair chunks; the ordered
    reduction stays bitwise repeatable and matches the unordered path."""
    device = require_device()
    case = Case('sink-anchor', 700, 700, 64, 64, batch=1)
    inputs = _llm_like_inputs('full', case, dtype, 1, device)
    options, _, _ = options_and_segments('full', case, inputs, 'sm90')
    dy = torch.randn(1, 700, 4, 64, device=device, dtype=dtype) / 700 ** 0.5
    results = {}
    for deterministic in (True, True, False):
        out = xattn.causal_flash_attn(*inputs, **options, deterministic=deterministic, high_precision_level=1)
        results.setdefault(deterministic, []).append(torch.autograd.grad(out, inputs, dy))
    for first, second in zip(*results[True]):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    for ordered, unordered in zip(results[True][0], results[False][0]):
        assert (ordered.double() - unordered.double()).norm() <= 2e-3 * unordered.double().norm()

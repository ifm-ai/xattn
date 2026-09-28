"""Shared numerical cases; test collection stays in the attention-family files."""
from dataclasses import dataclass, replace
import hashlib

import pytest
import torch
import torch.nn.functional as F
import xattn

from tests.softdelta.reference import _allowed_masks, _attention_from_mask
from tests.softdelta._inputs import ATOL, GRAD_ATOL, RTOL

DTYPES = (torch.float16, torch.bfloat16)
HEAD_DIMS = (32, 64, 96, 128, 160, 192, 256)
OPERATIONS = {
    'full': 'causal_flash_attn', 'window': 'flash_swa', 'chunk': 'flash_sca',
    'softdelta_full': 'softdelta_attention',
    'softdelta_window': 'sliding_window_softdelta_attention',
    'softdelta_chunk': 'sliding_chunk_softdelta_attention',
}


@dataclass(frozen=True)
class Case:
    name: str
    q_length: int = 193
    k_length: int = 193
    dim: int = 64
    value_dim: int = 64
    metadata: str = 'dense'
    previous: int = 0
    span: int = 64
    reset: bool = False
    groups: int = 4
    layout: str = 'contiguous'
    batch: int = 2


# Curated combinations replace three overlapping Cartesian short-Q matrices.
CASES = (
    Case('singleton', 1, 1, 32, 64, batch=1),
    Case('dense-uneven', dim=40, value_dim=56),
    Case('segment', dim=96, value_dim=128, metadata='segment'),
    Case('bos', dim=160, value_dim=192, metadata='bos'),
    Case('decode', 1, 257, 32, 64, batch=1),
    Case('short-paired', 17, 41, 64, 192, metadata='paired'),
    Case('short-independent', 129, 257, 128, 192, metadata='independent'),
    Case('short-segment', 255, 513, 160, 160, metadata='segment'),
    Case('short-bos', 65, 193, 192, 160, metadata='bos'),
    Case('empty-rows', 17, 41, 64, 64, metadata='unmatched'),
    Case('future-only', 4, 9, 32, 64, metadata='future'),
    Case('head-view', layout='head_slice'),
    Case('offset-view', layout='offset'),
    Case('strided-view', layout='last_stride'),
    Case('broadcast-view', layout='broadcast'),
)


def cases_for(operation):
    visibility = operation.removeprefix('softdelta_')
    cases = list(CASES)
    if operation != 'full':
        cases += [
            Case('previous', previous=65, dim=128, value_dim=104),
            Case('short-previous', 17, 41, 64, 192, previous=65, metadata='paired'),
            Case('bos-previous', 65, 193, 192, 96, previous=128, metadata='bos'),
        ]
    if visibility == 'window':
        cases += [Case('window-zero', span=0), Case('window-one', span=1),
                  Case('covering-window', span=512, previous=37, groups=2)]
    if visibility == 'chunk':
        cases += [
            Case('chunk-one', span=1, previous=1),
            Case('chunk-unaligned', 513, 513, 192, 96, previous=255, span=255),
            Case('chunk-wide', 517, 517, 256, 160, previous=257, span=257),
            Case('reset', metadata='segment', reset=True),
            Case('short-reset', 17, 41, 64, 192, metadata='independent', reset=True),
            Case('reset-previous', 65, 193, 192, 160, metadata='bos', previous=128, reset=True),
            Case('reset-unmatched', 17, 41, metadata='unmatched', reset=True),
            Case('reset-long-q-run', 17, 41, metadata='long_run', reset=True),
        ]
    if operation.startswith('softdelta_'):
        cases += [Case(f'gate-groups-{g}', 127, 127, 256, 256, groups=g, batch=1)
                  for g in (1, 2, 8)]
    if visibility == 'chunk':
        cases = [replace(case, span=case.previous) if case.previous else case for case in cases]
    return tuple(cases)


def require_device(backend='sm90'):
    if backend == 'torch':
        return 'cpu'
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('xattn_cuda')
    capability = torch.cuda.get_device_capability()
    if backend == 'sm90' and capability != (9, 0):
        pytest.skip('SM90 required')
    if backend == 'cuda' and capability not in ((8, 0), (8, 6), (8, 9)):
        pytest.skip('generic CUDA coverage requires SM80/SM86/SM89')
    return 'cuda'


def case_seed(operation, case, dtype, kv_heads):
    return int.from_bytes(hashlib.sha256(f'{operation}:{case}:{dtype}:{kv_heads}'.encode()).digest()[:4], 'little')


def make_inputs(operation, case, dtype, kv_heads, device):
    soft = operation.startswith('softdelta_')
    heads = 4
    seed = case_seed(operation, case, dtype, kv_heads)
    torch.manual_seed(seed)
    def rand(shape):
        return torch.randn(shape, device=device, dtype=dtype)
    q_shape = (case.batch, case.q_length, heads * (2 if soft else 1), case.dim)
    if case.layout == 'head_slice':
        q = F.normalize(rand((*q_shape[:2], q_shape[2] * 2, case.dim)), dim=-1)[:, :, 1::2]
    elif case.layout == 'offset':
        q = F.normalize(rand((*q_shape[:-1], case.dim + 1)), dim=-1)[..., 1:]
    elif case.layout == 'last_stride':
        q = F.normalize(rand((*q_shape[:-1], case.dim * 2)), dim=-1)[..., ::2]
    elif case.layout == 'broadcast':
        q = F.normalize(rand((*q_shape[:2], 1, case.dim)), dim=-1).expand(q_shape)
    else:
        q = F.normalize(rand(q_shape), dim=-1)
    total = case.k_length + case.previous
    kt = F.normalize(rand((case.batch, total, kv_heads, case.dim)), dim=-1)
    v_shape = (case.groups, case.value_dim // case.groups) if soft else (case.value_dim,)
    vt = F.silu(rand((case.batch, total, kv_heads, *v_shape))) + 0.1
    inputs = [q, kt[:, case.previous:], vt[:, case.previous:]]
    if soft:
        inputs.append(rand((case.batch, case.q_length, heads, case.groups, 1)) * 0.2)
    if case.previous:
        inputs += [kt[:, :case.previous], vt[:, :case.previous]]
    return [t.detach().requires_grad_() for t in inputs]


def options_and_segments(operation, case, inputs, backend):
    visibility = operation.removeprefix('softdelta_')
    options = {'backend': backend}
    if visibility == 'window':
        options['window_size'] = case.span
    elif visibility == 'chunk':
        options.update(chunk_size=case.span, reset_chunk_pos_per_seq=case.reset)
    if case.previous:
        options.update(prev_k=inputs[-2], prev_v=inputs[-1])
    qi = ki = None
    if case.metadata != 'dense':
        total = case.k_length + case.previous
        ki = (torch.arange(total, device=inputs[0].device) // 67).expand(case.batch, -1).clone()
        ki += torch.arange(case.batch, device=ki.device)[:, None] * 100
        qi = ki[:, -case.q_length:].clone()
        if case.metadata == 'independent':
            qi[:, :min(max(1, case.q_length // 3), 33)] = ki[:, :1]
        elif case.metadata == 'unmatched':
            qi.fill_(-1)
        elif case.metadata == 'long_run':
            qi = ki[:, :1].expand(case.batch, case.q_length).clone()
        elif case.metadata == 'future':
            ki.zero_(); ki[:, -3:] = 1; qi.fill_(1)
        if case.metadata == 'segment':
            options['segment_idx'] = ki
        elif case.metadata == 'bos':
            bos = torch.ones_like(ki, dtype=torch.bool)
            bos[:, 1:] = ki[:, 1:] != ki[:, :-1]
            options['bos_mask'] = bos
        else:
            options.update(q_segment_idx=qi, k_segment_idx=ki)
    return options, qi, ki


def oracle(operation, case, inputs, qi, ki):
    refs = [x.detach().double().requires_grad_() for x in inputs]
    soft = operation.startswith('softdelta_')
    q, k, v = refs[:3]
    if case.previous:
        k, v = torch.cat((refs[-2], k), 1), torch.cat((refs[-1], v), 1)
    masks = _allowed_masks(case.q_length, k.shape[1], case.batch, q.device,
                           operation.removeprefix('softdelta_'), case.span, qi, ki, case.reset)
    if not soft:
        expected = _attention_from_mask(q, k, v, masks[0], case.dim ** -0.5)
        return refs, expected, None
    reads = [_attention_from_mask(q[:, :, branch::2], k, v.flatten(3), masks[branch],
                                  case.dim ** -0.5).reshape(case.batch, case.q_length, 4,
                                                          case.groups, case.value_dim // case.groups)
             for branch in (0, 1)]
    return refs, reads[0] - refs[3].sigmoid() * reads[1], reads


def assert_output(actual, expected, dtype, reads=None, gate=None):
    assert torch.isfinite(actual).all()
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    if reads is None:
        torch.testing.assert_close(actual, expected.to(dtype), atol=ATOL[dtype], rtol=RTOL[dtype])
    else:
        budget = (ATOL[dtype] + RTOL[dtype] * reads[0].abs()
                  + gate.double().sigmoid() * (ATOL[dtype] + RTOL[dtype] * reads[1].abs())
                  + ATOL[dtype] + RTOL[dtype] * expected.abs())
        assert ((actual.double() - expected).abs() <= budget).all(), 'SoftDelta exceeds component rounding budget'


def assert_gradients(actual, expected, dtype):
    assert len(actual) == len(expected)
    for grad, ref in zip(actual, expected):
        assert grad is not None and torch.isfinite(grad).all()
        torch.testing.assert_close(grad, ref.to(dtype), atol=GRAD_ATOL[dtype], rtol=RTOL[dtype])


def check_case(operation, case, dtype, kv_heads, deterministic, high_precision, backend='sm90'):
    device = require_device(backend)
    inputs = make_inputs(operation, case, dtype, kv_heads, device)
    options, qi, ki = options_and_segments(operation, case, inputs, backend)
    refs, expected, reads = oracle(operation, case, inputs, qi, ki)
    soft = operation.startswith('softdelta_')
    base_inputs = inputs[:4 if soft else 3]
    gate = inputs[3] if soft else None
    fn = getattr(xattn, OPERATIONS[operation])
    actual = fn(*base_inputs, **options, deterministic=deterministic, high_precision_output=high_precision)
    assert_output(actual, expected, dtype, reads, gate)
    dy = torch.randn_like(actual) / actual.numel() ** 0.5
    expected_grads = torch.autograd.grad(expected, refs, dy.double())
    grads = torch.autograd.grad(actual, inputs, dy, retain_graph=deterministic)
    assert_gradients(grads, expected_grads, dtype)
    if deterministic:
        repeated = torch.autograd.grad(actual, inputs, dy)
        for first, second in zip(grads, repeated):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
    if case.name == 'full-dimensions' and operation in ('softdelta_window', 'softdelta_chunk'):
        from xattn.ops._flash_softdelta import flash_softdelta_paired_training
        paired = flash_softdelta_paired_training(
            *base_inputs, case.span, case.dim**-0.5,
            1 if operation == 'softdelta_window' else 2, deterministic, high_precision,
        )
        assert_output(paired, expected, dtype, reads, gate)
        assert_gradients(torch.autograd.grad(paired, inputs, dy), expected_grads, dtype)
    if backend != 'torch':
        fwd = getattr(xattn, OPERATIONS[operation] + '_fwd')
        bwd = getattr(xattn, OPERATIONS[operation] + '_bwd')
        fwd_options = dict(options, high_precision_output=high_precision)
        if soft:
            fwd_options['deterministic'] = deterministic
        with torch.no_grad():
            explicit, state, lse = fwd(*base_inputs, **fwd_options)
            assert_output(explicit, expected, dtype, reads, gate)
            saved = state if soft or high_precision else explicit
            if high_precision:
                assert state.dtype == torch.float32
            explicit_grads = bwd(dy, *base_inputs, saved, lse, **options,
                                 deterministic=deterministic)[:len(inputs)]
            assert_gradients(explicit_grads, expected_grads, dtype)
        with torch.inference_mode():
            inference = fn(*base_inputs, **options, high_precision_output=high_precision)
            assert_output(inference, expected, dtype, reads, gate)


def check_rounding_bias(operation, deterministic):
    require_device()
    q = torch.zeros(1, 128, 1, 64, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k = torch.ones_like(q, requires_grad=True)
    values = torch.full_like(q, -1.0078125)
    values[:, :32] = -1.0
    fn = getattr(xattn, OPERATIONS[operation])
    options = {'chunk_size': 64} if operation == 'chunk' else {'window_size': 127}
    dy = torch.zeros_like(q); dy[:, -1] = -1
    cumulative = torch.zeros_like(q, dtype=torch.float64)
    for step in range(4):
        v = values.roll(step, 1).detach().requires_grad_()
        y = fn(q, k, v, **options, backend='sm90', deterministic=deterministic,
               high_precision_output=True)
        # Identical K rows imply dQ=0 exactly, independently of V and Q.
        dq = torch.autograd.grad(y, q, dy)[0]
        assert y.dtype == torch.bfloat16
        assert dq.abs().max() <= 1e-4
        cumulative += dq.double()
    assert cumulative.abs().max() <= 1e-3


def check_packed(operation, dtype, kv_heads, deterministic):
    require_device()
    torch.manual_seed(513)
    q_lengths, k_lengths = (0, 1, 17, 65), (0, 5, 41, 129)
    q = F.normalize(torch.randn(sum(q_lengths), 4, 64, device='cuda', dtype=dtype), dim=-1).requires_grad_()
    k = F.normalize(torch.randn(sum(k_lengths), kv_heads, 64, device='cuda', dtype=dtype), dim=-1).requires_grad_()
    v = (F.silu(torch.randn(sum(k_lengths), kv_heads, 96, device='cuda', dtype=dtype)) + 0.1).requires_grad_()
    cu_q = torch.tensor((0, *torch.tensor(q_lengths).cumsum(0).tolist()), device='cuda', dtype=torch.int32)
    cu_k = torch.tensor((0, *torch.tensor(k_lengths).cumsum(0).tolist()), device='cuda', dtype=torch.int32)
    refs = [t.detach().double().requires_grad_() for t in (q, k, v)]
    parts = []
    qs = ks = 0
    for lq, lk in zip(q_lengths, k_lengths):
        if lq:
            mask, _ = _allowed_masks(lq, lk, 1, q.device, operation, 32, None, None, False)
            parts.append(_attention_from_mask(refs[0][None, qs:qs+lq], refs[1][None, ks:ks+lk],
                                             refs[2][None, ks:ks+lk], mask, 64**-0.5).squeeze(0))
        qs += lq; ks += lk
    expected = torch.cat(parts)
    options = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max(q_lengths),
                   max_seqlen_k=max(k_lengths), deterministic=deterministic, backend='sm90')
    if operation == 'chunk':
        from xattn.ops.sliding_chunk_attention import _flash_sca_varlen
        actual = _flash_sca_varlen(q, k, v, 32, reset_chunk_pos_per_seq=True, **options)
    else:
        from xattn.ops.sliding_window_attention import _flash_swa_varlen
        actual = _flash_swa_varlen(q, k, v, 32, **options)
    assert_output(actual, expected, dtype)
    dy = torch.randn_like(actual) / actual.numel()**0.5
    reference = torch.autograd.grad(expected, refs, dy.double())
    grads = torch.autograd.grad(actual, (q, k, v), dy, retain_graph=deterministic)
    assert_gradients(grads, reference, dtype)
    if deterministic:
        for a, b in zip(grads, torch.autograd.grad(actual, (q, k, v), dy)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


# Every D/V bucket uses these lengths; large cases require both opt-in flags.
MATRIX_LENGTHS = (7, 193, 513, *(
    pytest.param(length, marks=pytest.mark.long)
    for length in (4096, 16384, 32768, 65536)
))


def check_dimensions(operation, length, dim, value_dim, dtype, kv_heads,
                     deterministic, high_precision):
    if length <= 513:
        check_case(operation, Case('full-dimensions', length, length, dim, value_dim,
                                   span=3 if length == 7 else 255, batch=1),
                   dtype, kv_heads, deterministic, high_precision)
    else:
        check_long(operation, length, dim, value_dim, dtype, kv_heads,
                   high_precision, deterministic=deterministic)


def check_long(operation, length, dim, value_dim, dtype, kv_heads, high_precision, deterministic=False):
    """Sample rows against all keys; sparse dO also checks full-sized gradients."""
    require_device()
    case = Case('long', length, length, dim, value_dim, span=255, batch=1)
    inputs = make_inputs(operation, case, dtype, kv_heads, 'cuda')
    refs = [t.detach().double().requires_grad_() for t in inputs]
    positions = torch.tensor(sorted({0, 1, 127, 128, 191, 192, 255, 256, 257,
                                    511, 512, length//2, length-2, length-1}), device='cuda')
    qp = positions[:, None]; kp = torch.arange(length, device='cuda')[None, :]
    visibility = operation.removeprefix('softdelta_')
    left = torch.ones_like(kp, dtype=torch.bool)
    if visibility == 'window':
        left = kp >= qp - case.span
    elif visibility == 'chunk':
        left = kp >= qp // case.span * case.span - case.span
    soft = operation.startswith('softdelta_')
    reads = []
    for branch in range(2 if soft else 1):
        qr = refs[0][:, positions]
        if soft:
            qr = qr[:, :, branch::2]
        mask = (left & (kp <= qp - branch))[None]
        read = _attention_from_mask(qr, refs[1], refs[2].flatten(3) if soft else refs[2], mask, dim**-0.5)
        reads.append(read.reshape(1, len(positions), 4, case.groups, value_dim//case.groups) if soft else read)
    gate = refs[3][:, positions] if soft else None
    expected = reads[0] - gate.sigmoid()*reads[1] if soft else reads[0]
    options, _, _ = options_and_segments(operation, case, inputs, 'sm90')
    actual = getattr(xattn, OPERATIONS[operation])(
        *inputs, **options, high_precision_output=high_precision, deterministic=deterministic)
    assert torch.isfinite(actual).all()
    selected = actual[:, positions]
    assert_output(selected, expected, dtype, reads if soft else None, gate)
    dy = torch.randn_like(selected) / selected.numel()**0.5
    grads = torch.autograd.grad(selected, inputs, dy, retain_graph=deterministic)
    assert_gradients(grads, torch.autograd.grad(expected, refs, dy.double()), dtype)
    if deterministic:
        for actual_grad, repeated in zip(grads, torch.autograd.grad(selected, inputs, dy)):
            torch.testing.assert_close(actual_grad, repeated, atol=0, rtol=0)

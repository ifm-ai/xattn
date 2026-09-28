import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from xattn import flash_swa
from xattn.ops.sliding_window_attention import _flash_swa_varlen


ATOL = {
    torch.bfloat16: 1e-3,
    torch.float16: 2e-4,
}
GRAD_ATOL = {
    torch.bfloat16: 2e-3,
    torch.float16: 2e-4,
}
RTOL = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
}


def _requires_sm90_extension():
    if importlib.util.find_spec("xattn_cuda") is None:
        pytest.skip("xattn CUDA extension is not built")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("segmented FlashSWA currently requires SM90")


def segmented_swa_fp64_oracle(
    q,
    k,
    v,
    window_size,
    scale,
    prev_k=None,
    prev_v=None,
    q_segment_idx=None,
    k_segment_idx=None,
):
    """FP64 oracle over the already-quantized input tensors."""
    assert (prev_k is None) == (prev_v is None)
    assert (q_segment_idx is None) == (k_segment_idx is None)
    if prev_k is not None:
        k = torch.cat([prev_k, k], dim=1)
        v = torch.cat([prev_v, v], dim=1)

    batch, q_length, q_heads, _ = q.shape
    k_length = k.shape[1]
    kv_heads = k.shape[2]
    assert q_heads % kv_heads == 0
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)

    q_position = torch.arange(q_length, device=q.device)
    k_position = torch.arange(k_length, device=q.device)
    q_aligned = q_position + k_length - q_length
    allowed = (
        k_position.unsqueeze(0) <= q_aligned.unsqueeze(1)
    ) & (
        k_position.unsqueeze(0)
        >= q_aligned.unsqueeze(1) - window_size
    )
    allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
    if q_segment_idx is not None:
        allowed = allowed & (
            q_segment_idx.unsqueeze(2) == k_segment_idx.unsqueeze(1)
        )
    assert bool(allowed.any(dim=-1).all())

    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhv->bqhv", probabilities, v)


def packed_varlen_swa_fp64_oracle(
    q,
    k,
    v,
    q_lengths,
    k_lengths,
    window_size,
    scale,
):
    """FP64 oracle with each packed Q/K pair treated as one segment."""
    outputs = []
    q_start = 0
    k_start = 0
    for q_length, k_length in zip(q_lengths, k_lengths):
        q_segment = q[q_start : q_start + q_length]
        k_segment = k[k_start : k_start + k_length]
        v_segment = v[k_start : k_start + k_length]
        q_heads = q_segment.shape[1]
        kv_heads = k_segment.shape[1]
        if q_heads != kv_heads:
            repeats = q_heads // kv_heads
            k_segment = k_segment.repeat_interleave(repeats, dim=1)
            v_segment = v_segment.repeat_interleave(repeats, dim=1)

        q_position = torch.arange(q_length, device=q.device)
        k_position = torch.arange(k_length, device=q.device)
        q_aligned = q_position + k_length - q_length
        allowed = (
            k_position.unsqueeze(0) <= q_aligned.unsqueeze(1)
        ) & (
            k_position.unsqueeze(0)
            >= q_aligned.unsqueeze(1) - window_size
        )
        scores = (
            torch.einsum("qhd,khd->hqk", q_segment, k_segment)
            * scale
        )
        scores = scores.masked_fill(
            ~allowed.unsqueeze(0), float("-inf")
        )
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum(
                "hqk,khv->qhv", probabilities, v_segment
            )
        )
        q_start += q_length
        k_start += k_length
    assert q_start == q.shape[0]
    assert k_start == k.shape[0]
    return torch.cat(outputs, dim=0)


def _make_segment_idx(batch, total_length, device):
    positions = torch.arange(total_length, device=device)
    boundaries = torch.tensor(
        [0, 19, 63, 130, 190],
        device=device,
    )
    boundaries = boundaries[boundaries < total_length]
    segment_idx = torch.bucketize(positions, boundaries, right=True)
    return segment_idx.unsqueeze(0).expand(batch, -1).contiguous()


def _condition_qkv(q, k, v):
    with torch.no_grad():
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), F.silu(v) + 0.1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "qk_dim,value_dim",
    [
        (96, 160),
        (96, 256),
        (160, 96),
        (160, 160),
        (160, 256),
        (256, 160),
    ],
)
@pytest.mark.parametrize(
    "route,deterministic",
    [
        ("dense", False),
        ("segment", True),
        ("varlen", False),
        ("varlen", True),
    ],
)
def test_flash_swa_extended_instances_match_fp64_oracle(
    dtype, qk_dim, value_dim, route, deterministic
):
    _requires_sm90_extension()
    torch.manual_seed(20260802)
    heads = 2
    window_size = 17
    scale = qk_dim**-0.5

    if route == "varlen":
        q_lengths = [17, 23]
        k_lengths = [25, 23]
        q = torch.randn(
            sum(q_lengths), heads, qk_dim, dtype=dtype, device="cuda"
        )
        k = torch.randn(
            sum(k_lengths), heads, qk_dim, dtype=dtype, device="cuda"
        )
        v = torch.randn(
            sum(k_lengths), heads, value_dim, dtype=dtype, device="cuda"
        )
        with torch.no_grad():
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            v = F.silu(v) + 0.1
        cu_q = torch.tensor(
            [0, 17, 40], dtype=torch.int32, device="cuda"
        )
        cu_k = torch.tensor(
            [0, 25, 48], dtype=torch.int32, device="cuda"
        )
        reference_tensors = [
            tensor.detach().double().requires_grad_(True)
            for tensor in (q, k, v)
        ]
        expected = packed_varlen_swa_fp64_oracle(
            *reference_tensors,
            q_lengths,
            k_lengths,
            window_size,
            scale,
        )
        actual_tensors = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (q, k, v)
        ]
        actual = _flash_swa_varlen(
            *actual_tensors,
            window_size,
            scale,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
            deterministic=deterministic,
            backend="sm90",
        )
    else:
        batch, length = 1, 41
        q = torch.randn(
            batch, length, heads, qk_dim, dtype=dtype, device="cuda"
        )
        k = torch.randn(
            batch, length, heads, qk_dim, dtype=dtype, device="cuda"
        )
        v = torch.randn(
            batch, length, heads, value_dim, dtype=dtype, device="cuda"
        )
        q, k, v = _condition_qkv(q, k, v)
        segment_idx = (
            _make_segment_idx(batch, length, q.device)
            if route == "segment"
            else None
        )
        reference_tensors = [
            tensor.detach().double().requires_grad_(True)
            for tensor in (q, k, v)
        ]
        expected = segmented_swa_fp64_oracle(
            *reference_tensors,
            window_size,
            scale,
            q_segment_idx=segment_idx,
            k_segment_idx=segment_idx,
        )
        actual_tensors = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (q, k, v)
        ]
        actual = flash_swa(
            *actual_tensors,
            window_size,
            scale,
            segment_idx=segment_idx,
            deterministic=deterministic,
            backend="sm90",
        )

    torch.testing.assert_close(
        actual,
        expected.to(dtype),
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
    )
    token_count = actual.shape[0] if actual.dim() == 3 else actual.shape[1]
    weight = torch.randn_like(actual) / math.sqrt(token_count)
    expected.backward(weight.double())
    actual.backward(weight)
    for actual_tensor, expected_tensor in zip(
        actual_tensors, reference_tensors
    ):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad.to(dtype),
            rtol=RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )


def test_flash_swa_high_precision_varlen_matches_fp64_oracle():
    _requires_sm90_extension()
    torch.manual_seed(20260827)
    dtype = torch.bfloat16
    q_lengths = [17, 23]
    k_lengths = [25, 23]
    q_heads, kv_heads, qk_dim, value_dim = 4, 2, 64, 64
    window_size = 17
    scale = qk_dim**-0.5
    q = torch.randn(
        sum(q_lengths), q_heads, qk_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        sum(k_lengths), kv_heads, qk_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        sum(k_lengths), kv_heads, value_dim, dtype=dtype, device="cuda"
    )
    q, k, v = _condition_qkv(q, k, v)
    cu_q = torch.tensor([0, 17, 40], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 25, 48], dtype=torch.int32, device="cuda")
    expected_inputs = [
        tensor.detach().double().requires_grad_(True)
        for tensor in (q, k, v)
    ]
    actual_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in (q, k, v)
    ]
    expected = packed_varlen_swa_fp64_oracle(
        *expected_inputs,
        q_lengths,
        k_lengths,
        window_size,
        scale,
    )
    actual = _flash_swa_varlen(
        *actual_inputs,
        window_size,
        scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        backend="sm90",
        deterministic=True,
        high_precision_output=True,
    )
    torch.testing.assert_close(
        actual,
        expected.to(dtype),
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
    )

    weight = torch.randn_like(actual) / math.sqrt(actual.shape[0])
    actual_grads = torch.autograd.grad(actual, actual_inputs, weight)
    expected_grads = torch.autograd.grad(
        expected, expected_inputs, weight.double()
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad,
            expected_grad.to(dtype),
            rtol=RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "q_heads,kv_heads",
    [(2, 2), (4, 2), (4, 1)],
    ids=["mha", "gqa", "mqa"],
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_flash_swa_varlen_segments_match_fp64_oracle(
    dtype,
    q_heads,
    kv_heads,
    deterministic,
):
    _requires_sm90_extension()
    torch.manual_seed(20260732)
    q_lengths = [41, 67, 23]
    k_lengths = [58, 67, 51]
    qk_dim = 32
    value_dim = 64
    window_size = 33
    scale = qk_dim**-0.5

    q = torch.randn(
        sum(q_lengths),
        q_heads,
        qk_dim,
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        sum(k_lengths),
        kv_heads,
        qk_dim,
        dtype=dtype,
        device="cuda",
    )
    v = torch.randn(
        sum(k_lengths),
        kv_heads,
        value_dim,
        dtype=dtype,
        device="cuda",
    )
    with torch.no_grad():
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.silu(v) + 0.1
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )

    reference_tensors = [
        tensor.detach().double().requires_grad_(True)
        for tensor in (q, k, v)
    ]
    expected = packed_varlen_swa_fp64_oracle(
        *reference_tensors,
        q_lengths,
        k_lengths,
        window_size,
        scale,
    )
    actual_tensors = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in (q, k, v)
    ]
    actual = _flash_swa_varlen(
        *actual_tensors,
        window_size,
        scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        backend="sm90",
        deterministic=deterministic,
    )
    torch.testing.assert_close(
        actual,
        expected.to(dtype),
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
    )

    weight = torch.randn_like(actual) / math.sqrt(sum(q_lengths))
    expected.backward(weight.double())
    actual.backward(weight)
    for actual_tensor, expected_tensor in zip(
        actual_tensors, reference_tensors
    ):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad.to(dtype),
            rtol=RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )


from tests._numerics import (
    Case, DTYPES, HEAD_DIMS, MATRIX_LENGTHS, check_dimensions, cases_for, check_case, check_long,
    check_packed, check_rounding_bias,
)


@pytest.mark.parametrize('case', cases_for('window'), ids=lambda case: case.name)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('deterministic,high_precision', ((False, False), (True, True)))
def test_output_and_gradients(case, dtype, kv_heads, deterministic, high_precision):
    check_case('window', case, dtype, kv_heads, deterministic, high_precision)


@pytest.mark.full_matrix
@pytest.mark.parametrize('length', MATRIX_LENGTHS)
@pytest.mark.parametrize('dim', HEAD_DIMS)
@pytest.mark.parametrize('value_dim', HEAD_DIMS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('high_precision', (False, True))
def test_full_dimensions(length, dim, value_dim, dtype, kv_heads, deterministic, high_precision):
    check_dimensions('window', length, dim, value_dim, dtype, kv_heads,
                     deterministic, high_precision)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1))
@pytest.mark.parametrize('deterministic', (False, True))
def test_packed_output_and_gradients(dtype, kv_heads, deterministic):
    check_packed('window', dtype, kv_heads, deterministic)


@pytest.mark.parametrize('deterministic', (False, True))
def test_high_precision_gradient_rounding(deterministic):
    check_rounding_bias('window', deterministic)


@pytest.mark.long
@pytest.mark.parametrize('length', (4095, 4096, 4097, 8192, 8193, 32767, 32768, 32769))
@pytest.mark.parametrize('dim,value_dim', ((96, 128), (160, 192), (192, 160)))
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (2, 1))
@pytest.mark.parametrize('high_precision', (False, True))
def test_long_output_and_gradients(length, dim, value_dim, dtype, kv_heads, high_precision):
    check_long('window', length, dim, value_dim, dtype, kv_heads, high_precision)

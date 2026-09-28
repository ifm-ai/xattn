import importlib
import importlib.util

import pytest
import torch

from xattn import (
    sliding_chunk_softdelta_attention,
    sliding_window_softdelta_attention,
    softdelta_attention,
)
from xattn.ops._flash_softdelta import (
    flash_softdelta_gate,
    flash_softdelta_paired_training,
)
from xattn.ops._attention_core import AttentionVisibility
from xattn.ops.softdelta_attention import (
    _execute_flash_softdelta_reads,
    _make_plan,
)

from .reference import softdelta_fp64_oracle
from ._inputs import (
    ATOL,
    GRAD_ATOL,
    RTOL,
    _conditioned_inputs,
)


def _requires_sm90_extension():
    if importlib.util.find_spec("xattn_cuda") is None:
        pytest.skip("xattn CUDA extension is not built")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Flash SoftDelta requires SM90")


FLASH_CASES = (
    (
        "full",
        softdelta_attention,
        {},
        {"visibility": "full"},
    ),
    (
        "window",
        sliding_window_softdelta_attention,
        {"window_size": 3},
        {"visibility": "window", "span": 3},
    ),
    (
        "chunk",
        sliding_chunk_softdelta_attention,
        {"chunk_size": 3},
        {"visibility": "chunk", "span": 3},
    ),
)


def _plan_for_case(case_name, operation_kwargs):
    if case_name == "full":
        visibility = AttentionVisibility.CAUSAL_FULL
        span = None
    elif case_name == "window":
        visibility = AttentionVisibility.SLIDING_WINDOW
        span = operation_kwargs["window_size"]
    else:
        visibility = AttentionVisibility.SLIDING_CHUNK
        span = operation_kwargs["chunk_size"]
    return _make_plan(
        visibility,
        span,
        operation_kwargs.get("reset_chunk_pos_per_seq", False),
    )


def _assert_composed_close(
    actual,
    expected,
    expected_read,
    expected_past,
    gate,
    dtype,
):
    difference = (actual.double() - expected).abs()
    gate_weight = torch.sigmoid(gate.double())
    bound = (
        ATOL[dtype]
        + RTOL[dtype] * expected_read.abs()
        + gate_weight
        * (ATOL[dtype] + RTOL[dtype] * expected_past.abs())
        + ATOL[dtype]
        + RTOL[dtype] * expected.abs()
    )
    normalized = difference / bound
    assert bool((normalized <= 1).all().item()), (
        f"maximum composed normalized error={normalized.max().item()} "
        f"maximum absolute error={difference.max().item()}"
    )


def _reader_pair_kv_reuse_training(
    inputs,
    oracle_kwargs,
    deterministic,
    high_precision_output,
):
    visibility_name = oracle_kwargs["visibility"]
    visibility = {"full": 0, "window": 1, "chunk": 2}[visibility_name]
    span = (
        inputs[1].shape[1] - 1
        if visibility_name == "full"
        else oracle_kwargs["span"]
    )
    return flash_softdelta_paired_training(
        *inputs,
        span,
        inputs[0].shape[-1] ** -0.5,
        visibility,
        deterministic,
        high_precision_output,
    )


def _assert_flash_component_backward(
    q,
    k,
    v,
    plan,
    deterministic,
    dtype,
):
    actual_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in (q, k, v)
    ]
    expected_inputs = [
        tensor.detach().double().requires_grad_(True)
        for tensor in (q, k, v)
    ]
    actual_read, actual_past = _execute_flash_softdelta_reads(
        *actual_inputs,
        plan,
        q.shape[-1] ** -0.5,
        None,
        None,
        None,
        None,
        deterministic,
    )
    visibility = plan.primary.visibility
    if visibility is AttentionVisibility.CAUSAL_FULL:
        oracle_kwargs = {"visibility": "full"}
    elif visibility is AttentionVisibility.SLIDING_WINDOW:
        oracle_kwargs = {
            "visibility": "window",
            "span": plan.primary.span,
        }
    else:
        oracle_kwargs = {
            "visibility": "chunk",
            "span": plan.primary.span,
            "reset_position_per_segment": (
                plan.primary.reset_position_per_segment
            ),
        }
    gate = torch.zeros(
        q.shape[0],
        q.shape[1],
        q.shape[2] // 2,
        v.shape[3],
        1,
        dtype=torch.float64,
        device=q.device,
    )
    _, expected_read, expected_past, _, _ = softdelta_fp64_oracle(
        *expected_inputs,
        gate,
        return_aux=True,
        **oracle_kwargs,
    )
    for actual_component, expected_component in (
        (actual_read.reshape_as(expected_read), expected_read),
        (actual_past.reshape_as(expected_past), expected_past),
    ):
        component_grad = (
            torch.randn_like(actual_component)
            / actual_component.numel() ** 0.5
        )
        actual_grads = torch.autograd.grad(
            actual_component,
            actual_inputs,
            component_grad,
            retain_graph=True,
        )
        expected_grads = torch.autograd.grad(
            expected_component,
            expected_inputs,
            component_grad.double(),
            retain_graph=True,
        )
        for actual_grad, expected_grad in zip(
            actual_grads,
            expected_grads,
        ):
            torch.testing.assert_close(
                actual_grad,
                expected_grad.to(dtype),
                rtol=RTOL[dtype],
                atol=GRAD_ATOL[dtype],
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_dim", [1, 8, 16, 17, 64])
def test_flash_softdelta_gate_matches_fp64_forward_backward(
    dtype,
    group_dim,
):
    _requires_sm90_extension()
    torch.manual_seed(20260810 + group_dim)
    quantized = [
        tensor.to(dtype)
        for tensor in (
            torch.randn(2, 3, 2, 3, group_dim, device="cuda") * 0.2,
            torch.randn(2, 3, 2, 3, group_dim, device="cuda") * 0.2,
            torch.randn(2, 3, 2, 3, 1, device="cuda") * 0.5,
        )
    ]
    expected_inputs = [
        tensor.detach().double().requires_grad_(True)
        for tensor in quantized
    ]
    actual_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in quantized
    ]
    expected = expected_inputs[0] - torch.sigmoid(
        expected_inputs[2]
    ) * expected_inputs[1]
    actual = flash_softdelta_gate(*actual_inputs)
    torch.testing.assert_close(
        actual,
        expected.to(dtype),
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
    )

    output_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual.backward(output_grad)
    expected.backward(output_grad.double())
    for actual_input, expected_input in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad.to(dtype),
            rtol=RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case_name,operation_kwargs",
    [(case[0], case[2]) for case in FLASH_CASES],
    ids=[case[0] for case in FLASH_CASES],
)
def test_flash_softdelta_visibility_components_match_fp64_backward(
    dtype,
    case_name,
    operation_kwargs,
):
    _requires_sm90_extension()
    torch.manual_seed(20260812)
    q, k, v, _, _, _ = _conditioned_inputs(
        batch=2,
        length=11,
        q_heads=4,
        kv_heads=2,
        qk_dim=32,
        groups=4,
        group_dim=8,
        dtype=dtype,
        device="cuda",
    )
    _assert_flash_component_backward(
        q,
        k,
        v,
        _plan_for_case(case_name, operation_kwargs),
        True,
        dtype,
    )


@pytest.mark.parametrize("high_precision_output", [False, True])
@pytest.mark.parametrize(
    (
        "_case_name,operation,operation_kwargs,oracle_kwargs,length,"
        "qk_dim,groups,group_dim,dtype"
    ),
    [
        (
            "chunk-d32-v256-fp16",
            sliding_chunk_softdelta_attention,
            {"chunk_size": 5},
            {"visibility": "chunk", "span": 5},
            17,
            32,
            4,
            64,
            torch.float16,
        ),
        (
            "chunk-d32-v256-bf16",
            sliding_chunk_softdelta_attention,
            {"chunk_size": 5},
            {"visibility": "chunk", "span": 5},
            17,
            32,
            4,
            64,
            torch.bfloat16,
        ),
        (
            "full-length-one",
            softdelta_attention,
            {},
            {"visibility": "full"},
            1,
            64,
            2,
            32,
            torch.bfloat16,
        ),
        (
            "window-zero",
            sliding_window_softdelta_attention,
            {"window_size": 0},
            {"visibility": "window", "span": 0},
            17,
            64,
            2,
            32,
            torch.bfloat16,
        ),
        (
            "full-scalar-gate-v96",
            softdelta_attention,
            {},
            {"visibility": "full"},
            7,
            64,
            8,
            12,
            torch.bfloat16,
        ),
        (
            "window-scalar-gate-v160",
            sliding_window_softdelta_attention,
            {"window_size": 3},
            {"visibility": "window", "span": 3},
            7,
            32,
            32,
            5,
            torch.float16,
        ),
    ],
    ids=[
        "chunk-d32-v256-fp16",
        "chunk-d32-v256-bf16",
        "full-length-one",
        "window-zero",
        "full-scalar-gate-v96",
        "window-scalar-gate-v160",
    ],
)
def test_flash_softdelta_paired_backward_edge_cases(
    _case_name,
    operation,
    operation_kwargs,
    oracle_kwargs,
    length,
    qk_dim,
    groups,
    group_dim,
    dtype,
    high_precision_output,
):
    _requires_sm90_extension()
    torch.manual_seed(20260828)
    quantized = _conditioned_inputs(
        batch=1,
        length=length,
        q_heads=4,
        kv_heads=2,
        qk_dim=qk_dim,
        groups=groups,
        group_dim=group_dim,
        dtype=dtype,
        device="cuda",
    )[:4]
    actual_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in quantized
    ]
    expected_inputs = [
        tensor.detach().double().requires_grad_(True)
        for tensor in quantized
    ]
    actual = _reader_pair_kv_reuse_training(
        actual_inputs,
        oracle_kwargs,
        deterministic=True,
        high_precision_output=high_precision_output,
    )
    expected, expected_read, expected_past, _, _ = softdelta_fp64_oracle(
        *expected_inputs,
        return_aux=True,
        **oracle_kwargs,
    )
    _assert_composed_close(
        actual,
        expected,
        expected_read,
        expected_past,
        actual_inputs[3],
        dtype,
    )
    output_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, actual_inputs, output_grad)
    expected_grads = torch.autograd.grad(
        expected, expected_inputs, output_grad.double()
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad,
            expected_grad.to(dtype),
            rtol=RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )


@pytest.mark.parametrize("high_precision_output", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "_case_name,operation,operation_kwargs,oracle_kwargs",
    [
        (
            "window-block-boundary",
            sliding_window_softdelta_attention,
            {"window_size": 65},
            {"visibility": "window", "span": 65},
        ),
        (
            "chunk-block-boundary",
            sliding_chunk_softdelta_attention,
            {"chunk_size": 64},
            {"visibility": "chunk", "span": 64},
        ),
    ],
    ids=("window", "chunk"),
)
def test_flash_softdelta_paired_backward_block_boundaries_match_fp64(
    _case_name,
    operation,
    operation_kwargs,
    oracle_kwargs,
    deterministic,
    high_precision_output,
):
    _requires_sm90_extension()
    torch.manual_seed(20260829)
    dtype = torch.bfloat16
    quantized = _conditioned_inputs(
        batch=1,
        length=129,
        q_heads=4,
        kv_heads=2,
        qk_dim=64,
        groups=2,
        group_dim=32,
        dtype=dtype,
        device="cuda",
    )[:4]
    actual_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in quantized
    ]
    expected_inputs = [
        tensor.detach().double().requires_grad_(True)
        for tensor in quantized
    ]
    actual = _reader_pair_kv_reuse_training(
        actual_inputs,
        oracle_kwargs,
        deterministic=deterministic,
        high_precision_output=high_precision_output,
    )
    expected, expected_read, expected_past, _, _ = softdelta_fp64_oracle(
        *expected_inputs,
        return_aux=True,
        **oracle_kwargs,
    )
    _assert_composed_close(
        actual,
        expected,
        expected_read,
        expected_past,
        actual_inputs[3],
        dtype,
    )
    output_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, actual_inputs, output_grad)
    expected_grads = torch.autograd.grad(
        expected, expected_inputs, output_grad.double()
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
    "qk_dim,value_dim",
    [(192, 32), (32, 128)],
    ids=["v32", "v128"],
)
@pytest.mark.parametrize(
    "_case_name,operation,operation_kwargs,oracle_kwargs",
    FLASH_CASES,
    ids=[case[0] for case in FLASH_CASES],
)
def test_flash_softdelta_gate_extremes_match_fp64(
    dtype,
    qk_dim,
    value_dim,
    _case_name,
    operation,
    operation_kwargs,
    oracle_kwargs,
):
    _requires_sm90_extension()
    torch.manual_seed(20260814 + qk_dim + value_dim)
    q, k, v, _, _, _ = _conditioned_inputs(
        batch=1,
        length=7,
        q_heads=4,
        kv_heads=2,
        qk_dim=qk_dim,
        groups=4,
        group_dim=value_dim // 4,
        dtype=dtype,
        device="cuda",
    )
    gate = torch.tensor(
        [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0],
        dtype=dtype,
        device="cuda",
    ).reshape(1, 7, 1, 1, 1)
    gate = gate.expand(1, 7, 4, 4, 1).contiguous()
    expected, expected_read, expected_past, _, _ = softdelta_fp64_oracle(
        q.double(),
        k.double(),
        v.double(),
        gate.double(),
        return_aux=True,
        **oracle_kwargs,
    )
    actual = operation(
        q,
        k,
        v,
        gate,
        backend="sm90",
        **operation_kwargs,
    )
    assert bool(torch.isfinite(actual).all().item())
    _assert_composed_close(
        actual,
        expected,
        expected_read,
        expected_past,
        gate,
        dtype,
    )


from tests._numerics import Case, DTYPES, HEAD_DIMS, MATRIX_LENGTHS, check_dimensions, cases_for, check_case, check_long


@pytest.mark.parametrize('operation,case', [
    (operation, case) for operation in ('softdelta_full', 'softdelta_window', 'softdelta_chunk')
    for case in cases_for(operation)
], ids=[f'{operation}-{case.name}' for operation in ('softdelta_full', 'softdelta_window', 'softdelta_chunk')
        for case in cases_for(operation)])
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('deterministic,high_precision', ((False, False), (True, True)))
def test_output_and_gradients(operation, case, dtype, kv_heads, deterministic, high_precision):
    check_case(operation, case, dtype, kv_heads, deterministic, high_precision)


@pytest.mark.full_matrix
@pytest.mark.parametrize('length', MATRIX_LENGTHS)
@pytest.mark.parametrize('operation', ('softdelta_full', 'softdelta_window', 'softdelta_chunk'))
@pytest.mark.parametrize('dim', HEAD_DIMS)
@pytest.mark.parametrize('value_dim', HEAD_DIMS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('high_precision', (False, True))
def test_full_dimensions(length, operation, dim, value_dim, dtype, kv_heads, deterministic, high_precision):
    check_dimensions(operation, length, dim, value_dim, dtype, kv_heads,
                     deterministic, high_precision)


@pytest.mark.long
@pytest.mark.parametrize('length', (8192, 16384, 65536))
@pytest.mark.parametrize('dim', (64, 128, 256))
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1))
@pytest.mark.parametrize('high_precision', (False, True))
def test_long_output_and_gradients(length, dim, dtype, kv_heads, high_precision):
    check_long('softdelta_full', length, dim, dim, dtype, kv_heads, high_precision)

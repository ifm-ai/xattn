"""CUDA parity for causal_flash_attn against causal attention on already-quantized inputs."""

import pytest
import torch
import torch.nn.functional as F

from xattn import (
    flash_sca_sm90_available,
    causal_flash_attn,
    causal_flash_attn_bwd,
    causal_flash_attn_fwd,
)


def _oracle(q, k, v, prev_k, prev_v, segment, *, return_probabilities=False):
    if prev_k is not None:
        k = torch.cat((prev_k, k), dim=1)
        v = torch.cat((prev_v, v), dim=1)
    repeats = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(repeats, dim=2)
    v = v.repeat_interleave(repeats, dim=2)
    q_length, k_length = q.shape[1], k.shape[1]
    q_pos = torch.arange(q_length, device=q.device) + k_length - q_length
    k_pos = torch.arange(k_length, device=q.device)
    allowed = (k_pos[None, :] <= q_pos[:, None])[None]
    if segment is not None:
        allowed = allowed & (segment[:, -q_length:, None] == segment[:, None, :])
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~allowed[:, None], -torch.inf)
    probabilities = scores.softmax(-1)
    output = torch.einsum("bhqk,bkhv->bqhv", probabilities, v)
    return (output, probabilities) if return_probabilities else output


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("high_precision_output", [False, True])
@pytest.mark.parametrize(
    "head_dim,value_dim",
    [
        (40, 40), (104, 104), (160, 160),
        (128, 192), (160, 192), (192, 160),
        (128, 256), (256, 128), (256, 160), (256, 192),
        (120, 184), (248, 152),
    ],
)
def test_causal_flash_attn_grouped_gradients_with_padded_kernel_dimension(
    dtype, kv_heads, deterministic, high_precision_output, head_dim, value_dim
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(20260919)
    batch, length, heads = 2, 193, 4
    q = F.normalize(
        torch.randn(batch, length, heads, head_dim, device="cuda", dtype=dtype),
        dim=-1,
    ).requires_grad_()
    if not flash_sca_sm90_available(q):
        pytest.skip("the xattn SM90 extension is unavailable")
    k = F.normalize(torch.randn_like(q[:, :, :kv_heads]), dim=-1).requires_grad_()
    v = (F.silu(torch.randn(
        batch, length, kv_heads, value_dim, device="cuda", dtype=dtype
    )) + 0.1).requires_grad_()
    inputs = (q, k, v)
    reference = [x.detach().double().requires_grad_() for x in inputs]
    expected, probabilities = _oracle(
        *reference, None, None, None, return_probabilities=True
    )
    y = causal_flash_attn(
        *inputs, backend="sm90", deterministic=deterministic,
        high_precision_output=high_precision_output,
    )
    dy = torch.randn_like(y)
    actual_grads = torch.autograd.grad(y, inputs, dy, retain_graph=deterministic)
    expected_grads = torch.autograd.grad(expected, reference, dy.double())
    atol = 2e-4 if dtype == torch.float16 else 1e-3
    grad_atol = 2e-4 if dtype == torch.float16 else 2e-3
    rtol = 1e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(y, expected.to(dtype), atol=atol, rtol=rtol)
    probability_error = (
        probabilities.detach().to(dtype).double() - probabilities.detach()
    ).abs()
    dv_rounding = torch.einsum(
        "bhqk,bqhv->bkhv", probability_error, dy.double().abs()
    ).reshape(batch, length, kv_heads, heads // kv_heads, value_dim).sum(3)
    for index, (actual, ref, original) in enumerate(
        zip(actual_grads, expected_grads, inputs)
    ):
        assert actual.shape == original.shape
        assert actual.is_contiguous()
        ref = ref.to(dtype)
        if index == 2:
            error = (actual.double() - ref.double()).abs()
            budget = grad_atol + rtol * ref.double().abs() + dv_rounding
            assert bool((error <= budget).all()), "dV exceeds its FP64 rounding bound"
        else:
            torch.testing.assert_close(actual, ref, atol=grad_atol, rtol=rtol)
    if deterministic:
        repeated = torch.autograd.grad(y, inputs, dy)
        for actual, repeat in zip(actual_grads, repeated):
            torch.testing.assert_close(actual, repeat, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,kv_heads", [(2, 2), (4, 2), (4, 1)])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("high_precision_output", [False, True])
@pytest.mark.parametrize(
    "length,prev_length,metadata,head_dim,value_dim",
    [
        (1, 0, "dense", 32, 64),
        (193, 0, "dense", 32, 64),
        (193, 0, "dense", 40, 64),
        (193, 0, "dense", 128, 104),
                                (193, 0, "segment", 32, 64),
        (193, 0, "bos", 32, 64),
        (4097, 0, "dense", 64, 64),
        (4097, 0, "dense", 128, 128),
        (1025, 0, "dense", 256, 256),
    ],
)
def test_causal_flash_attn_matches_fp64_oracle(
    dtype,
    heads,
    kv_heads,
    deterministic,
    high_precision_output,
    length,
    prev_length,
    metadata,
    head_dim,
    value_dim,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("xattn_cuda")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("flash attention currently requires SM90")
    torch.manual_seed(20260918)

    def make(n, h, dim, normalize):
        x = torch.randn(1, n, h, dim, device="cuda", dtype=dtype)
        x = F.normalize(x, dim=-1) if normalize else F.silu(x) + 0.1
        return x.requires_grad_()

    q = make(length, heads, head_dim, True)
    if not flash_sca_sm90_available(q):
        pytest.skip("the xattn SM90 extension is unavailable")
    k = make(length, kv_heads, head_dim, True)
    v = make(length, kv_heads, value_dim, False)
    prev_k = make(prev_length, kv_heads, head_dim, True) if prev_length else None
    prev_v = make(prev_length, kv_heads, value_dim, False) if prev_length else None
    tensors = [q, k, v] + ([prev_k, prev_v] if prev_length else [])
    reference = [x.detach().double().requires_grad_() for x in tensors]
    kwargs = {}
    segment = None
    if metadata != "dense":
        segment = (torch.arange(length + prev_length, device="cuda") // 53)[None]
        if metadata == "segment":
            kwargs["segment_idx"] = segment
        else:
            bos = torch.zeros_like(segment, dtype=torch.bool)
            bos[:, 0] = True
            bos[:, 1:] = segment[:, 1:] != segment[:, :-1]
            kwargs["bos_mask"] = bos

    expected, probabilities = _oracle(
        *reference[:3],
        reference[3] if prev_length else None,
        reference[4] if prev_length else None,
        segment,
        return_probabilities=True,
    )
    y = causal_flash_attn(
        q,
        k,
        v,
        **kwargs,
        deterministic=deterministic,
        high_precision_output=high_precision_output,
    )
    dy = torch.randn_like(y)
    y.backward(dy)
    expected.backward(dy.double())
    # dV uses P cast to the kernel dtype even with FP32 output state.
    # Bound that rounding contribution in FP64: |round(P)-P|^T @ |dY|.
    # Summing the grouped query heads also covers GQA/MQA cancellation.
    probability_error = (
        probabilities.detach().to(dtype).double() - probabilities.detach()
    ).abs()
    dv_rounding = (
        torch.einsum("bhqk,bqhv->bkhv", probability_error, dy.double().abs())
        .reshape(1, length + prev_length, kv_heads, heads // kv_heads, value_dim)
        .sum(3)
    )
    atol = 2e-4 if dtype == torch.float16 else 1e-3
    grad_atol = 2e-4 if dtype == torch.float16 else 2e-3
    rtol = 1e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(y, expected.to(dtype), atol=atol, rtol=rtol)

    def check_grad(index, actual, ref):
        expected_grad = ref.grad.to(dtype)
        if index in (2, 4):
            budget = (
                dv_rounding[:, prev_length:]
                if index == 2
                else dv_rounding[:, :prev_length]
            )
            error = (actual.double() - expected_grad.double()).abs()
            tolerance = grad_atol + rtol * expected_grad.double().abs() + budget
            assert (error / tolerance).max().item() <= 1, (
                "dV exceeds its FP64 rounding bound"
            )
        else:
            torch.testing.assert_close(actual, expected_grad, atol=grad_atol, rtol=rtol)

    for index, (actual, ref) in enumerate(zip(tensors, reference)):
        check_grad(index, actual.grad, ref)

    actual_y, _, lse = causal_flash_attn_fwd(q, k, v, **kwargs)
    assert lse.shape == (1, heads, length)
    assert lse.dtype == torch.float32
    torch.testing.assert_close(actual_y, expected.to(dtype), atol=atol, rtol=rtol)
    grads = causal_flash_attn_bwd(
        dy,
        q,
        k,
        v,
        actual_y,
        lse,
        **kwargs,
        deterministic=deterministic,
    )
    if not prev_length:
        assert len(grads) == 3
    for index, (grad, ref) in enumerate(zip(grads, reference)):
        check_grad(index, grad, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [4, 1])
@pytest.mark.parametrize("high_precision_output", [False, True])
@pytest.mark.parametrize(
    "length,head_dim,value_dim,padded_dim,shifted_storage",
    [(8193, 40, 56, 64, False), (8193, 64, 40, 64, False),
     (16385, 104, 128, 128, False), (16385, 128, 104, 128, False),
     (8193, 40, 56, 64, True)],
)
@pytest.mark.long
def test_causal_flash_attn_long_padding_preserves_saved_state_backward(
    dtype, kv_heads, high_precision_output,
    length, head_dim, value_dim, padded_dim, shifted_storage,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    extension = pytest.importorskip("xattn_cuda")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("flash attention currently requires SM90")
    torch.manual_seed(20260920)
    batch, heads = 2, 4

    def make(h, dim, normalize=False):
        # Exercise host contiguity handling before the fused copies.
        x = torch.randn(length, batch, h, dim, device="cuda", dtype=dtype)
        x = F.normalize(x, dim=-1) if normalize else F.silu(x) + 0.1
        return x.transpose(0, 1)

    q, k, v = make(heads, head_dim, True), make(kv_heads, head_dim, True), make(kv_heads, value_dim)
    dy = make(heads, value_dim)
    scale = head_dim ** -0.5
    state = torch.empty(batch, length, heads, value_dim, device="cuda", dtype=torch.float32) if high_precision_output else None
    y, lse = extension.ops.causal_flash_attn_fwd(q, k, v, scale, output_state=state)
    saved = state if high_precision_output else y
    if shifted_storage:
        def shift(tensor):
            storage = torch.empty(
                tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype
            )
            view = storage[1:].view(tensor.shape)
            view.copy_(tensor)
            assert view.is_contiguous() and view.data_ptr() % 16 != 0
            return view

        q, k, v, dy, saved = map(shift, (q, k, v, dy, saved))
    actual = extension.ops.causal_flash_attn_bwd(
        dy, q, k, v, saved, lse, scale, deterministic=True
    )[:3]
    # Supply the same saved forward state to the aligned kernel, with explicit
    # reference padding. This isolates copies from forward rounding changes.
    pq, pk, pv, pdy, psaved = [
        F.pad(t, (0, padded_dim - t.size(-1))).contiguous()
        for t in (q, k, v, dy, saved)
    ]
    padded = extension.ops.causal_flash_attn_bwd(
        pdy, pq, pk, pv, psaved, lse, scale, deterministic=True
    )[:3]
    for got, ref, original in zip(actual, padded, (q, k, v)):
        assert got.shape == original.shape
        assert got.is_contiguous()
        torch.testing.assert_close(got, ref[..., :original.size(-1)], atol=0, rtol=0)

from tests._numerics import Case, DTYPES, HEAD_DIMS, MATRIX_LENGTHS, check_dimensions, cases_for, check_case, check_long


@pytest.mark.parametrize('case', cases_for('full'), ids=lambda case: case.name)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('deterministic,high_precision', ((False, False), (True, True)))
def test_output_and_gradients(case, dtype, kv_heads, deterministic, high_precision):
    check_case('full', case, dtype, kv_heads, deterministic, high_precision)


@pytest.mark.full_matrix
@pytest.mark.parametrize('length', MATRIX_LENGTHS)
@pytest.mark.parametrize('dim', HEAD_DIMS)
@pytest.mark.parametrize('value_dim', HEAD_DIMS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('high_precision', (False, True))
def test_full_dimensions(length, dim, value_dim, dtype, kv_heads, deterministic, high_precision):
    check_dimensions('full', length, dim, value_dim, dtype, kv_heads,
                     deterministic, high_precision)


@pytest.mark.long
@pytest.mark.parametrize('length', (4095, 4096, 4097, 8192, 8193, 32767, 32768, 32769))
@pytest.mark.parametrize('dim,value_dim', ((96, 128), (160, 192), (192, 160)))
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (2, 1))
@pytest.mark.parametrize('high_precision', (False, True))
def test_long_output_and_gradients(length, dim, value_dim, dtype, kv_heads, high_precision):
    check_long('full', length, dim, value_dim, dtype, kv_heads, high_precision)

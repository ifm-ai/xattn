import math

import pytest
import torch
import torch.nn.functional as F

from xattn import (
    flash_sca,
    flash_sca_bwd,
    flash_sca_fwd,
    flash_sca_sm90_available,
)
from xattn.ops.sliding_chunk_attention import (
    _flash_sca,
    _flash_sca_varlen,
)


ATOL = {
    torch.float32: 1e-6,
    torch.bfloat16: 1e-3,
    torch.float16: 2e-4,
}
GRAD_ATOL = {
    torch.float32: 1e-6,
    torch.bfloat16: 2e-3,
    torch.float16: 2e-4,
}
RTOL = {
    torch.float32: 1e-4,
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
}


def naive_sliding_chunk_attention(
    query,
    key,
    value,
    prev_key_chunk,
    prev_value_chunk,
    chunk_size,
    q_segment_idx,
    k_segment_idx,
    scale,
):
    """Chunk-loop reference for global chunk positions."""
    bsz, seq_len, _, _ = query.shape
    assert seq_len == key.shape[1] and seq_len == value.shape[1]
    outputs = []
    for chunk_id in range((seq_len + chunk_size - 1) // chunk_size):
        q_start = chunk_id * chunk_size
        kv_start = q_start if chunk_id == 0 else (chunk_id - 1) * chunk_size
        end = min((chunk_id + 1) * chunk_size, seq_len)

        q = query[:, q_start:end]
        k = key[:, kv_start:end]
        v = value[:, kv_start:end]
        q_idx = (
            None if q_segment_idx is None else q_segment_idx[:, q_start:end]
        )
        if k_segment_idx is None:
            k_idx = None
        elif chunk_id == 0 and prev_key_chunk is not None:
            k_idx = k_segment_idx[:, : chunk_size + end]
        else:
            k_offset = chunk_size if prev_key_chunk is not None else 0
            k_idx = k_segment_idx[:, k_offset + kv_start : k_offset + end]

        if chunk_id == 0 and prev_key_chunk is not None:
            k = torch.cat([prev_key_chunk, k], dim=1)
            v = torch.cat([prev_value_chunk, v], dim=1)

        ctx_len = k.shape[1]
        q_len = q.shape[1]
        q_pos = torch.arange(q_start, end, device=query.device)
        if chunk_id == 0 and prev_key_chunk is not None:
            k_pos = torch.cat(
                [
                    torch.arange(-chunk_size, 0, device=query.device),
                    torch.arange(0, end, device=query.device),
                ]
            )
        else:
            k_pos = torch.arange(kv_start, end, device=query.device)

        attn_mask = torch.zeros(
            (bsz, q_len, ctx_len),
            dtype=torch.float32,
            device=query.device,
        )
        attn_mask = attn_mask.masked_fill(
            (k_pos.unsqueeze(0) > q_pos.unsqueeze(1)).unsqueeze(0),
            float("-inf"),
        )
        if q_idx is not None:
            attn_mask = attn_mask.masked_fill(
                q_idx.unsqueeze(2) != k_idx.unsqueeze(1),
                float("-inf"),
            )

        scores = torch.matmul(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float().transpose(2, 3),
        )
        scores = scores * scale + attn_mask.unsqueeze(1)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        y = torch.matmul(weights.to(v.dtype), v.transpose(1, 2)).transpose(1, 2)
        outputs.append(y)
    return torch.cat(outputs, dim=1)


def _append_runs(segment_idx, batch, length, runs):
    values = segment_idx[batch, :length].detach().cpu().tolist()
    start = 0
    current = values[0]
    for idx in range(1, length):
        if values[idx] != current:
            runs.append((batch, start, idx - start, current))
            start = idx
            current = values[idx]
    runs.append((batch, start, length - start, current))


def _reset_runs(q_segment_idx, k_segment_idx, batch, q_len, k_len):
    if q_segment_idx is None:
        q_runs = [(b, 0, q_len, b) for b in range(batch)]
        k_runs = [(b, 0, k_len, b) for b in range(batch)]
        return q_runs, k_runs

    q_runs = []
    all_k_runs = []
    for b in range(batch):
        _append_runs(q_segment_idx, b, q_len, q_runs)
        _append_runs(k_segment_idx, b, k_len, all_k_runs)

    k_pos = 0
    k_runs = []
    for q_run in q_runs:
        q_batch, _, _, q_segment = q_run
        while k_pos < len(all_k_runs) and (
            all_k_runs[k_pos][0] < q_batch
            or (
                all_k_runs[k_pos][0] == q_batch
                and all_k_runs[k_pos][3] != q_segment
            )
        ):
            k_pos += 1
        assert k_pos < len(all_k_runs)
        k_runs.append(all_k_runs[k_pos])
        k_pos += 1
    return q_runs, k_runs


def naive_sliding_chunk_attention_reset_chunk_pos_per_seq(
    query,
    key,
    value,
    prev_key_chunk,
    prev_value_chunk,
    chunk_size,
    q_segment_idx,
    k_segment_idx,
    scale,
):
    """Run-loop reference for reset chunk positions."""
    bsz, seq_len, n_heads, _ = query.shape
    if prev_key_chunk is not None:
        key = torch.cat([prev_key_chunk, key], dim=1)
        value = torch.cat([prev_value_chunk, value], dim=1)

    q_runs, k_runs = _reset_runs(
        q_segment_idx, k_segment_idx, bsz, seq_len, key.shape[1]
    )
    outputs = []
    for q_run, k_run in zip(q_runs, k_runs):
        q_batch, q_start, q_len, _ = q_run
        k_batch, k_start, k_len, _ = k_run
        assert q_batch == k_batch
        assert k_len >= q_len
        q = query[q_batch, q_start : q_start + q_len]
        k = key[k_batch, k_start : k_start + k_len]
        v = value[k_batch, k_start : k_start + k_len]

        q_pos = torch.arange(q_len, device=query.device) + (k_len - q_len)
        k_pos = torch.arange(k_len, device=query.device)
        chunk_left = (
            torch.div(q_pos, chunk_size, rounding_mode="floor") * chunk_size
            - chunk_size
        )
        attn_mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1)) | (
            k_pos.unsqueeze(0) < chunk_left.unsqueeze(1)
        )

        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
        scores = scores.masked_fill(attn_mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        outputs.append(torch.einsum("hqk,khv->qhv", weights.to(v.dtype), v))

    return torch.cat(outputs, dim=0).reshape(
        bsz, seq_len, n_heads, value.shape[-1]
    )


def _ordered_segment_runs(segment_idx):
    """Return per-batch contiguous runs and reject ambiguous segment order."""
    result = []
    for batch_idx in range(segment_idx.shape[0]):
        values = segment_idx[batch_idx].detach().cpu().tolist()
        assert values
        assert all(left <= right for left, right in zip(values, values[1:]))
        runs = []
        start = 0
        for pos in range(1, len(values)):
            if values[pos] != values[start]:
                runs.append((start, pos, values[start]))
                start = pos
        runs.append((start, len(values), values[start]))
        assert len({segment for _, _, segment in runs}) == len(runs)
        result.append(runs)
    return result


def dense_sliding_chunk_allowed_mask(
    query,
    key,
    prev_key_chunk,
    chunk_size,
    q_segment_idx,
    k_segment_idx,
    reset_chunk_pos_per_seq,
):
    """Construct the complete token-level FlashSCA allow matrix."""
    batch, q_len = query.shape[:2]
    assert key.shape[:2] == (batch, q_len)
    prev_len = 0 if prev_key_chunk is None else prev_key_chunk.shape[1]
    assert prev_len in (0, chunk_size)
    k_len = prev_len + key.shape[1]
    assert (q_segment_idx is None) == (k_segment_idx is None)

    if not reset_chunk_pos_per_seq:
        q_position = torch.arange(q_len, device=query.device)
        k_position = torch.arange(-prev_len, q_len, device=query.device)
        chunk_left = (
            torch.div(q_position, chunk_size, rounding_mode="floor")
            * chunk_size
            - chunk_size
        )
        allowed = (k_position.unsqueeze(0) <= q_position.unsqueeze(1)) & (
            k_position.unsqueeze(0) >= chunk_left.unsqueeze(1)
        )
        allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
        if q_segment_idx is not None:
            assert q_segment_idx.shape == (batch, q_len)
            assert k_segment_idx.shape == (batch, k_len)
            allowed = allowed & (
                q_segment_idx.unsqueeze(2) == k_segment_idx.unsqueeze(1)
            )
        return allowed

    q_local = torch.empty(
        (batch, q_len), dtype=torch.long, device=query.device
    )
    k_local = torch.empty(
        (batch, k_len), dtype=torch.long, device=query.device
    )
    q_problem = torch.empty_like(q_local)
    k_problem = torch.empty_like(k_local)
    if q_segment_idx is None:
        q_local.copy_(
            torch.arange(prev_len, prev_len + q_len, device=query.device)
        )
        k_local.copy_(torch.arange(k_len, device=query.device))
        q_problem.zero_()
        k_problem.zero_()
    else:
        assert q_segment_idx.shape == (batch, q_len)
        assert k_segment_idx.shape == (batch, k_len)
        q_runs_by_batch = _ordered_segment_runs(q_segment_idx)
        k_runs_by_batch = _ordered_segment_runs(k_segment_idx)
        for batch_idx, (q_runs, k_runs) in enumerate(
            zip(q_runs_by_batch, k_runs_by_batch)
        ):
            k_by_segment = {
                segment: (run_idx, start, end)
                for run_idx, (start, end, segment) in enumerate(k_runs)
            }
            for run_idx, (start, end, _) in enumerate(k_runs):
                k_local[batch_idx, start:end] = torch.arange(
                    end - start, device=query.device
                )
                k_problem[batch_idx, start:end] = run_idx
            for q_start, q_end, segment in q_runs:
                assert segment in k_by_segment
                run_idx, k_start, k_end = k_by_segment[segment]
                q_run_len = q_end - q_start
                k_run_len = k_end - k_start
                assert k_run_len >= q_run_len
                q_local[batch_idx, q_start:q_end] = torch.arange(
                    q_run_len, device=query.device
                ) + (k_run_len - q_run_len)
                q_problem[batch_idx, q_start:q_end] = run_idx

    chunk_left = (
        torch.div(q_local, chunk_size, rounding_mode="floor") * chunk_size
        - chunk_size
    )
    allowed = q_problem.unsqueeze(2) == k_problem.unsqueeze(1)
    allowed = allowed & (k_local.unsqueeze(1) <= q_local.unsqueeze(2))
    allowed = allowed & (k_local.unsqueeze(1) >= chunk_left.unsqueeze(2))
    return allowed


def dense_sliding_chunk_attention_reference(
    query,
    key,
    value,
    prev_key_chunk,
    prev_value_chunk,
    chunk_size,
    q_segment_idx,
    k_segment_idx,
    scale,
    reset_chunk_pos_per_seq,
):
    """High-precision dense-mask reference for both position modes."""
    assert key.shape[1] == value.shape[1] == query.shape[1]
    assert (prev_key_chunk is None) == (prev_value_chunk is None)
    allowed = dense_sliding_chunk_allowed_mask(
        query,
        key,
        prev_key_chunk,
        chunk_size,
        q_segment_idx,
        k_segment_idx,
        reset_chunk_pos_per_seq,
    )
    if prev_key_chunk is not None:
        key = torch.cat([prev_key_chunk, key], dim=1)
        value = torch.cat([prev_value_chunk, value], dim=1)
    assert torch.all(allowed.any(dim=-1))
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * scale
    scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhv->bqhv", weights, value)


def make_segments(
    batch, seq_len, chunk_size, prev_chunk, device, prev_continuation=False
):
    bos_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    bos_mask[:, 0] = True
    if seq_len > 3:
        bos_mask[:, 3::5] = True

    if not prev_chunk:
        segment_idx = torch.cumsum(bos_mask, dim=-1)
        return segment_idx, segment_idx

    if prev_continuation:
        bos_mask[:, 0] = False
    prev_bos_mask = torch.zeros(batch, chunk_size, dtype=torch.bool, device=device)
    prev_bos_mask[:, 0] = True
    total_bos_mask = torch.cat([prev_bos_mask, bos_mask], dim=1)
    k_segment_idx = torch.cumsum(total_bos_mask, dim=-1)
    q_segment_idx = k_segment_idx[:, chunk_size:]
    return q_segment_idx, k_segment_idx


def make_inputs(
    batch,
    seq_len,
    heads,
    qk_dim,
    v_dim,
    chunk_size,
    prev_chunk,
    dtype,
    prev_continuation=False,
):
    device = "cuda"
    with torch.no_grad():
        q = torch.randn(batch, seq_len, heads, qk_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, heads, qk_dim, dtype=dtype, device=device)
        v = torch.randn(batch, seq_len, heads, v_dim, dtype=dtype, device=device)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.silu(v) + 0.1
        q_segment_idx, k_segment_idx = make_segments(
            batch,
            seq_len,
            chunk_size,
            prev_chunk,
            device,
            prev_continuation,
        )
        if prev_chunk:
            prev_k = torch.randn(
                batch, chunk_size, heads, qk_dim, dtype=dtype, device=device
            )
            prev_v = torch.randn(
                batch, chunk_size, heads, v_dim, dtype=dtype, device=device
            )
            prev_k = F.normalize(prev_k, dim=-1)
            prev_v = F.silu(prev_v) + 0.1
        else:
            prev_k = None
            prev_v = None
    return q, k, v, prev_k, prev_v, q_segment_idx, k_segment_idx


def _cpu_reference_case(prev_mode):
    torch.manual_seed(17)
    batch, seq_len, heads, qk_dim, v_dim, chunk_size = 2, 11, 2, 3, 4, 4
    q = torch.randn(batch, seq_len, heads, qk_dim, dtype=torch.double)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, heads, v_dim, dtype=torch.double)
    if prev_mode == "none":
        prev_k = prev_v = None
        q_segment_idx = torch.tensor(
            [
                [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3],
                [4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7],
            ]
        )
        k_segment_idx = q_segment_idx
    else:
        prev_k = torch.randn(
            batch, chunk_size, heads, qk_dim, dtype=torch.double
        )
        prev_v = torch.randn(
            batch, chunk_size, heads, v_dim, dtype=torch.double
        )
        if prev_mode == "boundary":
            prev_segment_idx = torch.tensor(
                [[0, 0, 0, 0], [10, 10, 10, 10]]
            )
            q_segment_idx = torch.tensor(
                [
                    [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4],
                    [11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14],
                ]
            )
        else:
            assert prev_mode == "continuation"
            prev_segment_idx = torch.tensor(
                [[0, 0, 1, 1], [10, 10, 10, 11]]
            )
            q_segment_idx = torch.tensor(
                [
                    [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4],
                    [11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14],
                ]
            )
        k_segment_idx = torch.cat(
            [prev_segment_idx, q_segment_idx], dim=1
        )
    return (
        q,
        k,
        v,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        chunk_size,
    )


def test_dense_golden_encodes_global_and_reset_chunk_positions():
    batch, seq_len, heads, qk_dim, v_dim, chunk_size = 1, 10, 1, 1, 1, 4
    q = torch.ones(batch, seq_len, heads, qk_dim, dtype=torch.double)
    k = torch.zeros_like(q)
    v = torch.zeros(batch, seq_len, heads, v_dim, dtype=torch.double)
    v[:, 3] = 10.0
    segment_idx = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])

    global_mask = dense_sliding_chunk_allowed_mask(
        q, k, None, chunk_size, segment_idx, segment_idx, False
    )
    reset_mask = dense_sliding_chunk_allowed_mask(
        q, k, None, chunk_size, segment_idx, segment_idx, True
    )
    assert torch.where(global_mask[0, 8])[0].tolist() == [4, 5, 6, 7, 8]
    assert torch.where(reset_mask[0, 8])[0].tolist() == [3, 4, 5, 6, 7, 8]

    y_global = dense_sliding_chunk_attention_reference(
        q, k, v, None, None, chunk_size, segment_idx, segment_idx, 1.0, False
    )
    y_reset = dense_sliding_chunk_attention_reference(
        q, k, v, None, None, chunk_size, segment_idx, segment_idx, 1.0, True
    )
    torch.testing.assert_close(y_global[:, 8], torch.zeros_like(y_global[:, 8]))
    torch.testing.assert_close(
        y_reset[:, 8], torch.full_like(y_reset[:, 8], 10.0 / 6.0)
    )


@pytest.mark.parametrize("reset_chunk_pos_per_seq", [False, True])
@pytest.mark.parametrize("prev_mode", ["none", "boundary", "continuation"])
def test_dense_golden_matches_loop_structural_reference(
    reset_chunk_pos_per_seq, prev_mode
):
    case = _cpu_reference_case(prev_mode)
    q, k, v, prev_k, prev_v, q_segment_idx, k_segment_idx, chunk_size = case
    scale = 1.0 / math.sqrt(q.shape[-1])

    dense_tensors = [
        tensor.clone().requires_grad_() for tensor in (q, k, v)
    ]
    loop_tensors = [
        tensor.clone().requires_grad_() for tensor in (q, k, v)
    ]
    dense_prev = (
        []
        if prev_k is None
        else [prev_k.clone().requires_grad_(), prev_v.clone().requires_grad_()]
    )
    loop_prev = (
        []
        if prev_k is None
        else [prev_k.clone().requires_grad_(), prev_v.clone().requires_grad_()]
    )
    dense_y = dense_sliding_chunk_attention_reference(
        *dense_tensors,
        *(dense_prev if dense_prev else [None, None]),
        chunk_size,
        q_segment_idx,
        k_segment_idx,
        scale,
        reset_chunk_pos_per_seq,
    )
    loop_reference = (
        naive_sliding_chunk_attention_reset_chunk_pos_per_seq
        if reset_chunk_pos_per_seq
        else naive_sliding_chunk_attention
    )
    loop_y = loop_reference(
        *loop_tensors,
        *(loop_prev if loop_prev else [None, None]),
        chunk_size,
        q_segment_idx,
        k_segment_idx,
        scale,
    )
    torch.testing.assert_close(loop_y, dense_y, rtol=2e-6, atol=2e-7)

    weight = torch.randn_like(dense_y)
    dense_grads = torch.autograd.grad(
        (dense_y * weight).sum(), dense_tensors + dense_prev
    )
    loop_grads = torch.autograd.grad(
        (loop_y * weight).sum(), loop_tensors + loop_prev
    )
    for loop_grad, dense_grad in zip(loop_grads, dense_grads):
        torch.testing.assert_close(
            loop_grad, dense_grad, rtol=3e-6, atol=3e-7
        )


def requires_xattn_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("xattn_cuda")


def _pack_varlen_from_segments(q, k, v, q_segment_idx, k_segment_idx):
    batch, q_len = q.shape[:2]
    q_runs, k_runs = _reset_runs(
        q_segment_idx, k_segment_idx, batch, q_len, k.shape[1]
    )
    q_parts = []
    k_parts = []
    v_parts = []
    q_lengths = []
    k_lengths = []
    position_offsets = []
    q_dense_indices = []
    k_dense_indices = []
    for q_run, k_run in zip(q_runs, k_runs):
        q_batch, q_start, q_run_len, _ = q_run
        k_batch, k_start, k_run_len, _ = k_run
        q_parts.append(q[q_batch, q_start : q_start + q_run_len])
        k_parts.append(k[k_batch, k_start : k_start + k_run_len])
        v_parts.append(v[k_batch, k_start : k_start + k_run_len])
        q_lengths.append(q_run_len)
        k_lengths.append(k_run_len)
        position_offsets.append(q_start)
        q_dense_indices.extend(
            q_batch * q_len + pos
            for pos in range(q_start, q_start + q_run_len)
        )
        k_dense_indices.extend(
            k_batch * k.shape[1] + pos
            for pos in range(k_start, k_start + k_run_len)
        )

    def cumulative_lengths(lengths):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device=q.device)

    return {
        "q": torch.cat(q_parts, dim=0).contiguous(),
        "k": torch.cat(k_parts, dim=0).contiguous(),
        "v": torch.cat(v_parts, dim=0).contiguous(),
        "cu_q": cumulative_lengths(q_lengths),
        "cu_k": cumulative_lengths(k_lengths),
        "max_q": max(q_lengths),
        "max_k": max(k_lengths),
        "position_offsets": torch.tensor(
            position_offsets, dtype=torch.int32, device=q.device
        ),
        "q_dense_indices": torch.tensor(
            q_dense_indices, dtype=torch.long, device=q.device
        ),
        "k_dense_indices": torch.tensor(
            k_dense_indices, dtype=torch.long, device=q.device
        ),
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "qk_dim,v_dim",
    [
        (32, 64),
        (32, 192),
        (48, 176),
        (64, 192),
        (128, 192),
        (96, 160),
        (96, 256),
        (160, 96),
        (160, 160),
        (160, 256),
        (256, 160),
        (256, 192),
    ],
)
@pytest.mark.parametrize("reset_chunk_pos_per_seq", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_flash_sca_routes_match_reference(
    dtype, qk_dim, v_dim, reset_chunk_pos_per_seq, deterministic
):
    requires_xattn_cuda()
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip("the three-route FlashSCA test requires the SM90 backend")

    torch.manual_seed(4)
    batch, seq_len, heads, chunk_size = 2, 67, 2, 32
    scale = 1.0 / math.sqrt(qk_dim)
    q = F.normalize(
        torch.randn(
            batch, seq_len, heads, qk_dim, dtype=dtype, device="cuda"
        ),
        dim=-1,
    )
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = F.silu(
        torch.randn(
            batch, seq_len, heads, v_dim, dtype=dtype, device="cuda"
        )
    )
    bos = torch.zeros(batch, seq_len, dtype=torch.bool, device="cuda")
    bos[1, 0] = True
    bos[0, [11, 44]] = True
    bos[1, [7, 39]] = True
    segment_idx = torch.cumsum(bos, dim=-1)

    q_ref = q.detach().double().requires_grad_(True)
    k_ref = k.detach().double().requires_grad_(True)
    v_ref = v.detach().double().requires_grad_(True)
    y_ref = dense_sliding_chunk_attention_reference(
        q_ref,
        k_ref,
        v_ref,
        None,
        None,
        chunk_size,
        segment_idx,
        segment_idx,
        scale,
        reset_chunk_pos_per_seq,
    )

    dense_tensors = [
        x.detach().clone().requires_grad_(True) for x in (q, k, v)
    ]
    y_dense = flash_sca(
        *dense_tensors,
        chunk_size=chunk_size,
        scale=scale,
        segment_idx=segment_idx,
        reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        deterministic=deterministic,
        backend="sm90",
    )
    assert not bos[0, 0]
    packed = _pack_varlen_from_segments(
        q, k, v, segment_idx, segment_idx
    )
    packed_tensors = [
        packed[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v")
    ]
    varlen_tensors = [
        tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)
    ]
    bos_tensors = [
        tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)
    ]
    y_bos = flash_sca(
        *bos_tensors,
        chunk_size=chunk_size,
        scale=scale,
        bos_mask=bos,
        reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        deterministic=deterministic,
        backend="sm90",
    )
    y_varlen = _flash_sca(
        *varlen_tensors,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens_q=packed["cu_q"],
        cu_seqlens_k=packed["cu_k"],
        max_seqlen_q=packed["max_q"],
        max_seqlen_k=packed["max_k"],
        position_offsets=(
            None
            if reset_chunk_pos_per_seq
            else packed["position_offsets"]
        ),
        reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        deterministic=deterministic,
        backend="sm90",
    )
    y_packed = _flash_sca_varlen(
        *packed_tensors,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens_q=packed["cu_q"],
        cu_seqlens_k=packed["cu_k"],
        max_seqlen_q=packed["max_q"],
        max_seqlen_k=packed["max_k"],
        position_offsets=(
            None
            if reset_chunk_pos_per_seq
            else packed["position_offsets"]
        ),
        reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        deterministic=deterministic,
        backend="sm90",
    )

    expected = y_ref.to(dtype)
    torch.testing.assert_close(
        y_dense, expected, rtol=RTOL[dtype], atol=ATOL[dtype]
    )
    torch.testing.assert_close(
        y_bos, expected, rtol=RTOL[dtype], atol=ATOL[dtype]
    )
    torch.testing.assert_close(
        y_varlen, expected, rtol=RTOL[dtype], atol=ATOL[dtype]
    )
    torch.testing.assert_close(
        y_packed.reshape_as(expected),
        expected,
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
    )

    weight = torch.randn_like(y_ref) / math.sqrt(seq_len)
    y_ref.backward(weight)
    y_dense.backward(weight.to(dtype))
    y_bos.backward(weight.to(dtype))
    y_varlen.backward(weight.to(dtype))
    y_packed.backward(weight.to(dtype).reshape_as(y_packed))
    reference_grads = (q_ref.grad, k_ref.grad, v_ref.grad)
    for route_tensors in (dense_tensors, bos_tensors, varlen_tensors):
        for actual, expected_grad in zip(route_tensors, reference_grads):
            torch.testing.assert_close(
                actual.grad,
                expected_grad.to(dtype),
                rtol=RTOL[dtype],
                atol=GRAD_ATOL[dtype],
            )
    for actual, expected_grad in zip(packed_tensors, reference_grads):
        torch.testing.assert_close(
            actual.grad.reshape_as(expected_grad),
            expected_grad.to(dtype),
            rtol=RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )


def test_flash_sca_high_precision_varlen_matches_fp64_reference():
    requires_xattn_cuda()
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the high-precision varlen test requires SM90")

    torch.manual_seed(20260827)
    dtype = torch.bfloat16
    batch, seq_len, heads = 1, 67, 2
    qk_dim = v_dim = 64
    chunk_size = 32
    scale = qk_dim**-0.5
    with torch.no_grad():
        q = F.normalize(
            torch.randn(
                batch,
                seq_len,
                heads,
                qk_dim,
                dtype=dtype,
                device="cuda",
            ),
            dim=-1,
        )
        k = F.normalize(torch.randn_like(q), dim=-1)
        v = F.silu(torch.randn_like(q)) + 0.1
    bos = torch.zeros(batch, seq_len, dtype=torch.bool, device="cuda")
    bos[0, [0, 11, 44]] = True
    segment_idx = bos.cumsum(dim=-1)

    q_ref = q.detach().double().requires_grad_(True)
    k_ref = k.detach().double().requires_grad_(True)
    v_ref = v.detach().double().requires_grad_(True)
    y_ref = dense_sliding_chunk_attention_reference(
        q_ref,
        k_ref,
        v_ref,
        None,
        None,
        chunk_size,
        segment_idx,
        segment_idx,
        scale,
        False,
    )
    packed = _pack_varlen_from_segments(
        q, k, v, segment_idx, segment_idx
    )
    actual_inputs = [
        packed[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v")
    ]
    actual = _flash_sca_varlen(
        *actual_inputs,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens_q=packed["cu_q"],
        cu_seqlens_k=packed["cu_k"],
        max_seqlen_q=packed["max_q"],
        max_seqlen_k=packed["max_k"],
        position_offsets=packed["position_offsets"],
        backend="sm90",
        deterministic=True,
        high_precision_output=True,
    )
    expected = y_ref.flatten(0, 1).index_select(
        0, packed["q_dense_indices"]
    )
    torch.testing.assert_close(
        actual,
        expected.to(dtype),
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
    )

    weight = torch.randn_like(actual) / math.sqrt(actual.shape[0])
    actual_grads = torch.autograd.grad(actual, actual_inputs, weight)
    dense_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), weight.double()
    )
    expected_grads = (
        dense_grads[0].flatten(0, 1).index_select(
            0, packed["q_dense_indices"]
        ),
        dense_grads[1].flatten(0, 1).index_select(
            0, packed["k_dense_indices"]
        ),
        dense_grads[2].flatten(0, 1).index_select(
            0, packed["k_dense_indices"]
        ),
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad,
            expected_grad.to(dtype),
            rtol=RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reset_chunk_pos_per_seq", [False, True])
def test_flash_sca_varlen_previous_chunk_continuation(
    dtype, reset_chunk_pos_per_seq
):
    requires_xattn_cuda()
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip("the varlen previous-chunk test requires the SM90 backend")

    torch.manual_seed(5)
    batch, seq_len, heads, qk_dim, v_dim, chunk_size = 2, 67, 1, 32, 64, 32
    scale = 1.0 / math.sqrt(qk_dim)
    q, k, v, prev_k, prev_v, q_segment_idx, k_segment_idx = make_inputs(
        batch,
        seq_len,
        heads,
        qk_dim,
        v_dim,
        chunk_size,
        True,
        dtype,
        prev_continuation=True,
    )
    combined_k = torch.cat([prev_k, k], dim=1)
    combined_v = torch.cat([prev_v, v], dim=1)
    packed = _pack_varlen_from_segments(
        q,
        combined_k,
        combined_v,
        q_segment_idx,
        k_segment_idx,
    )
    assert torch.any(
        (packed["cu_k"][1:] - packed["cu_k"][:-1])
        > (packed["cu_q"][1:] - packed["cu_q"][:-1])
    )

    q_ref = q.detach().double().requires_grad_(True)
    k_ref = k.detach().double().requires_grad_(True)
    v_ref = v.detach().double().requires_grad_(True)
    prev_k_ref = prev_k.detach().double().requires_grad_(True)
    prev_v_ref = prev_v.detach().double().requires_grad_(True)
    y_ref = dense_sliding_chunk_attention_reference(
        q_ref,
        k_ref,
        v_ref,
        prev_k_ref,
        prev_v_ref,
        chunk_size,
        q_segment_idx,
        k_segment_idx,
        scale,
        reset_chunk_pos_per_seq,
    )

    packed_tensors = [
        packed[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v")
    ]
    y_varlen = _flash_sca_varlen(
        *packed_tensors,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens_q=packed["cu_q"],
        cu_seqlens_k=packed["cu_k"],
        max_seqlen_q=packed["max_q"],
        max_seqlen_k=packed["max_k"],
        position_offsets=(
            None
            if reset_chunk_pos_per_seq
            else packed["position_offsets"]
        ),
        reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        backend="sm90",
    )

    q_index = packed["q_dense_indices"]
    k_index = packed["k_dense_indices"]
    expected_y = y_ref.flatten(0, 1).index_select(0, q_index)
    torch.testing.assert_close(
        y_varlen,
        expected_y.to(dtype),
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
    )

    weight = torch.randn_like(y_ref) / math.sqrt(seq_len)
    y_ref.backward(weight)
    y_varlen.backward(
        weight.flatten(0, 1).index_select(0, q_index).to(dtype)
    )
    combined_k_grad = torch.cat([prev_k_ref.grad, k_ref.grad], dim=1)
    combined_v_grad = torch.cat([prev_v_ref.grad, v_ref.grad], dim=1)
    expected_grads = (
        q_ref.grad.flatten(0, 1).index_select(0, q_index),
        combined_k_grad.flatten(0, 1).index_select(0, k_index),
        combined_v_grad.flatten(0, 1).index_select(0, k_index),
    )
    for actual, expected_grad in zip(packed_tensors, expected_grads):
        torch.testing.assert_close(
            actual.grad,
            expected_grad.to(dtype),
            rtol=RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "prev_chunk,prev_continuation",
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize(
    "batch,seq_len,heads,qk_dim,v_dim,chunk_size",
    [
        (1, 64, 2, 32, 32, 32),
        (1, 96, 1, 64, 32, 32),
        (2, 65, 1, 32, 64, 32),
    ],
)
def test_flash_sca_global_chunk_positions(
    batch,
    seq_len,
    heads,
    qk_dim,
    v_dim,
    chunk_size,
    prev_chunk,
    prev_continuation,
    dtype,
):
    requires_xattn_cuda()
    torch.manual_seed(1)
    scale = 1.0 / math.sqrt(qk_dim)
    q, k, v, prev_k, prev_v, q_segment_idx, k_segment_idx = make_inputs(
        batch,
        seq_len,
        heads,
        qk_dim,
        v_dim,
        chunk_size,
        prev_chunk,
        dtype,
        prev_continuation,
    )
    if not flash_sca_sm90_available(q):
        pytest.skip("FlashSCA SM90 backend is not available in this build/device")

    q_ref = q.detach().double().requires_grad_(True)
    k_ref = k.detach().double().requires_grad_(True)
    v_ref = v.detach().double().requires_grad_(True)
    prev_k_ref = None if prev_k is None else prev_k.detach().double().requires_grad_(True)
    prev_v_ref = None if prev_v is None else prev_v.detach().double().requires_grad_(True)
    y_ref = dense_sliding_chunk_attention_reference(
        q_ref,
        k_ref,
        v_ref,
        prev_k_ref,
        prev_v_ref,
        chunk_size,
        q_segment_idx,
        k_segment_idx,
        scale,
        False,
    )

    y, _, lse = flash_sca_fwd(
        q,
        k,
        v,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        segment_idx=k_segment_idx,
        backend="auto",
        reset_chunk_pos_per_seq=False,
    )
    torch.testing.assert_close(y, y_ref.to(dtype), rtol=RTOL[dtype], atol=ATOL[dtype])

    weight = torch.randn_like(y_ref) / math.sqrt(seq_len)
    y_ref.backward(weight)
    grads = flash_sca_bwd(
        weight.to(dtype),
        q,
        k,
        v,
        y,
        lse,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        segment_idx=k_segment_idx,
        deterministic=False,
        backend="auto",
        reset_chunk_pos_per_seq=False,
    )
    q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad = grads
    torch.testing.assert_close(q_grad, q_ref.grad.to(dtype), rtol=RTOL[dtype], atol=GRAD_ATOL[dtype])
    torch.testing.assert_close(k_grad, k_ref.grad.to(dtype), rtol=RTOL[dtype], atol=GRAD_ATOL[dtype])
    torch.testing.assert_close(v_grad, v_ref.grad.to(dtype), rtol=RTOL[dtype], atol=GRAD_ATOL[dtype])
    if prev_chunk:
        torch.testing.assert_close(prev_k_grad, prev_k_ref.grad.to(dtype), rtol=RTOL[dtype], atol=GRAD_ATOL[dtype])
        torch.testing.assert_close(prev_v_grad, prev_v_ref.grad.to(dtype), rtol=RTOL[dtype], atol=GRAD_ATOL[dtype])
    else:
        assert prev_k_grad is None
        assert prev_v_grad is None


from tests._numerics import Case, DTYPES, HEAD_DIMS, MATRIX_LENGTHS, check_dimensions, cases_for, check_case, check_packed, check_rounding_bias


@pytest.mark.parametrize('case', cases_for('chunk'), ids=lambda case: case.name)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('deterministic,high_precision', ((False, False), (True, True)))
def test_output_and_gradients(case, dtype, kv_heads, deterministic, high_precision):
    check_case('chunk', case, dtype, kv_heads, deterministic, high_precision)


@pytest.mark.full_matrix
@pytest.mark.parametrize('length', MATRIX_LENGTHS)
@pytest.mark.parametrize('dim', HEAD_DIMS)
@pytest.mark.parametrize('value_dim', HEAD_DIMS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1), ids=('mha', 'gqa', 'mqa'))
@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('high_precision', (False, True))
def test_full_dimensions(length, dim, value_dim, dtype, kv_heads, deterministic, high_precision):
    check_dimensions('chunk', length, dim, value_dim, dtype, kv_heads,
                     deterministic, high_precision)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_heads', (4, 2, 1))
@pytest.mark.parametrize('deterministic', (False, True))
def test_packed_output_and_gradients(dtype, kv_heads, deterministic):
    check_packed('chunk', dtype, kv_heads, deterministic)


@pytest.mark.parametrize('deterministic', (False, True))
def test_high_precision_gradient_rounding(deterministic):
    check_rounding_bias('chunk', deterministic)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('previous', (0, 32))
def test_generic_cuda_output_and_gradients(dtype, previous):
    check_case('chunk', Case('generic', 65, 65, 32, 64, previous=previous, span=32),
               dtype, 4, False, False, backend='cuda')

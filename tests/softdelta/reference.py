from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor


def _contiguous_runs(values) -> Dict[int, Tuple[int, int]]:
    result = {}
    start = 0
    for position in range(1, len(values) + 1):
        if position == len(values) or values[position] != values[start]:
            segment = values[start]
            assert segment not in result
            result[segment] = (start, position)
            start = position
    return result


def _allowed_masks(
    q_length: int,
    k_length: int,
    batch: int,
    device: torch.device,
    visibility: str,
    span: Optional[int],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    reset_position_per_segment: bool,
) -> Tuple[Tensor, Tensor]:
    if q_segment_idx is None and not reset_position_per_segment:
        q_position = torch.arange(q_length, device=device)[:, None]
        q_position = q_position + k_length - q_length
        k_position = torch.arange(k_length, device=device)[None, :]
        if visibility == "full":
            left = torch.ones((q_length, k_length), device=device, dtype=torch.bool)
        elif visibility == "window":
            assert span is not None
            left = k_position >= q_position - span
        elif visibility == "chunk":
            assert span is not None
            left = k_position >= (q_position // span) * span - span
        else:
            raise AssertionError(visibility)
        return (
            (left & (k_position <= q_position)).expand(batch, -1, -1),
            (left & (k_position < q_position)).expand(batch, -1, -1),
        )

    inclusive = torch.zeros(
        batch, q_length, k_length, dtype=torch.bool, device=device
    )
    strict_past = torch.zeros_like(inclusive)

    for batch_idx in range(batch):
        if q_segment_idx is None:
            q_segments = [batch_idx] * q_length
            k_segments = [batch_idx] * k_length
        else:
            assert k_segment_idx is not None
            q_segments = q_segment_idx[batch_idx].detach().cpu().tolist()
            k_segments = k_segment_idx[batch_idx].detach().cpu().tolist()

        if reset_position_per_segment:
            q_runs = _contiguous_runs(q_segments)
            k_runs = _contiguous_runs(k_segments)
        else:
            q_runs = k_runs = {}

        for q_index in range(q_length):
            q_segment = q_segments[q_index]
            if reset_position_per_segment:
                q_start, q_end = q_runs[q_segment]
                k_start, k_end = k_runs.get(q_segment, (0, 0))
                q_position = (
                    q_index
                    - q_start
                    + (k_end - k_start)
                    - (q_end - q_start)
                )
            else:
                q_position = q_index + k_length - q_length

            for k_index in range(k_length):
                if k_segments[k_index] != q_segment:
                    continue
                if reset_position_per_segment:
                    k_position = k_index - k_runs[q_segment][0]
                else:
                    k_position = k_index

                if visibility == "full":
                    in_left_range = True
                elif visibility == "window":
                    assert span is not None
                    in_left_range = k_position >= q_position - span
                elif visibility == "chunk":
                    assert span is not None
                    chunk_begin = (q_position // span) * span
                    in_left_range = k_position >= chunk_begin - span
                else:
                    raise AssertionError(visibility)

                if in_left_range and k_position <= q_position:
                    inclusive[batch_idx, q_index, k_index] = True
                if in_left_range and k_position < q_position:
                    strict_past[batch_idx, q_index, k_index] = True
    return inclusive, strict_past


def _attention_from_mask(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allowed: Tensor,
    scale: float,
) -> Tensor:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)

    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    allowed_heads = allowed.unsqueeze(1)
    masked_scores = scores.masked_fill(~allowed_heads, float("-inf"))
    nonempty = allowed.any(dim=-1, keepdim=True).unsqueeze(1)
    scores = torch.where(nonempty, masked_scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = probabilities.masked_fill(~allowed_heads, 0.0)
    return torch.einsum("bhqk,bkhv->bqhv", probabilities, v)


def softdelta_fp64_oracle(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    *,
    visibility: str,
    span: Optional[int] = None,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    reset_position_per_segment: bool = False,
    return_aux: bool = False,
):
    """Independent FP64 oracle over caller-provided quantized values."""
    assert q.dtype == k.dtype == v.dtype == g.dtype == torch.float64
    assert (prev_k is None) == (prev_v is None)
    if prev_k is not None:
        assert prev_v is not None
        k = torch.cat((prev_k, k), dim=1)
        v = torch.cat((prev_v, v), dim=1)

    q1 = q[:, :, 0::2]
    q2 = q[:, :, 1::2]
    flat_v = v.flatten(3)
    resolved_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    inclusive, strict_past = _allowed_masks(
        q.shape[1],
        k.shape[1],
        q.shape[0],
        q.device,
        visibility,
        span,
        q_segment_idx,
        k_segment_idx,
        reset_position_per_segment,
    )
    assert bool(inclusive.any(dim=-1).all())

    read = _attention_from_mask(q1, k, flat_v, inclusive, resolved_scale)
    past = _attention_from_mask(
        q2, k, flat_v, strict_past, resolved_scale
    )
    output_shape = (
        q.shape[0],
        q.shape[1],
        q1.shape[2],
        v.shape[3],
        v.shape[4],
    )
    read = read.reshape(output_shape)
    past = past.reshape(output_shape)
    output = read - torch.sigmoid(g) * past
    if return_aux:
        return output, read, past, inclusive, strict_past
    return output


def softdelta_causal_sampled_rows_fp64_oracle(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    query_indices: Sequence[int],
    *,
    scale: Optional[float] = None,
    return_aux: bool = False,
):
    """FP64 full-causal SoftDelta oracle for selected query rows."""
    assert q.dtype == k.dtype == v.dtype == g.dtype
    assert q.dim() == k.dim() == 4
    assert v.dim() == g.dim() == 5
    assert q.shape[0] == k.shape[0] == v.shape[0] == g.shape[0]
    assert q.shape[1] == k.shape[1] == v.shape[1] == g.shape[1]
    assert q.shape[2] % 2 == 0
    logical_q_heads = q.shape[2] // 2
    kv_heads = k.shape[2]
    assert logical_q_heads % kv_heads == 0
    assert v.shape[2] == kv_heads
    assert g.shape[2] == logical_q_heads
    assert g.shape[3] == v.shape[3] and g.shape[4] == 1

    indices = tuple(int(index) for index in query_indices)
    assert indices
    assert len(indices) == len(set(indices))
    assert all(0 <= index < q.shape[1] for index in indices)
    resolved_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    flat_v = v.flatten(3)
    read = torch.empty(
        q.shape[0],
        len(indices),
        logical_q_heads,
        flat_v.shape[-1],
        dtype=torch.float64,
        device=q.device,
    )
    past = torch.zeros_like(read)
    q_heads_per_kv_head = logical_q_heads // kv_heads

    for batch_index in range(q.shape[0]):
        for q_head in range(logical_q_heads):
            kv_head = q_head // q_heads_per_kv_head
            k_head = k[batch_index, :, kv_head].double()
            v_head = flat_v[batch_index, :, kv_head].double()
            for sampled_row, query_index in enumerate(indices):
                primary_scores = torch.mv(
                    k_head[: query_index + 1],
                    q[batch_index, query_index, 2 * q_head].double(),
                ) * resolved_scale
                primary_probability = torch.softmax(primary_scores, dim=0)
                read[batch_index, sampled_row, q_head] = torch.mv(
                    v_head[: query_index + 1].transpose(0, 1),
                    primary_probability,
                )
                if query_index > 0:
                    past_scores = torch.mv(
                        k_head[:query_index],
                        q[batch_index, query_index, 2 * q_head + 1].double(),
                    ) * resolved_scale
                    past_probability = torch.softmax(past_scores, dim=0)
                    past[batch_index, sampled_row, q_head] = torch.mv(
                        v_head[:query_index].transpose(0, 1),
                        past_probability,
                    )

    output_shape = (
        q.shape[0],
        len(indices),
        logical_q_heads,
        v.shape[3],
        v.shape[4],
    )
    read = read.reshape(output_shape)
    past = past.reshape(output_shape)
    index_tensor = torch.tensor(indices, device=g.device, dtype=torch.long)
    sampled_gate = g.index_select(1, index_tensor).double()
    output = read - torch.sigmoid(sampled_gate) * past
    if return_aux:
        return output, read, past
    return output

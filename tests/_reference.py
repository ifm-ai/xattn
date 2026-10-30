"""FP64 attention masks and reads over quantized inputs, and the shared tolerances."""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

ATOL = {
    torch.float16: 2e-4,
    torch.bfloat16: 1e-3,
}

GRAD_ATOL = {
    torch.float16: 2e-4,
    torch.bfloat16: 2e-3,
}

RTOL = {
    torch.float16: 1e-3,
    torch.bfloat16: 1e-2,
}


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

    # Keep the independent literal mask oracle, but build its bits on the CPU.
    # Per-element CUDA assignments turn a small reference into thousands of
    # kernel launches and dominate comprehensive correctness sweeps.
    count = batch * q_length * k_length
    if count == 0:
        empty = torch.zeros(batch, q_length, k_length, dtype=torch.bool, device=device)
        return empty, torch.zeros_like(empty)
    inclusive_bits = bytearray(count)
    strict_bits = bytearray(count)

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
                    inclusive_bits[(batch_idx * q_length + q_index) * k_length + k_index] = 1
                if in_left_range and k_position < q_position:
                    strict_bits[(batch_idx * q_length + q_index) * k_length + k_index] = 1
    return (
        torch.frombuffer(inclusive_bits, dtype=torch.bool).reshape(batch, q_length, k_length).to(device),
        torch.frombuffer(strict_bits, dtype=torch.bool).reshape(batch, q_length, k_length).to(device),
    )


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

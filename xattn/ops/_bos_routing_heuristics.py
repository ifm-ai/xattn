"""BOS routing from host metadata; True selects packed varlen.

Forward and backward use separate thresholds for run lengths, attention
visibility, head sharing, and determinism.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor


def _validate_bos_plan(num_runs: int, max_seqlen: int) -> None:
    if num_runs <= 0:
        raise ValueError("BOS metadata must contain at least one run")
    if max_seqlen <= 0:
        raise ValueError("BOS metadata must contain a positive run length")


def _select_routes(
    q: Tensor, k: Tensor, v: Tensor, span: int,
    num_runs: int, max_seqlen: int, deterministic: bool, needs_backward: bool,
    *, sliding_chunk: bool = False, full_visibility: bool = False,
    softdelta: bool = False, reset_chunk_pos_per_seq: bool = False,
    has_previous: bool = False, high_precision_output: bool = False,
    previous_length: int = 0, num_k_runs: Optional[int] = None, standalone: bool = False,
    max_seqlen_k: Optional[int] = None,
) -> Tuple[bool, bool]:
    _validate_bos_plan(num_runs, max_seqlen)
    batch, length, heads, d = q.shape
    kv_heads, value_dim = k.shape[2], v.shape[-1]
    if q.dtype not in (torch.float16, torch.bfloat16) or not (
        0 < d <= 256 and 0 < value_dim <= 256
    ):
        return True, True
    # Prefix and short-query routes require independent K statistics.
    if (has_previous or length != k.shape[1]) and (
        num_k_runs is None or max_seqlen_k is None
    ):
        return True, True
    num_k_runs = num_runs if num_k_runs is None else num_k_runs
    max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k
    _validate_bos_plan(num_k_runs, max_seqlen_k)
    # Batched full/window packed metadata requires matching Q/K run counts.
    if not sliding_chunk and batch > 1 and num_k_runs != num_runs:
        return False, False
    tokens = batch * length
    k_length = k.shape[1] + previous_length
    span = k_length if full_visibility else max(1, span)
    deterministic = deterministic and needs_backward
    # Packed route for intermediate short queries.
    if length != k.shape[1] and length != 1:
        return True, True
    if length == 1:
        # Packed route for standalone decode.
        if standalone:
            return True, True
        if max_seqlen_k <= 2048:
            return (False, False)
        else:
            if d <= 64:
                return (True, True)
            else:
                return (True, False) if needs_backward else (True, True)
    # Forward routing thresholds.
    if tokens <= 32 * (num_runs):
        if max_seqlen <= 128:
            fwd = False
        else:
            if d <= 128:
                fwd = True
            else:
                fwd = False
    else:
        fwd = True
    if not needs_backward:
        return fwd, fwd
    # Segment backward for long deterministic local-attention runs.
    if (deterministic and not full_visibility
            and tokens >= max(8192, 2 * span) * num_runs):
        return fwd, False
    # Segment backward for short deterministic runs with high head sharing.
    if (deterministic and d > 64 and batch * heads > 8
            and heads >= 8 * kv_heads and tokens <= 128 * num_runs):
        return fwd, False
    # Packed backward for moderate deterministic runs.
    if deterministic and max_seqlen > 128 and (32 if d > 128 else 16) * num_runs < tokens <= 512 * num_runs:
        return fwd, True
    # Packed backward for narrow Q/K with wider V.
    if not deterministic and d <= 64 and value_dim > d and tokens > 8 * num_runs:
        return fwd, True
    if not (deterministic):
        if tokens <= 16 * (num_runs):
            return fwd, False
        else:
            if d <= 128:
                return fwd, True
            else:
                if tokens <= 32 * (num_runs):
                    return fwd, False
                else:
                    return fwd, True
    else:
        if d <= 64:
            if tokens <= 8 * (num_runs):
                return fwd, False
            else:
                return fwd, True
        else:
            if tokens <= 8192:
                if max_seqlen <= 128:
                    return fwd, False
                else:
                    return fwd, True
            else:
                return fwd, False


def select_flash_swa_bos_routes(
    q: Tensor, k: Tensor, v: Tensor, window_size: int,
    num_runs: int, max_seqlen: int, deterministic: bool, needs_backward: bool,
    *, full_visibility: bool = False, has_previous: bool = False,
    high_precision_output: bool = False, previous_length: int = 0,
    num_k_runs: Optional[int] = None, max_seqlen_k: Optional[int] = None,
) -> Tuple[bool, bool]:
    """Return (FWD varlen, BWD varlen) for causal full or sliding-window BOS."""
    if window_size < 0:
        raise ValueError("window_size must be nonnegative")
    return _select_routes(
        q, k, v, window_size + 1, num_runs, max_seqlen,
        deterministic, needs_backward, full_visibility=full_visibility,
        has_previous=has_previous, high_precision_output=high_precision_output,
        previous_length=previous_length, num_k_runs=num_k_runs,
        max_seqlen_k=max_seqlen_k,
    )


def select_flash_sca_bos_routes(
    q: Tensor, k: Tensor, v: Tensor, chunk_size: int,
    num_runs: int, max_seqlen: int, deterministic: bool, needs_backward: bool,
    reset_chunk_pos_per_seq: bool = False, has_previous: bool = False,
    *, high_precision_output: bool = False, previous_length: int = 0,
    num_k_runs: Optional[int] = None, max_seqlen_k: Optional[int] = None, standalone: bool = False,
) -> Tuple[bool, bool]:
    """Return (FWD varlen, BWD varlen) for sliding-chunk BOS."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return _select_routes(
        q, k, v, chunk_size, num_runs, max_seqlen, deterministic, needs_backward,
        sliding_chunk=True, reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        has_previous=has_previous, high_precision_output=high_precision_output,
        previous_length=previous_length, num_k_runs=num_k_runs,
        max_seqlen_k=max_seqlen_k, standalone=standalone,
    )


def select_flash_softdelta_bos_routes(
    q: Tensor, k: Tensor, v: Tensor, span: int,
    num_runs: int, max_seqlen: int, deterministic: bool, needs_backward: bool,
    *, sliding_chunk: bool, full_visibility: bool = False,
    reset_chunk_pos_per_seq: bool = False, has_previous: bool = False,
    high_precision_output: bool = False, previous_length: int = 0,
    num_k_runs: Optional[int] = None, max_seqlen_k: Optional[int] = None,
) -> Tuple[bool, bool]:
    """Return joint reader routes; Q heads and V width are logical/flattened."""
    if span < 0 or (sliding_chunk and span == 0):
        raise ValueError("invalid attention span")
    return _select_routes(
        q, k, v, span if sliding_chunk else span + 1,
        num_runs, max_seqlen, deterministic, needs_backward,
        sliding_chunk=sliding_chunk, full_visibility=full_visibility,
        softdelta=True, reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        has_previous=has_previous, high_precision_output=high_precision_output,
        previous_length=previous_length, num_k_runs=num_k_runs,
        max_seqlen_k=max_seqlen_k,
    )

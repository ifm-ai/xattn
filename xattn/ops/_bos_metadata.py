from typing import Any, Optional, Tuple

import torch
from torch import Tensor


def validate_bos_mask(
    bos_mask: Tensor,
    reference: Tensor,
    expected_length: int,
) -> None:
    expected_shape = (reference.shape[0], expected_length)
    if (
        bos_mask.device != reference.device
        or bos_mask.dtype != torch.bool
        or bos_mask.dim() != 2
        or tuple(bos_mask.shape) != expected_shape
    ):
        raise ValueError(
            "bos_mask must be a bool tensor on the Q/K/V device with shape "
            f"{list(expected_shape)}"
        )


def _bos_plan(ops: Any, bos_mask: Tensor) -> Tuple[Tensor, int, int]:
    plan = getattr(ops, "_flash_sca_sm90_bos_plan", None)
    if plan is not None:
        cu_seqlens, max_seqlen, num_runs = plan(bos_mask)
        return cu_seqlens, int(max_seqlen), int(num_runs)
    fallback = getattr(ops, "_flash_sca_sm90_bos_to_cu_seqlens", None)
    if fallback is None:
        raise RuntimeError(
            "BOS-mask attention requires an extension built with SM90 support"
        )
    cu_seqlens, max_seqlen = fallback(bos_mask)
    return cu_seqlens, int(max_seqlen), cu_seqlens.numel() - 1


def build_bos_metadata(
    ops: Any,
    q: Tensor,
    bos_mask: Tensor,
    k_length: int,
) -> Tuple[Tensor, Tensor, int, int, int, int, bool]:
    if q.shape[1] != k_length:
        paired_plan = getattr(
            ops, "_flash_sca_sm90_paired_bos_plan", None
        )
        if paired_plan is not None:
            result = paired_plan(bos_mask, q.shape[1])
            cu_q, cu_k, max_q, max_k, num_q, num_k = result
            return (
                cu_q,
                cu_k,
                int(max_q),
                int(max_k),
                int(num_q),
                int(num_k),
                int(num_q) == q.shape[0]
                and int(num_k) == q.shape[0],
            )

    q_bos = (
        bos_mask
        if q.shape[1] == k_length
        else bos_mask[:, k_length - q.shape[1] :]
    )
    cu_q, max_q, num_q = _bos_plan(ops, q_bos)
    if q_bos is bos_mask:
        cu_k, max_k, num_k = cu_q, max_q, num_q
    else:
        cu_k, max_k, num_k = _bos_plan(ops, bos_mask)
    return (
        cu_q,
        cu_k,
        max_q,
        max_k,
        num_q,
        num_k,
        num_q == q.shape[0] and num_k == q.shape[0],
    )


def bos_to_segment_idx(ops: Any, bos_mask: Tensor) -> Tensor:
    convert = getattr(ops, "_flash_sca_sm90_bos_to_segment_idx", None)
    if convert is None:
        raise RuntimeError(
            "BOS-mask attention requires an extension built with SM90 support"
        )
    return convert(bos_mask)


def requires_row_aligned_k_runs(
    q: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
) -> bool:
    return q.shape[0] > 1 and cu_seqlens_k.numel() != cu_seqlens_q.numel()


def pack_batched_qkv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    if prev_k is None:
        return q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
    assert prev_v is not None
    return (
        q.flatten(0, 1),
        torch.cat((prev_k, k), dim=1).flatten(0, 1),
        torch.cat((prev_v, v), dim=1).flatten(0, 1),
    )


def restore_batched_output(y: Tensor, q: Tensor, v: Tensor) -> Tensor:
    return y.reshape(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])


def restore_batched_lse(lse: Tensor, q: Tensor) -> Tensor:
    return (
        lse.reshape(q.shape[2], q.shape[0], q.shape[1])
        .permute(1, 0, 2)
        .contiguous()
    )


def pack_batched_backward(
    y_grad: Tensor,
    y: Tensor,
    lse: Tensor,
    q: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    return (
        y_grad.flatten(0, 1),
        y.flatten(0, 1),
        lse.permute(1, 0, 2).reshape(q.shape[2], -1).contiguous(),
    )


def unpack_batched_grads(
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    dq = dq.reshape_as(q)
    if prev_k is None:
        return dq, dk.reshape_as(k), dv.reshape_as(v), None, None
    assert prev_v is not None
    dk = dk.reshape(
        q.shape[0], prev_k.shape[1] + k.shape[1], *k.shape[2:]
    )
    dv = dv.reshape(
        q.shape[0], prev_v.shape[1] + v.shape[1], *v.shape[2:]
    )
    prev_length = prev_k.shape[1]
    return (
        dq,
        dk[:, prev_length:].contiguous(),
        dv[:, prev_length:].contiguous(),
        dk[:, :prev_length].contiguous(),
        dv[:, :prev_length].contiguous(),
    )

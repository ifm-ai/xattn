from typing import Optional

import torch
from torch import Tensor

from tests.softdelta.reference import _allowed_masks, _attention_from_mask


def group_softdelta_fp64_oracle(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    neg_q: Optional[Tensor] = None,
    *,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    return_aux: bool = False,
):
    """Independent FP64 oracle over caller-provided quantized values.

    The negative read uses ``neg_q`` (default: the aligned current keys) as a
    per-KV-head query over the strict past; every query head of the group
    subtracts its gated copy.
    """
    tensors = (q, k, v, g) if neg_q is None else (q, k, v, g, neg_q)
    assert all(t.dtype == torch.float64 for t in tensors)
    assert (prev_k is None) == (prev_v is None)
    batch, length, q_heads, _ = q.shape
    if neg_q is None:
        neg_q = k[:, k.shape[1] - length:]
    if prev_k is not None:
        assert prev_v is not None
        k = torch.cat((prev_k, k), dim=1)
        v = torch.cat((prev_v, v), dim=1)
    kv_heads = k.shape[2]
    groups, group_dim = v.shape[3:]
    resolved_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    inclusive, strict_past = _allowed_masks(
        length, k.shape[1], batch, q.device, "full", None,
        q_segment_idx, k_segment_idx, False,
    )
    assert bool(inclusive.any(dim=-1).all())
    flat_v = v.flatten(3)
    read = _attention_from_mask(q, k, flat_v, inclusive, resolved_scale)
    neg_read = _attention_from_mask(neg_q, k, flat_v, strict_past, resolved_scale)
    rep = q_heads // kv_heads
    read = read.reshape(batch, length, kv_heads, rep, groups, group_dim)
    neg_read = neg_read.reshape(batch, length, kv_heads, 1, groups, group_dim)
    gate = torch.sigmoid(g).reshape(batch, length, kv_heads, rep, groups, 1)
    y = (read - gate * neg_read).reshape(batch, length, q_heads, groups, group_dim)
    if return_aux:
        return (
            y,
            read.reshape(batch, length, q_heads, groups, group_dim),
            neg_read.reshape(batch, length, kv_heads, groups, group_dim),
        )
    return y

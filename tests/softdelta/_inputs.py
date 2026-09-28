"""Shared quantized inputs and numerical tolerances."""
import torch
import torch.nn.functional as F

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

def _conditioned_inputs(
    *,
    batch,
    length,
    q_heads,
    kv_heads,
    qk_dim,
    groups,
    group_dim,
    dtype,
    device,
    prev_length=0,
):
    q = torch.randn(
        batch, length, 2 * q_heads, qk_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        batch, length, kv_heads, qk_dim, dtype=dtype, device=device
    )
    v = torch.randn(
        batch,
        length,
        kv_heads,
        groups,
        group_dim,
        dtype=dtype,
        device=device,
    )
    g = torch.randn(
        batch,
        length,
        q_heads,
        groups,
        1,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.silu(v) + 0.1
        g = g * 0.75

    if not prev_length:
        return q, k, v, g, None, None
    prev_k = torch.randn(
        batch,
        prev_length,
        kv_heads,
        qk_dim,
        dtype=dtype,
        device=device,
    )
    prev_v = torch.randn(
        batch,
        prev_length,
        kv_heads,
        groups,
        group_dim,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        prev_k = F.normalize(prev_k, dim=-1)
        prev_v = F.silu(prev_v) + 0.1
    return q, k, v, g, prev_k, prev_v

"""Any batch the device memory holds: a large-batch call must match the same inputs run in batch chunks.

The 66000-row cases pass the 65535 grid-dimension limit (kernels stride over batches); the B64 x 4K case is a
long-context training shape. Forward and backward of full, SWA, SCA and SoftDelta, including deterministic,
and high-precision SM90 paths.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

import xattn

SHAPES = {
    # name: (batch, length, q_heads, kv_heads, chunks, free GiB needed)
    "b66000-l64": (66000, 64, 2, 1, 2, 40),
    # B * Hkv past 65535 with B below it (precision-level repair grids).
    "b16500-l64-gqa": (16500, 64, 8, 4, 2, 40),
    "b64-l4096": (64, 4096, 16, 4, 8, 40),
}


def _free_bytes():
    """Free device memory, counting this process's cached but unallocated blocks."""
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


def _requires_sm90():
    if importlib.util.find_spec("xattn_cuda") is None:
        pytest.skip("xattn CUDA extension is not built")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires an SM90 GPU")


def _ops():
    """name -> (call(q, k, v, g, k_read, **options), uses_gate)."""
    return {
        "full": (lambda q, k, v, g, k_read, **o: xattn.causal_flash_attn(q, k, v, **o), False),
        "swa": (lambda q, k, v, g, k_read, **o: xattn.flash_swa(q, k, v, 96, **o), False),
        "sca": (lambda q, k, v, g, k_read, **o: xattn.flash_sca(q, k, v, 32, **o), False),
        "softdelta": (lambda q, k, v, g, k_read, **o: xattn.softdelta_attention(q, k, k_read, v, g, **o), True),
        "swa-softdelta": (lambda q, k, v, g, k_read, **o: xattn.sliding_window_softdelta_attention(q, k, k_read, v, g, 96, **o),
                          True),
        "sca-softdelta": (lambda q, k, v, g, k_read, **o: xattn.sliding_chunk_softdelta_attention(q, k, k_read, v, g, 32, **o),
                          True),
    }


MODES = {
    "level0": dict(),
    "deterministic": dict(deterministic=True),
    "level1": dict(high_precision_level=1),
    "deterministic-level1": dict(deterministic=True, high_precision_level=1),
}


def _relative(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


def _cases():
    for shape in SHAPES:
        for op in _ops():
            for mode in MODES:
                if shape == "b64-l4096" and mode != "level0":
                    continue
                if shape == "b16500-l64-gqa" and mode in ("level0", "deterministic"):
                    continue
                yield pytest.param(shape, op, mode, False, id=f"{shape}-{op}-{mode}")
                if op == "softdelta" and mode == "level0":
                    yield pytest.param(shape, op, mode, True, id=f"{shape}-{op}-{mode}-k-read")


@pytest.mark.parametrize("shape,op,mode,with_k_read", list(_cases()))
def test_large_batch_matches_chunked(shape, op, mode, with_k_read):
    _requires_sm90()
    batch, length, q_heads, kv_heads, chunks, gib = SHAPES[shape]
    if _free_bytes() < gib * 2**30:
        pytest.skip(f"needs {gib} GiB of free device memory")
    call, uses_gate = _ops()[op]
    options = MODES[mode]
    torch.manual_seed(31)
    dtype, dim = torch.bfloat16, 128
    q = F.normalize(torch.randn(batch, length, q_heads, dim, device="cuda", dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(batch, length, kv_heads, dim, device="cuda", dtype=dtype), dim=-1)
    v = torch.randn(batch, length, kv_heads, dim, device="cuda", dtype=dtype)
    g = torch.randn(batch, length, q_heads, device="cuda", dtype=dtype) if uses_gate else None
    k_read = F.normalize(torch.randn_like(k), dim=-1) if with_k_read else None
    tensors = dict(q=q, k=k, v=v, g=g, k_read=k_read)
    names = [name for name, tensor in tensors.items() if tensor is not None]
    output_grad = torch.randn(batch, length, q_heads, dim, device="cuda", dtype=dtype)

    def run(rows):
        inputs = {name: tensors[name][rows].detach().clone().requires_grad_(True) for name in names}
        out = call(inputs["q"], inputs["k"], inputs["v"], inputs.get("g"), inputs.get("k_read"), **options)
        grads = torch.autograd.grad(out, list(inputs.values()), output_grad[rows])
        return [out.detach()] + list(grads)

    full = run(slice(None))
    step = batch // chunks
    parts = [run(slice(i, i + step)) for i in range(0, batch, step)]
    tail = slice(batch - batch // 64, None)  # includes every batch past the 65535 grid limit
    for index, name in enumerate(["out"] + names):
        expected = torch.cat([p[index] for p in parts])
        if name == "out" and uses_gate and mode == "level0":
            torch.testing.assert_close(full[index], expected, rtol=0, atol=0)
        for label, rows in (("all", slice(None)), ("tail", tail)):
            error = _relative(full[index][rows], expected[rows])
            assert error < 2e-3, f"{shape} {op} {mode} {name} ({label}): relative error {error:.2e}"

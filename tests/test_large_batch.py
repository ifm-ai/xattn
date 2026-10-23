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
    """name -> (call(q, k, v, g, **options), uses_gate, q head multiplier)."""
    return {
        "full": (lambda q, k, v, g, **o: xattn.causal_flash_attn(q, k, v, **o), False, 1),
        "swa": (lambda q, k, v, g, **o: xattn.flash_swa(q, k, v, 96, **o), False, 1),
        "sca": (lambda q, k, v, g, **o: xattn.flash_sca(q, k, v, 32, **o), False, 1),
        "softdelta": (lambda q, k, v, g, **o: xattn.softdelta_attention(q, k, v, g, **o), True, 2),
        "swa-softdelta": (lambda q, k, v, g, **o: xattn.sliding_window_softdelta_attention(q, k, v, g, 96, **o),
                          True, 2),
        "sca-softdelta": (lambda q, k, v, g, **o: xattn.sliding_chunk_softdelta_attention(q, k, v, g, 32, **o),
                          True, 2),
    }


MODES = {
    "level0": dict(),
    "deterministic": dict(deterministic=True),
    "level2": dict(high_precision_level=2),
    "level3": dict(high_precision_level=3),
    "level4": dict(high_precision_level=4),
    "deterministic-level4": dict(deterministic=True, high_precision_level=4),
}


def _relative(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("mode", list(MODES), ids=list(MODES))
@pytest.mark.parametrize("op", list(_ops()), ids=list(_ops()))
@pytest.mark.parametrize("shape", list(SHAPES), ids=list(SHAPES))
def test_large_batch_matches_chunked(shape, op, mode):
    _requires_sm90()
    batch, length, q_heads, kv_heads, chunks, gib = SHAPES[shape]
    if shape == "b64-l4096" and mode != "level0":
        pytest.skip("the long-context shape runs the default SM90 path only")
    if shape == "b16500-l64-gqa" and mode in ("level0", "deterministic"):
        pytest.skip("B * Hkv past the grid limit matters only for the precision-level repair kernels")
    if _free_bytes() < gib * 2**30:
        pytest.skip(f"needs {gib} GiB of free device memory")
    call, uses_gate, q_mult = _ops()[op]
    torch.manual_seed(31)
    dtype, dim = torch.bfloat16, 128
    q = F.normalize(torch.randn(batch, length, q_heads * q_mult, dim, device="cuda", dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(batch, length, kv_heads, dim, device="cuda", dtype=dtype), dim=-1)
    v = torch.randn(batch, length, kv_heads, dim, device="cuda", dtype=dtype)
    g = None
    if uses_gate:
        v = v.view(batch, length, kv_heads, 1, dim)
        g = torch.randn(batch, length, q_heads, 1, 1, device="cuda", dtype=dtype)
    leaves = [t for t in (q, k, v, g) if t is not None]
    out_shape = (batch, length, q_heads, 1, dim) if uses_gate else (batch, length, q_heads, dim)
    output_grad = torch.randn(out_shape, device="cuda", dtype=dtype)

    def run(rows):
        inputs = [t[rows].detach().clone().requires_grad_(True) for t in leaves]
        args = inputs + ([None] if not uses_gate else [])
        out = call(*args, **MODES[mode])
        grads = torch.autograd.grad(out, inputs, output_grad[rows])
        return [out.detach()] + list(grads)

    full = run(slice(None))
    step = batch // chunks
    parts = [run(slice(i, i + step)) for i in range(0, batch, step)]
    tail = slice(batch - batch // 64, None)  # includes every batch past the 65535 grid limit
    for index, name in enumerate(["out", "q", "k", "v", "g"][: len(full)]):
        expected = torch.cat([p[index] for p in parts])
        for label, rows in (("all", slice(None)), ("tail", tail)):
            error = _relative(full[index][rows], expected[rows])
            assert error < 2e-3, f"{shape} {op} {mode} {name} ({label}): relative error {error:.2e}"

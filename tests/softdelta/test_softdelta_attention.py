import importlib.util

import pytest
import torch
import torch.nn.functional as F

from xattn import softdelta_attention, softdelta_attention_bwd, softdelta_attention_fwd

from tests._reference import ATOL, GRAD_ATOL, RTOL
from .reference import softdelta_fp64_oracle


def _requires_sm90_extension():
    if importlib.util.find_spec("xattn_cuda") is None:
        pytest.skip("xattn CUDA extension is not built")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SoftDelta requires SM90")


def _inputs(*, batch, length, q_heads, kv_heads, qk_dim, value_dim, dtype,
            device="cuda", with_k_read=False, q_length=None, prev_length=0):
    q_length = length if q_length is None else q_length
    q = torch.randn(batch, q_length, q_heads, qk_dim, dtype=dtype, device=device)
    k = torch.randn(batch, length, kv_heads, qk_dim, dtype=dtype, device=device)
    v = torch.randn(batch, length, kv_heads, value_dim, dtype=dtype, device=device)
    g = torch.randn(batch, q_length, q_heads, dtype=dtype, device=device)
    k_read = torch.randn(batch, q_length, kv_heads, qk_dim, dtype=dtype, device=device) if with_k_read else None
    with torch.no_grad():
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.silu(v) + 0.1
        g = g * 0.75
        k_read = None if k_read is None else F.normalize(F.silu(k_read), dim=-1)
    prev_k = prev_v = None
    if prev_length:
        prev_k = F.normalize(torch.randn(batch, prev_length, kv_heads, qk_dim, dtype=dtype, device=device), dim=-1)
        prev_v = F.silu(torch.randn(batch, prev_length, kv_heads, value_dim, dtype=dtype, device=device)) + 0.1
    return dict(q=q, k=k, v=v, g=g, k_read=k_read, prev_k=prev_k, prev_v=prev_v)


def _segments(batch, length, lengths, device="cuda"):
    """Contiguous segment ids from a run-length list, repeated to cover ``length``."""
    rows = []
    for b in range(batch):
        ids, sid, i = [], 0, b
        while len(ids) < length:
            ids += [sid] * lengths[i % len(lengths)]
            sid += 1
            i += 1
        rows.append(ids[:length])
    return torch.tensor(rows, dtype=torch.int64, device=device)


def _assert_composed_close(actual, expected, read, neg_read, g, dtype):
    q_heads, kv_heads = read.shape[2], neg_read.shape[2]
    neg_read = neg_read.repeat_interleave(q_heads // kv_heads, dim=2)
    gate = torch.sigmoid(g.double())
    gate = gate if gate.dim() == read.dim() else gate[..., None]  # scalar gates [B, L, H]
    difference = (actual.double() - expected).abs()
    bound = (
        ATOL[dtype] + RTOL[dtype] * read.abs()
        + gate * (ATOL[dtype] + RTOL[dtype] * neg_read.abs())
        + ATOL[dtype] + RTOL[dtype] * expected.abs()
    )
    normalized = difference / bound
    assert bool((normalized <= 1).all().item()), (
        f"maximum composed normalized error={normalized.max().item()} "
        f"maximum absolute error={difference.max().item()}"
    )


def _oracle_metadata(metadata, q_length):
    """Segment ids of the call's metadata for the oracle."""
    oracle_metadata = dict(metadata)
    if "segment_idx" in oracle_metadata:
        segment_idx = oracle_metadata.pop("segment_idx")
        oracle_metadata["k_segment_idx"] = segment_idx
        oracle_metadata["q_segment_idx"] = segment_idx[:, -q_length:]
    if "bos_mask" in oracle_metadata:
        bos_mask = oracle_metadata.pop("bos_mask")
        segment_idx = bos_mask.cumsum(dim=1)
        oracle_metadata["k_segment_idx"] = segment_idx
        oracle_metadata["q_segment_idx"] = segment_idx[:, -q_length:]
    return oracle_metadata


def _oracle(tensors, metadata):
    """(expected, read, neg_read, FP64 leaves) over the call's already-quantized tensors."""
    leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    expected, read, neg_read = softdelta_fp64_oracle(
        **leaves, return_aux=True, **_oracle_metadata(metadata, tensors["q"].shape[1]))
    return expected, read, neg_read, leaves


def _assert_output(actual, expected, read, neg_read, g, dtype):
    assert actual.shape == expected.shape and actual.dtype == dtype
    assert bool(torch.isfinite(actual).all().item())
    _assert_composed_close(actual, expected, read, neg_read, g, dtype)


def _assert_gradients(names, actual_grads, expected_grads, dtype):
    for name, a, e in zip(names, actual_grads, expected_grads):
        assert a.dtype == dtype and a.shape == e.shape, name
        torch.testing.assert_close(a, e.to(dtype), rtol=RTOL[dtype], atol=GRAD_ATOL[dtype], msg=lambda m: f"{name}: {m}")
        # Elementwise tolerances alone can hide a missing gradient term; bound the relative norm error too.
        floor = GRAD_ATOL[dtype] * e.numel() ** 0.5  # exact-zero gradients fall back to the atol check
        relative = ((a.double() - e).norm() / e.norm().clamp_min(floor)).item()
        assert relative < GRAD_REL_NORM[dtype], f"{name}: relative norm error {relative}"


def _check(inputs, dtype, *, backend="sm90", deterministic=False, high_precision_level=0, **metadata):
    names = [n for n in ("q", "k", "v", "g", "k_read", "prev_k", "prev_v") if inputs[n] is not None]
    actual_inputs = {n: inputs[n].detach().clone().requires_grad_(True) for n in names}
    actual = softdelta_attention(
        backend=backend,
        deterministic=deterministic,
        high_precision_level=high_precision_level,
        **{'k_read': None, **actual_inputs, **metadata},
    )
    expected, read, neg_read, expected_inputs = _oracle({n: inputs[n] for n in names}, metadata)
    _assert_output(actual, expected, read, neg_read, inputs["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(actual_inputs.values()), y_grad)
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, actual_grads, expected_grads, dtype)
    return actual, actual_grads


GRAD_REL_NORM = {torch.float16: 5e-3, torch.bfloat16: 3e-2}
NEG_IDS = ("neg-key", "neg-given")
# (deterministic, high_precision_level): every backward algorithm.
ALL_MODES = tuple((det, level) for level in range(2) for det in (False, True))
MODE_IDS = [f"{'det' if det else 'nondet'}-level{level}" for det, level in ALL_MODES]

HEADS = ((4, 4), (4, 2), (4, 1), (12, 3))
CASES = {
    "length-one": dict(length=1),
    "length-two": dict(length=2),
    "uneven": dict(length=17),
    "block-boundary": dict(length=129),
    "segments": dict(length=150, segments=(1, 2, 40, 3, 64, 7)),
    "bos": dict(length=150, bos=(5, 1, 70, 2)),
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@pytest.mark.parametrize("heads", HEADS, ids=[f"h{h}kv{k}" for h, k in HEADS])
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_matches_fp64_oracle(case, heads, with_k_read, dtype, deterministic, high_precision_level):
    _requires_sm90_extension()
    torch.manual_seed(20261020)
    spec = CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=64,
                     value_dim=64, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    if "bos" in spec:
        metadata["bos_mask"] = _segments(batch, length, spec["bos"]).diff(dim=1, prepend=torch.full(
            (batch, 1), -1, dtype=torch.int64, device="cuda")) != 0
    _check(inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level, **metadata)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_torch_backend_matches_oracle(dtype):
    torch.manual_seed(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _inputs(batch=2, length=33, q_heads=4, kv_heads=2, qk_dim=32, value_dim=32,
                     dtype=dtype, device=device, with_k_read=True)
    _check(inputs, dtype, backend="torch",
           segment_idx=_segments(2, 33, (1, 9, 4, 30), device=device))


def test_gqa16x4_d128_packed_bf16():
    """GQA 16/4, D = V = 128, packed documents."""
    _requires_sm90_extension()
    torch.manual_seed(11)
    inputs = _inputs(batch=1, length=320, q_heads=16, kv_heads=4, qk_dim=128, value_dim=128,
                     dtype=torch.bfloat16)
    _check(inputs, torch.bfloat16, segment_idx=_segments(1, 320, (100, 1, 150, 69)))


@pytest.mark.parametrize("gate_value", (-30.0, 30.0))
def test_gate_extremes(gate_value):
    _requires_sm90_extension()
    torch.manual_seed(5)
    inputs = _inputs(batch=1, length=65, q_heads=4, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=torch.bfloat16)
    inputs["g"] = torch.full_like(inputs["g"], gate_value)
    _check(inputs, torch.bfloat16)


def test_deterministic_backward_is_repeatable():
    _requires_sm90_extension()
    torch.manual_seed(9)
    inputs = _inputs(batch=2, length=257, q_heads=8, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=torch.bfloat16)
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g")]
    grad = torch.randn(2, 257, 8, 64, dtype=torch.bfloat16, device="cuda")
    runs = []
    for _ in range(2):
        out = softdelta_attention(q=leaves[0], k=leaves[1], k_read=None, v=leaves[2], g=leaves[3], deterministic=True)
        runs.append(torch.autograd.grad(out, leaves, grad))
    for a, b in zip(*runs):
        assert torch.equal(a, b)


def test_rejects_invalid_arguments():
    inputs = _inputs(batch=1, length=8, q_heads=4, kv_heads=2, qk_dim=32, value_dim=32,
                     dtype=torch.bfloat16, device="cpu")
    q, k, v, g = inputs["q"], inputs["k"], inputs["v"], inputs["g"]
    with pytest.raises(ValueError, match="k_read must have shape"):
        softdelta_attention(q=q, k=k, v=v, g=g, k_read=q)
    with pytest.raises(ValueError, match="divisible"):
        softdelta_attention(q=q[:, :, :3], k=k, k_read=None, v=v, g=g[:, :, :3])
    with pytest.raises(ValueError, match="g must have shape"):
        softdelta_attention(q=q, k=k, k_read=None, v=v, g=g[:, :, :2])
    # Scalar gates only: the value-group layouts are not accepted.
    with pytest.raises(ValueError, match="4D q/k/v"):
        softdelta_attention(q=q, k=k, k_read=None, v=v.unsqueeze(3), g=g)
    with pytest.raises(ValueError, match="g must have shape"):
        softdelta_attention(q=q, k=k, k_read=None, v=v, g=g[..., None, None])
    with pytest.raises(ValueError, match="finite"):
        softdelta_attention(q=q, k=k, k_read=None, v=v, g=g, scale=float('nan'))
    with pytest.raises(ValueError, match="SM90"):
        softdelta_attention(q=q, k=k, k_read=None, v=v, g=g, backend='torch', high_precision_level=1)
    with pytest.raises(ValueError, match="0 or 1"):
        softdelta_attention(q=q, k=k, k_read=None, v=v, g=g, backend='torch', high_precision_level=2)
    with pytest.raises(ValueError, match="SM90"):
        softdelta_attention_fwd(q=q, k=k, k_read=None, v=v, g=g, backend='torch')


FUSED_CASES = {
    "length-one": dict(length=1),
    "uneven": dict(length=77),
    "block-boundary": dict(length=257),
    "segments": dict(length=300, segments=(1, 2, 130, 3, 64, 7)),
    # Every token its own segment: every negative-query tile is empty (zero-tile epilogue).
    "singletons": dict(length=300, segments=(1,)),
}


@pytest.mark.parametrize("case", list(FUSED_CASES), ids=list(FUSED_CASES))
@pytest.mark.parametrize("heads", ((4, 4), (8, 2), (16, 4), (8, 1), (16, 1)),
                         ids=("h4kv4", "h8kv2", "h16kv4", "h8kv1", "h16kv1"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ((False, 0), (True, 1)), ids=("level0", "det-level1"))
def test_fused_bucket_matches_fp64_oracle(case, heads, with_k_read, dtype, deterministic, high_precision_level):
    """D = V = 128 across head mappings, deterministic mode on the composition."""
    _requires_sm90_extension()
    torch.manual_seed(20261021)
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=128,
                     value_dim=128, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    _check(inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level, **metadata)


# (D, V) across the fused buckets: exact buckets, dims rounded up to a bucket, the
# dense forward full-span tiles ((32, 64), (192, 128), (256, 32)) and the dense backward square remaps
# ((64, 96) -> (96, 96); (160, 128), (152, 136) -> (160, 160), (192, 128), (192, 160) -> (192, 192);
# (128, 256), (256, 128) -> (256, 256)).
FUSED_DIMS = {
    "d32v32": (32, 32),
    "d32v64": (32, 64),
    "d40v56": (40, 56),
    "d64v96": (64, 96),
    "d80v72": (80, 72),
    "d128v64": (128, 64),
    "d152v136": (152, 136),
    "d160v128": (160, 128),
    "d192v128": (192, 128),
    "d192v160": (192, 160),
    "d192v192": (192, 192),
    "d128v256": (128, 256),
    "d256v32": (256, 32),
    "d256v128": (256, 128),
    "d256v256": (256, 256),
    "d64v256": (64, 256),
}


DEFAULT_DIMS = {"d32v32", "d40v56", "d64v96", "d256v128"}
DIM_CASES = [
    pytest.param(name, id=name, marks=() if name in DEFAULT_DIMS else pytest.mark.full_matrix)
    for name in FUSED_DIMS
]


@pytest.mark.parametrize("dims", DIM_CASES)
@pytest.mark.parametrize("case", ("uneven", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_dims_match_fp64_oracle(dims, case, with_k_read, dtype, high_precision_level):
    """Every D, V <= 256 takes the fused SM90 reads (bucketed like causal_flash_attn)."""
    _requires_sm90_extension()
    torch.manual_seed(20261023)
    qk_dim, value_dim = FUSED_DIMS[dims]
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=qk_dim,
                     value_dim=value_dim, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    _check(inputs, dtype, high_precision_level=high_precision_level, **metadata)


@pytest.mark.parametrize("case", list(FUSED_CASES), ids=list(FUSED_CASES))
@pytest.mark.parametrize("heads", ((4, 4), (8, 8), (8, 4), (8, 2), (16, 1)),
                         ids=("h4kv4", "h8kv8", "h8kv4", "h8kv2", "h16kv1"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_reader_groups_match_fp64_oracle(monkeypatch, case, heads, with_k_read, dtype, high_precision_level):
    """The reader-group backward (one work tile per key tile and KV head over all of its readers, dK / dV
    written once) forced for any group size; the extension picks it for MHA and groups of two."""
    _requires_sm90_extension()
    monkeypatch.setattr(importlib.import_module("xattn.ops.softdelta_attention"), "_BWD_READER_GROUPS", 1)
    torch.manual_seed(20261026)
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=128,
                     value_dim=128, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    _check(inputs, dtype, high_precision_level=high_precision_level, **metadata)


@pytest.mark.parametrize("dims", DIM_CASES)
@pytest.mark.parametrize("case", ("uneven", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_fused_reader_groups_dims_match_fp64_oracle(monkeypatch, dims, case, with_k_read, dtype):
    """Reader groups across the fused (D, V) buckets (MHA heads)."""
    _requires_sm90_extension()
    monkeypatch.setattr(importlib.import_module("xattn.ops.softdelta_attention"), "_BWD_READER_GROUPS", 1)
    torch.manual_seed(20261027)
    qk_dim, value_dim = FUSED_DIMS[dims]
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=4, kv_heads=4, qk_dim=qk_dim,
                     value_dim=value_dim, dtype=dtype, with_k_read=with_k_read)
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    _check(inputs, dtype, **metadata)


@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("segments", (False, True), ids=("dense", "segments"))
def test_fused_mha_automatic_layout_matches_fp64_oracle(with_k_read, segments):
    """MHA with enough work tiles for the automatic reader-group layout (B Hkv ceil(L / 128) >= 256 dense,
    >= 1024 with segments)."""
    _requires_sm90_extension()
    torch.manual_seed(20261028)
    batch, length, heads = (8, 2048, 8) if segments else (2, 2048, 8)
    inputs = _inputs(batch=batch, length=length, q_heads=heads, kv_heads=heads, qk_dim=128,
                     value_dim=128, dtype=torch.bfloat16, with_k_read=with_k_read)
    metadata = {"segment_idx": _segments(batch, length, (700, 1, 1300, 64))} if segments else {}
    _check(inputs, torch.bfloat16, **metadata)


def _bos_mask(batch, length, lengths):
    ids = _segments(batch, length, lengths)
    return ids.diff(dim=1, prepend=torch.full((batch, 1), -1, dtype=torch.int64, device="cuda")) != 0


# (D, V) for the packed (BOS) route.
BOS_DIMS = {"d128v128": (128, 128), "d64v96": (64, 96), "d40v56": (40, 56),
            "d256v256": (256, 256), "d192v192": (192, 192), "d128v64": (128, 64)}


@pytest.mark.parametrize("dims", list(BOS_DIMS), ids=list(BOS_DIMS))
@pytest.mark.parametrize("heads", ((8, 2), (16, 4), (8, 8), (8, 1)), ids=("h8kv2", "h16kv4", "h8kv8", "h8kv1"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ((False, 0), (True, 1)), ids=("level0", "det-level1"))
def test_fused_bos_varlen_forward_matches_fp64_oracle(monkeypatch, dims, heads, with_k_read, dtype, deterministic,
                                                      high_precision_level):
    """BOS masks: the packed varlen forward (forced here) writes both reads into the fused readers for the
    segment backward; level 1 keeps the fused segment forward."""
    _requires_sm90_extension()
    module = importlib.import_module("xattn.ops.softdelta_attention")
    monkeypatch.setattr(module, "select_flash_swa_bos_routes", lambda *args, **kwargs: (True, True))
    monkeypatch.setattr(module, "_VARLEN_MIN_TOKENS", 0)
    torch.manual_seed(20261024)
    qk_dim, value_dim = BOS_DIMS[dims]
    batch, length = 2, 300
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=qk_dim,
                     value_dim=value_dim, dtype=dtype, with_k_read=with_k_read)
    _check(inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level,
           bos_mask=_bos_mask(batch, length, (1, 2, 130, 3, 64, 7)))


BOS_CASES = {"runs": (1, 2, 130, 3, 64, 7), "one-run": (300,), "singletons": (1,), "long-runs": (129, 171)}


@pytest.mark.parametrize("case", list(BOS_CASES), ids=list(BOS_CASES))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_fused_bos_routes_match_fp64_oracle(monkeypatch, case, dtype):
    """BOS masks through the router (token threshold lifted): packed varlen or segment forward, dense reads
    for one run per row."""
    _requires_sm90_extension()
    monkeypatch.setattr(importlib.import_module("xattn.ops.softdelta_attention"), "_VARLEN_MIN_TOKENS", 0)
    torch.manual_seed(20261025)
    batch, length = 2, 300
    inputs = _inputs(batch=batch, length=length, q_heads=16, kv_heads=4, qk_dim=128,
                     value_dim=128, dtype=dtype, with_k_read=False)
    _check(inputs, dtype, bos_mask=_bos_mask(batch, length, BOS_CASES[case]))


def test_output_flattens_without_copy_semantics_issues():
    """The fused path may return a strided view; downstream reshapes and in-graph use must behave."""
    _requires_sm90_extension()
    torch.manual_seed(13)
    inputs = _inputs(batch=2, length=130, q_heads=8, kv_heads=2, qk_dim=128, value_dim=128,
                     dtype=torch.bfloat16)
    leaves = {n: inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g")}
    out = softdelta_attention(**{'k_read': None, **leaves})
    flat = out.reshape(2, 130, -1)
    weight = torch.randn(flat.shape[-1], 64, device="cuda", dtype=torch.bfloat16)
    loss = (flat @ weight).float().square().mean()
    grads = torch.autograd.grad(loss, list(leaves.values()))
    reference = softdelta_attention(**{'k_read': None, **leaves}).contiguous().reshape(2, 130, -1)
    torch.testing.assert_close(flat, reference)
    assert all(bool(torch.isfinite(g).all()) for g in grads)


@pytest.mark.parametrize("q_length", (1, 37, 129), ids=("q1", "q37", "q129"))
@pytest.mark.parametrize("segments", (False, True), ids=("dense", "segments"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_short_query_matches_fp64_oracle(q_length, segments, with_k_read, dtype, deterministic,
                                         high_precision_level):
    """Lq < Lk without previous K/V: the fused forward, then the composed backward (the fused backward
    needs equal lengths)."""
    _requires_sm90_extension()
    torch.manual_seed(20261103)
    batch, length = 2, 150
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=128, value_dim=128,
                     dtype=dtype, with_k_read=with_k_read, q_length=q_length)
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))} if segments else {}
    _check(inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level, **metadata)


# Standalone fwd / bwd cases: fused forward and backward (dense, segments, BOS) and short Q (composed
# backward).
STANDALONE_CASES = {
    "dense": dict(length=150),
    "segments": dict(length=150, segments=(1, 2, 40, 3, 64, 7)),
    "bos": dict(length=150, bos=(5, 1, 70, 2)),
    "short-q": dict(length=150, q_length=61),
}


@pytest.mark.parametrize("case", list(STANDALONE_CASES), ids=list(STANDALONE_CASES))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_standalone_fwd_bwd_match_fp64_oracle(case, with_k_read, dtype, deterministic, high_precision_level):
    """softdelta_attention_fwd returns the autograd output, the negative reads and both reads' LSE;
    softdelta_attention_bwd turns them into the oracle's gradients (bitwise the autograd ones when
    deterministic)."""
    _requires_sm90_extension()
    torch.manual_seed(20261104)
    spec = STANDALONE_CASES[case]
    batch, length = 2, spec["length"]
    q_heads, kv_heads = 8, 2
    value_dim = 64
    inputs = _inputs(batch=batch, length=length, q_heads=q_heads, kv_heads=kv_heads, qk_dim=64, value_dim=value_dim,
                     dtype=dtype, with_k_read=with_k_read, q_length=spec.get("q_length"))
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, length, spec["segments"])
    if "bos" in spec:
        metadata["bos_mask"] = _bos_mask(batch, length, spec["bos"])
    names = [n for n in ("q", "k", "v", "g", "k_read") if inputs[n] is not None]
    tensors = {n: inputs[n] for n in names}
    options = dict(metadata, deterministic=deterministic, high_precision_level=high_precision_level)
    if "k_read" in tensors:
        options["k_read"] = tensors["k_read"]
    base = (tensors["q"], tensors["k"], tensors["v"], tensors["g"])
    with torch.inference_mode():
        y, actual_neg_read, lse = softdelta_attention_fwd(q=base[0], k=base[1], v=base[2], g=base[3], **{'k_read': None, **options})
    q_length = tensors["q"].shape[1]
    assert y.shape == (batch, q_length, q_heads, value_dim) and y.dtype == dtype
    assert actual_neg_read.shape == (batch, q_length, kv_heads, value_dim)
    assert actual_neg_read.dtype == (torch.float32 if high_precision_level else dtype)
    assert lse.shape == (batch, kv_heads + q_heads, q_length) and lse.dtype == torch.float32
    expected, read, neg_read, expected_inputs = _oracle(tensors, metadata)
    _assert_output(y, expected, read, neg_read, tensors["g"], dtype)
    torch.testing.assert_close(actual_neg_read.double(), neg_read, atol=ATOL[dtype], rtol=RTOL[dtype])

    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}
    autograd_y = softdelta_attention(
        q=leaves['q'],
        k=leaves['k'],
        v=leaves['v'],
        g=leaves['g'],
        **{'k_read': None, **dict(options, **{n: leaves[n] for n in ('k_read',) if n in leaves})},
    )
    assert torch.equal(autograd_y, y)
    y_grad = torch.randn_like(y) / y.numel() ** 0.5
    dq, dk, dv, dg, dk_read = softdelta_attention_bwd(
        y_grad=y_grad,
        q=base[0],
        k=base[1],
        v=base[2],
        g=base[3],
        y=y,
        neg_read=actual_neg_read,
        lse=lse,
        **{'k_read': None, **options},
    )
    assert (dk_read is None) == (not with_k_read)
    gradients = {"q": dq, "k": dk, "v": dv, "g": dg, "k_read": dk_read}
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, [gradients[n] for n in names], expected_grads, dtype)
    if deterministic:
        autograd_grads = torch.autograd.grad(autograd_y, list(leaves.values()), y_grad)
        for name, a, b in zip(names, autograd_grads, (gradients[n] for n in names)):
            assert torch.equal(a, b), name


@pytest.mark.parametrize("scale", (-4.0, 0.0), ids=("negative", "zero"))
@pytest.mark.parametrize("q_length", (None, 61), ids=("dense", "short-q"))
@pytest.mark.parametrize("with_k_read", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_nonpositive_scale_matches_fp64_oracle(scale, q_length, with_k_read, high_precision_level):
    """Nonpositive scales run as positive ones over negated or zeroed queries, like the other attention
    APIs; the default negative query keeps the keys' sign (autograd and standalone fwd / bwd)."""
    _requires_sm90_extension()
    torch.manual_seed(20261106)
    dtype = torch.bfloat16
    inputs = _inputs(batch=2, length=150, q_heads=8, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=dtype, with_k_read=with_k_read, q_length=q_length)
    _check(inputs, dtype, high_precision_level=high_precision_level, scale=scale)
    names = [n for n in ("q", "k", "v", "g", "k_read") if inputs[n] is not None]
    tensors = {n: inputs[n] for n in names}
    options = dict(scale=scale, high_precision_level=high_precision_level)
    if with_k_read:
        options["k_read"] = tensors["k_read"]
    base = (tensors["q"], tensors["k"], tensors["v"], tensors["g"])
    y, actual_neg_read, lse = softdelta_attention_fwd(q=base[0], k=base[1], v=base[2], g=base[3], **{'k_read': None, **options})
    expected, read, neg_read, expected_inputs = _oracle(tensors, {"scale": scale})
    _assert_output(y, expected, read, neg_read, tensors["g"], dtype)
    y_grad = torch.randn_like(y) / y.numel() ** 0.5
    gradients = dict(zip(("q", "k", "v", "g", "k_read"), softdelta_attention_bwd(
        y_grad=y_grad,
        q=base[0],
        k=base[1],
        v=base[2],
        g=base[3],
        y=y,
        neg_read=actual_neg_read,
        lse=lse,
        **{'k_read': None, **options},
    )))
    assert (gradients["k_read"] is None) == (not with_k_read)
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, [gradients[n] for n in names], expected_grads, dtype)


def test_standalone_rejects_mismatched_state():
    _requires_sm90_extension()
    torch.manual_seed(20261105)
    inputs = _inputs(batch=1, length=64, q_heads=4, kv_heads=2, qk_dim=64, value_dim=64,
                     dtype=torch.bfloat16)
    base = (inputs["q"], inputs["k"], inputs["v"], inputs["g"])
    y, neg_read, lse = softdelta_attention_fwd(q=base[0], k=base[1], k_read=None, v=base[2], g=base[3], high_precision_level=1)
    y_grad = torch.randn_like(y)
    with pytest.raises(ValueError, match="neg_read has incompatible"):
        softdelta_attention_bwd(
            y_grad=y_grad,
            q=base[0],
            k=base[1],
            k_read=None,
            v=base[2],
            g=base[3],
            y=y,
            neg_read=neg_read.to(torch.bfloat16),
            lse=lse,
            high_precision_level=1,
        )
    with pytest.raises(ValueError, match="neg_read has incompatible"):
        softdelta_attention_bwd(
            y_grad=y_grad,
            q=base[0],
            k=base[1],
            k_read=None,
            v=base[2],
            g=base[3],
            y=y,
            neg_read=neg_read,
            lse=lse,
            high_precision_level=0,
        )
    with pytest.raises(ValueError, match="lse has incompatible"):
        softdelta_attention_bwd(
            y_grad=y_grad,
            q=base[0],
            k=base[1],
            k_read=None,
            v=base[2],
            g=base[3],
            y=y,
            neg_read=neg_read,
            lse=lse[:, 1:],
            high_precision_level=1,
        )
    with pytest.raises(ValueError, match="neg_read has incompatible"):
        softdelta_attention_bwd(
            y_grad=y_grad,
            q=base[0],
            k=base[1],
            k_read=None,
            v=base[2],
            g=base[3],
            y=y,
            neg_read=neg_read[:, :, 1:],
            lse=lse,
            high_precision_level=1,
        )
    with pytest.raises(ValueError, match="y has incompatible"):
        softdelta_attention_bwd(
            y_grad=y_grad,
            q=base[0],
            k=base[1],
            k_read=None,
            v=base[2],
            g=base[3],
            y=y.float(),
            neg_read=neg_read,
            lse=lse,
            high_precision_level=1,
        )
    with pytest.raises(ValueError, match="y_grad has incompatible"):
        softdelta_attention_bwd(
            y_grad=y_grad.float(),
            q=base[0],
            k=base[1],
            k_read=None,
            v=base[2],
            g=base[3],
            y=y,
            neg_read=neg_read,
            lse=lse,
            high_precision_level=1,
        )


def _rope(x, base=500000.0):
    """Adjacent-pair RoPE (2m, 2m + 1) over positions 0..L-1 in FP32 (FP64 for FP64 inputs)."""
    compute = torch.float64 if x.dtype == torch.float64 else torch.float32
    length, dim = x.shape[1], x.shape[-1]
    inverse = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float64) / dim)
    angle = torch.arange(length, device=x.device, dtype=torch.float64)[:, None] * inverse
    cos, sin = (f(angle).to(compute)[None, :, None, :] for f in (torch.cos, torch.sin))
    even, odd = x[..., 0::2].to(compute), x[..., 1::2].to(compute)
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)


def _negative_query_transforms(source, leaves):
    """(q, k, k_read) from the leaves: post-RoPE queries and keys, and the negative-branch query the
    caller builds (None: the keys; the keys passed explicitly; rope(silu(k)); a second query)."""
    q, k = _rope(leaves["q_pre"]), _rope(leaves["k_pre"])
    k_read = {
        "key": None,
        "explicit-key": k,
        "silu-key": _rope(F.silu(leaves["k_pre"])),
        "second-query": _rope(leaves["q2_pre"]) if "q2_pre" in leaves else None,
    }[source]
    return q, k, k_read


@pytest.mark.parametrize("source", ("key", "explicit-key", "silu-key", "second-query"))
@pytest.mark.parametrize("heads", ((8, 2), (4, 4)), ids=("h8kv2", "h4kv4"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ((False, 0), (True, 0), (False, 1), (True, 1)),
                         ids=("level0", "det-level0", "level1", "det-level1"))
def test_k_read_sources_match_fp64_oracle(source, heads, dtype, deterministic, high_precision_level):
    """The negative branch reads with whatever post-RoPE query the caller builds; gradients flow back
    through the caller's transforms. Reference: the FP64 oracle on the call's quantized tensors, chained
    through the same transforms in FP64."""
    _requires_sm90_extension()
    torch.manual_seed(20261106)
    batch, length, (q_heads, kv_heads), dim = 2, 150, heads, 64
    leaves = {
        "q_pre": F.normalize(torch.randn(batch, length, q_heads, dim, device="cuda"), dim=-1),
        "k_pre": F.normalize(torch.randn(batch, length, kv_heads, dim, device="cuda"), dim=-1),
        "v": F.silu(torch.randn(batch, length, kv_heads, 64, device="cuda")) + 0.1,
        "g": torch.randn(batch, length, q_heads, device="cuda") * 0.75,
    }
    if source == "second-query":
        leaves["q2_pre"] = F.normalize(torch.randn(batch, length, kv_heads, dim, device="cuda"), dim=-1)
    leaves = {n: t.to(dtype).requires_grad_(True) for n, t in leaves.items()}
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))}
    q, k, k_read = _negative_query_transforms(source, leaves)
    actual = softdelta_attention(
        q=q,
        k=k,
        v=leaves['v'],
        g=leaves['g'],
        k_read=k_read,
        deterministic=deterministic,
        high_precision_level=high_precision_level,
        **metadata,
    )
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(leaves.values()), y_grad)

    tensors = {"q": q, "k": k, "v": leaves["v"], "g": leaves["g"]}
    if k_read is not None:
        tensors["k_read"] = k_read
    expected, read, neg_read, post = _oracle(tensors, metadata)
    _assert_output(actual, expected, read, neg_read, leaves["g"], dtype)
    post_grads = dict(zip(post, torch.autograd.grad(expected, list(post.values()), y_grad.double())))
    double_leaves = {n: t.detach().double().requires_grad_(True) for n, t in leaves.items()}
    q64, k64, neg64 = _negative_query_transforms(source, double_leaves)
    outputs, grads = [q64, k64], [post_grads["q"], post_grads["k"]]
    if neg64 is not None:
        outputs.append(neg64)
        grads.append(post_grads["k_read"])
    transform_leaves = [n for n in double_leaves if n not in ("v", "g")]
    chained = dict(zip(transform_leaves, torch.autograd.grad(
        outputs, [double_leaves[n] for n in transform_leaves], grads)))
    chained.update(v=post_grads["v"], g=post_grads["g"])
    _assert_gradients(list(leaves), actual_grads, [chained[n] for n in leaves], dtype)


@pytest.mark.parametrize("layout", ("packed-projection", "head-slice", "offset", "last-stride"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_input_views_match_fp64_oracle(layout, dtype, high_precision_level):
    """q / k / k_read as views: slices of one packed projection are read in place by the fused launches;
    views TMA cannot address (odd base offset, strided last dimension) are copied first."""
    _requires_sm90_extension()
    torch.manual_seed(20261107)
    batch, length, q_heads, kv_heads, dim = 2, 150, 8, 2, 64

    def make(heads, offset=0):
        if layout == "packed-projection":
            return None
        shape = {"head-slice": (batch, length, 2 * heads, dim), "offset": (batch, length, heads, dim + 1),
                 "last-stride": (batch, length, heads, 2 * dim)}[layout]
        base = F.normalize(torch.randn(shape, device="cuda", dtype=dtype), dim=-1).requires_grad_(True)
        view = {"head-slice": lambda t: t[:, :, 1::2], "offset": lambda t: t[..., 1:],
                "last-stride": lambda t: t[..., ::2]}[layout](base)
        return base, view

    if layout == "packed-projection":
        packed = F.normalize(torch.randn(batch, length, q_heads + 2 * kv_heads, dim, device="cuda", dtype=dtype),
                             dim=-1).requires_grad_(True)
        q, k, k_read = packed[:, :, :q_heads], packed[:, :, q_heads:q_heads + kv_heads], packed[:, :, -kv_heads:]
    else:
        (_, q), (_, k), (_, k_read) = make(q_heads), make(kv_heads), make(kv_heads)
    v = (F.silu(torch.randn(batch, length, kv_heads, 64, device="cuda", dtype=dtype)) + 0.1).requires_grad_(True)
    g = (torch.randn(batch, length, q_heads, device="cuda", dtype=dtype) * 0.75).requires_grad_(True)
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))}
    actual = softdelta_attention(q=q, k=k, v=v, g=g, k_read=k_read, high_precision_level=high_precision_level, **metadata)
    tensors = {"q": q, "k": k, "v": v, "g": g, "k_read": k_read}
    expected, read, neg_read, expected_inputs = _oracle(tensors, metadata)
    _assert_output(actual, expected, read, neg_read, g, dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(tensors.values()), y_grad)
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(list(tensors), actual_grads, expected_grads, dtype)


@pytest.mark.parametrize("visibility", ("full", "window", "chunk"))
@pytest.mark.parametrize("prev", (False, True), ids=("dense", "prev"))
@pytest.mark.parametrize("deterministic", (False, True), ids=("nondet", "det"))
def test_reader_backward_fp32_kv_grads(visibility, prev, deterministic):
    """The readers' FP32 dK / dV (the composed level-0 merge input) are the accumulators the
    16-bit gradients are cast from."""
    _requires_sm90_extension()
    from xattn.ops._causal_attention import _dense_bwd, _dense_fwd
    from xattn.ops.sliding_chunk_attention import _flash_sca_bwd, _flash_sca_fwd

    dtype, span = torch.bfloat16, 64
    torch.manual_seed(0)

    def setup(q_heads, kv_heads):
        t = _inputs(batch=2, length=200, q_heads=q_heads, kv_heads=kv_heads, qk_dim=64, value_dim=64,
                    dtype=dtype, prev_length=span if prev else 0)
        q, k, v, prev_k, prev_v = t["q"], t["k"], t["v"], t["prev_k"], t["prev_v"]
        scale = q.shape[-1] ** -0.5
        if visibility == "chunk":
            y, lse = _flash_sca_fwd(q, k, v, span, scale, prev_k, prev_v, backend="sm90")
            return lambda **options: _flash_sca_bwd(
                torch.ones_like(y), q, k, v, y, lse, span, scale, prev_k, prev_v,
                deterministic=deterministic, backend="sm90", **options)
        window = None if visibility == "full" else span
        args = (scale, prev_k, prev_v, None, None)
        y, lse = _dense_fwd(q, k, v, window, *args, "sm90")
        return lambda **options: _dense_bwd(
            torch.ones_like(y), q, k, v, y, lse, window, *args, deterministic, "sm90", **options)

    backward = setup(8, 2)
    plain = backward()
    fp32 = backward(fp32_kv_grads=True)
    torch.testing.assert_close(fp32[0], plain[0], rtol=RTOL[dtype], atol=GRAD_ATOL[dtype])
    for index in (1, 2, 3, 4):
        if plain[index] is None:
            assert not prev and fp32[index] is None
            continue
        assert fp32[index].dtype == torch.float32 and fp32[index].shape == plain[index].shape
        if deterministic:
            assert torch.equal(fp32[index].to(dtype), plain[index])
        else:
            torch.testing.assert_close(fp32[index].to(dtype), plain[index], rtol=RTOL[dtype],
                                       atol=GRAD_ATOL[dtype])
    with pytest.raises(RuntimeError, match="grouped query heads"):
        setup(4, 4)(fp32_kv_grads=True)

"""Sliding window and sliding chunk group SoftDelta against the FP64 oracle."""
import importlib

import pytest
import torch

from xattn import (
    group_softdelta_attention,
    sliding_chunk_group_softdelta_attention,
    sliding_chunk_group_softdelta_attention_bwd,
    sliding_chunk_group_softdelta_attention_fwd,
    sliding_window_group_softdelta_attention,
    sliding_window_group_softdelta_attention_bwd,
    sliding_window_group_softdelta_attention_fwd,
)

from tests.softdelta._inputs import ATOL, RTOL
from .reference import group_softdelta_fp64_oracle
from .test_group_softdelta_attention import (
    ALL_MODES,
    FUSED_DIMS,
    MODE_IDS,
    NEG_IDS,
    _assert_gradients,
    _assert_output,
    _bos_mask,
    _inputs,
    _oracle_metadata,
    _requires_sm90_extension,
    _segments,
)

# (autograd, standalone forward, standalone backward, span argument)
VARIANTS = {
    "window": (sliding_window_group_softdelta_attention, sliding_window_group_softdelta_attention_fwd,
               sliding_window_group_softdelta_attention_bwd, "window_size"),
    "chunk": (sliding_chunk_group_softdelta_attention, sliding_chunk_group_softdelta_attention_fwd,
              sliding_chunk_group_softdelta_attention_bwd, "chunk_size"),
}
# Window 0 (empty negative reads), short and long spans for each variant.
SPANS = {"window": (0, 3, 64), "chunk": (1, 16, 64)}
VARIANT_SPANS = [(variant, span) for variant, spans in SPANS.items() for span in spans]
VARIANT_SPAN_IDS = [f"{variant}{span}" for variant, span in VARIANT_SPANS]


def _span_options(variant, span, reset=False):
    options = {VARIANTS[variant][3]: span}
    if variant == "chunk":
        options["reset_chunk_pos_per_seq"] = reset
    return options


def _oracle(variant, span, reset, tensors, metadata):
    """(expected, read, neg_read, FP64 leaves) over the call's already-quantized tensors."""
    leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    expected, read, neg_read = group_softdelta_fp64_oracle(
        **leaves, visibility=variant, span=span, reset_position_per_segment=reset,
        return_aux=True, **_oracle_metadata(metadata, tensors["q"].shape[1]))
    return expected, read, neg_read, leaves


def _check(variant, span, inputs, dtype, *, reset=False, backend="sm90", deterministic=False,
           high_precision_level=0, **metadata):
    names = [n for n in ("q", "k", "v", "g", "neg_q", "prev_k", "prev_v") if inputs[n] is not None]
    actual_inputs = {n: inputs[n].detach().clone().requires_grad_(True) for n in names}
    actual = VARIANTS[variant][0](
        **actual_inputs, **_span_options(variant, span, reset), backend=backend, deterministic=deterministic,
        high_precision_level=high_precision_level, **metadata,
    )
    expected, read, neg_read, expected_inputs = _oracle(variant, span, reset, {n: inputs[n] for n in names},
                                                        metadata)
    _assert_output(actual, expected, read, neg_read, inputs["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(actual_inputs.values()), y_grad)
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, actual_grads, expected_grads, dtype)
    return actual, actual_grads


HEADS = ((4, 4), (4, 2), (4, 1), (12, 3))
CASES = {
    "length-one": dict(length=1),
    "uneven": dict(length=17),
    "block-boundary": dict(length=129),
    "segments": dict(length=150, segments=(1, 2, 40, 3, 64, 7)),
    "bos": dict(length=150, bos=(5, 1, 70, 2)),
}


def _metadata(spec, batch, total):
    metadata = {}
    if "segments" in spec:
        metadata["segment_idx"] = _segments(batch, total, spec["segments"])
    if "bos" in spec:
        metadata["bos_mask"] = _bos_mask(batch, total, spec["bos"])
    return metadata


@pytest.mark.parametrize("variant,span", VARIANT_SPANS, ids=VARIANT_SPAN_IDS)
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@pytest.mark.parametrize("heads", HEADS, ids=[f"h{h}kv{k}" for h, k in HEADS])
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ((False, 0), (True, 1)), ids=("level0", "det-level1"))
def test_matches_fp64_oracle(variant, span, case, heads, with_neg_q, dtype, deterministic, high_precision_level):
    _requires_sm90_extension()
    torch.manual_seed(20261023)
    spec = CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=64,
                     groups=2, group_dim=32, dtype=dtype, with_neg_q=with_neg_q)
    _check(variant, span, inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level,
           **_metadata(spec, batch, length))


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("q_length", (None, 6), ids=("aligned", "short-q"))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_previous_kv(variant, q_length, with_neg_q, dtype, deterministic, high_precision_level):
    """Previous K/V (exactly one chunk for sliding chunks) across every backward algorithm."""
    _requires_sm90_extension()
    torch.manual_seed(20261024)
    span, prev_length = (5, 7) if variant == "window" else (8, 8)
    inputs = _inputs(batch=2, length=20, q_heads=4, kv_heads=2, qk_dim=64, groups=1, group_dim=64,
                     dtype=dtype, with_neg_q=with_neg_q, q_length=q_length, prev_length=prev_length)
    _check(variant, span, inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("q_length", (1, 37, 129), ids=("q1", "q37", "q129"))
@pytest.mark.parametrize("segments", (False, True), ids=("dense", "segments"))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_short_query_matches_fp64_oracle(variant, q_length, segments, with_neg_q, dtype, deterministic,
                                         high_precision_level):
    _requires_sm90_extension()
    torch.manual_seed(20261025)
    batch, length = 2, 150
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=128, groups=1, group_dim=128,
                     dtype=dtype, with_neg_q=with_neg_q, q_length=q_length)
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))} if segments else {}
    _check(variant, 48, inputs, dtype, deterministic=deterministic, high_precision_level=high_precision_level,
           **metadata)


RESET_METADATA = ("segment_idx", "bos_mask", "q-k-ids")


@pytest.mark.parametrize("metadata_form", RESET_METADATA)
@pytest.mark.parametrize("q_length", (None, 61), ids=("aligned", "short-q"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_chunk_reset_matches_fp64_oracle(metadata_form, q_length, dtype, deterministic, high_precision_level):
    """reset_chunk_pos_per_seq restarts chunk positions in every segment (right-aligned Q / K runs for
    separately given ids)."""
    _requires_sm90_extension()
    torch.manual_seed(20261026)
    batch, length = 2, 150
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=64, groups=2, group_dim=32,
                     dtype=dtype, with_neg_q=True, q_length=q_length)
    ids = _segments(batch, length, (5, 70, 2, 33))
    if metadata_form == "segment_idx":
        metadata = {"segment_idx": ids}
    elif metadata_form == "bos_mask":
        metadata = {"bos_mask": _bos_mask(batch, length, (5, 70, 2, 33))}
    else:
        metadata = {"q_segment_idx": ids[:, length - inputs["q"].shape[1]:].clone(), "k_segment_idx": ids}
    _check("chunk", 24, inputs, dtype, reset=True, deterministic=deterministic,
           high_precision_level=high_precision_level, **metadata)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_torch_backend_matches_oracle(variant, dtype):
    torch.manual_seed(20261027)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _inputs(batch=2, length=33, q_heads=4, kv_heads=2, qk_dim=32, groups=2, group_dim=16,
                     dtype=dtype, device=device, with_neg_q=True)
    _check(variant, 7, inputs, dtype, backend="torch", segment_idx=_segments(2, 33, (1, 9, 4, 30), device=device))


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_covering_span_matches_causal(variant, dtype):
    """A window covering the sequence, or one chunk holding it, reads like the causal API."""
    _requires_sm90_extension()
    torch.manual_seed(20261028)
    batch, length = 2, 150
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=64, groups=2, group_dim=32,
                     dtype=dtype, with_neg_q=True)
    tensors = {n: inputs[n] for n in ("q", "k", "v", "g", "neg_q")}
    metadata = {"segment_idx": _segments(batch, length, (1, 2, 40, 3, 64, 7))}
    span = length - 1 if variant == "window" else length
    local = VARIANTS[variant][0](**tensors, **_span_options(variant, span), **metadata)
    causal = group_softdelta_attention(**tensors, **metadata)
    torch.testing.assert_close(local, causal, atol=ATOL[dtype], rtol=RTOL[dtype])


# Standalone fwd / bwd cases (dense, segments, BOS, short Q, previous K/V, gate groups whose width is not a
# multiple of 8).
STANDALONE_CASES = {
    "dense": dict(length=150),
    "segments": dict(length=150, segments=(1, 2, 40, 3, 64, 7)),
    "bos": dict(length=150, bos=(5, 1, 70, 2)),
    "short-q": dict(length=150, q_length=61),
    "previous": dict(length=40, prev_length=32),
    "composed-groups": dict(length=70, groups=4, group_dim=12),
}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case", list(STANDALONE_CASES), ids=list(STANDALONE_CASES))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", ALL_MODES, ids=MODE_IDS)
def test_standalone_fwd_bwd_match_fp64_oracle(variant, case, with_neg_q, dtype, deterministic,
                                              high_precision_level):
    """The standalone forward returns the autograd output and both reads in class-major readers; the
    standalone backward turns them into the oracle's gradients (bitwise the autograd ones when
    deterministic)."""
    _requires_sm90_extension()
    torch.manual_seed(20261029)
    autograd_api, forward_api, backward_api, _ = VARIANTS[variant]
    spec = STANDALONE_CASES[case]
    span = 32
    batch, length = 2, spec["length"]
    q_heads, kv_heads = 8, 2
    groups, group_dim = spec.get("groups", 2), spec.get("group_dim", 32)
    inputs = _inputs(batch=batch, length=length, q_heads=q_heads, kv_heads=kv_heads, qk_dim=64, groups=groups,
                     group_dim=group_dim, dtype=dtype, with_neg_q=with_neg_q, q_length=spec.get("q_length"),
                     prev_length=spec.get("prev_length", 0))
    metadata = _metadata(spec, batch, length + spec.get("prev_length", 0))
    names = [n for n in ("q", "k", "v", "g", "neg_q", "prev_k", "prev_v") if inputs[n] is not None]
    tensors = {n: inputs[n] for n in names}
    options = dict(metadata, deterministic=deterministic, high_precision_level=high_precision_level,
                   **_span_options(variant, span))
    options.update({n: tensors[n] for n in ("neg_q", "prev_k", "prev_v") if n in tensors})
    base = (tensors["q"], tensors["k"], tensors["v"], tensors["g"])
    with torch.inference_mode():
        y, readers, lse = forward_api(*base, **options)
    q_length = tensors["q"].shape[1]
    assert readers.shape == (batch, q_length, kv_heads + q_heads, groups, group_dim)
    assert readers.dtype == (torch.float32 if high_precision_level else dtype)
    assert lse.shape == (batch, kv_heads + q_heads, q_length) and lse.dtype == torch.float32
    expected, read, neg_read, expected_inputs = _oracle(variant, span, False, tensors, metadata)
    _assert_output(y, expected, read, neg_read, tensors["g"], dtype)
    for actual_read, reference in ((readers[:, :, :kv_heads], neg_read), (readers[:, :, kv_heads:], read)):
        torch.testing.assert_close(actual_read.double(), reference, atol=ATOL[dtype], rtol=RTOL[dtype])

    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}
    autograd_y = autograd_api(
        leaves["q"], leaves["k"], leaves["v"], leaves["g"],
        **dict(options, **{n: leaves[n] for n in ("neg_q", "prev_k", "prev_v") if n in leaves}))
    assert torch.equal(autograd_y, y)
    y_grad = torch.randn_like(y) / y.numel() ** 0.5
    dq, dk, dv, dg, dneg_q, dprev_k, dprev_v = backward_api(y_grad, *base, readers, lse, **options)
    assert (dneg_q is None) == (not with_neg_q)
    assert (dprev_k is None) == (dprev_v is None) == ("prev_k" not in tensors)
    gradients = {"q": dq, "k": dk, "v": dv, "g": dg, "neg_q": dneg_q, "prev_k": dprev_k, "prev_v": dprev_v}
    expected_grads = torch.autograd.grad(expected, list(expected_inputs.values()), y_grad.double())
    _assert_gradients(names, [gradients[n] for n in names], expected_grads, dtype)
    if deterministic:
        autograd_grads = torch.autograd.grad(autograd_y, list(leaves.values()), y_grad)
        for name, a, b in zip(names, autograd_grads, (gradients[n] for n in names)):
            assert torch.equal(a, b), name


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("scale", (-4.0, 0.0), ids=("negative", "zero"))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("high_precision_level", (0, 2), ids=("level0", "level2"))
def test_nonpositive_scale_matches_fp64_oracle(variant, scale, with_neg_q, high_precision_level):
    """Nonpositive scales run over negated or zeroed queries (autograd and standalone fwd / bwd)."""
    _requires_sm90_extension()
    torch.manual_seed(20261030)
    dtype, span = torch.bfloat16, 32
    inputs = _inputs(batch=2, length=150, q_heads=8, kv_heads=2, qk_dim=64, groups=2, group_dim=32,
                     dtype=dtype, with_neg_q=with_neg_q)
    names = [n for n in ("q", "k", "v", "g", "neg_q") if inputs[n] is not None]
    tensors = {n: inputs[n] for n in names}
    leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    expected, read, neg_read = group_softdelta_fp64_oracle(
        **leaves, scale=scale, visibility=variant, span=span, return_aux=True)
    autograd_api, forward_api, backward_api, _ = VARIANTS[variant]
    options = dict(scale=scale, high_precision_level=high_precision_level, **_span_options(variant, span))
    if with_neg_q:
        options["neg_q"] = tensors["neg_q"]
    actual_leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}
    actual = autograd_api(actual_leaves["q"], actual_leaves["k"], actual_leaves["v"], actual_leaves["g"],
                          **dict(options, **({"neg_q": actual_leaves["neg_q"]} if with_neg_q else {})))
    _assert_output(actual, expected, read, neg_read, tensors["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    expected_grads = torch.autograd.grad(expected, list(leaves.values()), y_grad.double())
    _assert_gradients(names, torch.autograd.grad(actual, list(actual_leaves.values()), y_grad), expected_grads,
                      dtype)
    base = (tensors["q"], tensors["k"], tensors["v"], tensors["g"])
    y, readers, lse = forward_api(*base, **options)
    _assert_output(y, expected, read, neg_read, tensors["g"], dtype)
    gradients = dict(zip(("q", "k", "v", "g", "neg_q"), backward_api(y_grad, *base, readers, lse, **options)))
    assert (gradients["neg_q"] is None) == (not with_neg_q)
    _assert_gradients(names, [gradients[n] for n in names], expected_grads, dtype)


FUSED_CASES = {
    "uneven": dict(length=77),
    "block-boundary": dict(length=257),
    "segments": dict(length=300, segments=(1, 2, 130, 3, 64, 7)),
    # Every token its own segment: every negative-query tile is empty (zero-tile epilogue).
    "singletons": dict(length=300, segments=(1,)),
}
# Spans below every fused case's length (local kernels, not the causal route), across tile edges.
FUSED_SPANS = [("window", 1), ("window", 63), ("window", 200), ("chunk", 64), ("chunk", 100), ("chunk", 128)]
FUSED_SPAN_IDS = [f"{variant}{span}" for variant, span in FUSED_SPANS]
DIMS_SPANS = [("window", 70), ("chunk", 48)]


def _module():
    return importlib.import_module("xattn.ops.group_softdelta_attention")


def test_fused_path_is_used():
    """An SM90 build runs equal-length local calls on the fused reads (not the composition)."""
    _requires_sm90_extension()
    module = _module()
    inputs = _inputs(batch=1, length=64, q_heads=4, kv_heads=2, qk_dim=64, groups=1, group_dim=64,
                     dtype=torch.bfloat16)
    for kind in (module.AttentionVisibility.SLIDING_WINDOW, module.AttentionVisibility.SLIDING_CHUNK):
        local = module._Local(kind, 8)
        assert module._fused_ops(inputs["q"], inputs["v"], None, local, inputs["k"].shape[1]) is not None, kind


@pytest.mark.parametrize("variant,span", FUSED_SPANS, ids=FUSED_SPAN_IDS)
@pytest.mark.parametrize("case", list(FUSED_CASES), ids=list(FUSED_CASES))
@pytest.mark.parametrize("heads", ((4, 4), (8, 2), (16, 4), (8, 1), (16, 1)),
                         ids=("h4kv4", "h8kv2", "h16kv4", "h8kv1", "h16kv1"))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_bucket_matches_fp64_oracle(variant, span, case, heads, with_neg_q, dtype, high_precision_level):
    """D = V = 128 on the fused local reads across head mappings and span / tile edges."""
    _requires_sm90_extension()
    torch.manual_seed(20261032)
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=128,
                     groups=1, group_dim=128, dtype=dtype, with_neg_q=with_neg_q)
    _check(variant, span, inputs, dtype, high_precision_level=high_precision_level,
           **_metadata(spec, batch, length))


@pytest.mark.parametrize("dims", list(FUSED_DIMS), ids=list(FUSED_DIMS))
@pytest.mark.parametrize("variant,span", DIMS_SPANS, ids=[f"{v}{s}" for v, s in DIMS_SPANS])
@pytest.mark.parametrize("case", ("block-boundary", "segments"))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_dims_match_fp64_oracle(dims, variant, span, case, with_neg_q, dtype, high_precision_level):
    """Every D, V <= 256 bucket on the fused local reads."""
    _requires_sm90_extension()
    torch.manual_seed(20261033)
    qk_dim, groups, group_dim = FUSED_DIMS[dims]
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=8, kv_heads=2, qk_dim=qk_dim,
                     groups=groups, group_dim=group_dim, dtype=dtype, with_neg_q=with_neg_q)
    _check(variant, span, inputs, dtype, high_precision_level=high_precision_level,
           **_metadata(spec, batch, length))


@pytest.mark.parametrize("variant,span", DIMS_SPANS, ids=[f"{v}{s}" for v, s in DIMS_SPANS])
@pytest.mark.parametrize("case", list(FUSED_CASES), ids=list(FUSED_CASES))
@pytest.mark.parametrize("heads", ((4, 4), (8, 8), (8, 4), (8, 2), (16, 1)),
                         ids=("h4kv4", "h8kv8", "h8kv4", "h8kv2", "h16kv1"))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("high_precision_level", (0, 1), ids=("level0", "level1"))
def test_fused_reader_groups_match_fp64_oracle(monkeypatch, variant, span, case, heads, with_neg_q, dtype,
                                               high_precision_level):
    """The reader-group backward forced for any group size on local support."""
    _requires_sm90_extension()
    monkeypatch.setattr(_module(), "_BWD_READER_GROUPS", 1)
    torch.manual_seed(20261034)
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=128,
                     groups=1, group_dim=128, dtype=dtype, with_neg_q=with_neg_q)
    _check(variant, span, inputs, dtype, high_precision_level=high_precision_level,
           **_metadata(spec, batch, length))


@pytest.mark.parametrize("dims", list(FUSED_DIMS), ids=list(FUSED_DIMS))
@pytest.mark.parametrize("variant,span", DIMS_SPANS, ids=[f"{v}{s}" for v, s in DIMS_SPANS])
@pytest.mark.parametrize("case", ("block-boundary", "segments"))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_fused_reader_groups_dims_match_fp64_oracle(monkeypatch, dims, variant, span, case, with_neg_q, dtype):
    """Reader groups across the fused (D, V) buckets (MHA heads) on local support."""
    _requires_sm90_extension()
    monkeypatch.setattr(_module(), "_BWD_READER_GROUPS", 1)
    torch.manual_seed(20261035)
    qk_dim, groups, group_dim = FUSED_DIMS[dims]
    spec = FUSED_CASES[case]
    batch, length = 2, spec["length"]
    inputs = _inputs(batch=batch, length=length, q_heads=4, kv_heads=4, qk_dim=qk_dim,
                     groups=groups, group_dim=group_dim, dtype=dtype, with_neg_q=with_neg_q)
    _check(variant, span, inputs, dtype, **_metadata(spec, batch, length))


def test_deterministic_backward_is_repeatable():
    _requires_sm90_extension()
    torch.manual_seed(20261031)
    inputs = _inputs(batch=2, length=257, q_heads=8, kv_heads=2, qk_dim=64, groups=1, group_dim=64,
                     dtype=torch.bfloat16)
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g")]
    grad = torch.randn(2, 257, 8, 1, 64, dtype=torch.bfloat16, device="cuda")
    for variant in VARIANTS:
        runs = []
        for _ in range(2):
            out = VARIANTS[variant][0](*leaves, **_span_options(variant, 100), deterministic=True)
            runs.append(torch.autograd.grad(out, leaves, grad))
        for a, b in zip(*runs):
            assert torch.equal(a, b), variant


def test_rejects_invalid_arguments():
    inputs = _inputs(batch=1, length=8, q_heads=4, kv_heads=2, qk_dim=32, groups=1, group_dim=32,
                     dtype=torch.bfloat16, device="cpu", prev_length=3)
    q, k, v, g = inputs["q"], inputs["k"], inputs["v"], inputs["g"]
    window, chunk = sliding_window_group_softdelta_attention, sliding_chunk_group_softdelta_attention
    with pytest.raises(ValueError, match="nonnegative"):
        window(q, k, v, g, -1)
    with pytest.raises(TypeError, match="integer"):
        window(q, k, v, g, True)
    with pytest.raises(TypeError, match="integer"):
        window(q, k, v, g, 2.0)
    with pytest.raises(ValueError, match="positive"):
        chunk(q, k, v, g, 0)
    with pytest.raises(TypeError, match="reset_chunk_pos_per_seq must be a bool"):
        chunk(q, k, v, g, 4, reset_chunk_pos_per_seq=1)
    with pytest.raises(ValueError, match="must equal chunk_size"):
        chunk(q, k, v, g, 4, prev_k=inputs["prev_k"], prev_v=inputs["prev_v"])
    with pytest.raises(ValueError, match="neg_q must have shape"):
        window(q, k, v, g, 4, neg_q=q)
    with pytest.raises(ValueError, match="SM90"):
        sliding_window_group_softdelta_attention_fwd(q, k, v, g, 4, backend="torch")
    with pytest.raises(ValueError, match="SM90"):
        chunk(q, k, v, g, 4, backend="torch", high_precision_level=2)

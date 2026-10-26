"""The internal value-group layout of Group SoftDelta against the FP64 oracle.

The public API takes scalar gates (one value group spanning V). The implementation keeps value groups
(``v`` ``[B, L, Hkv, C, Dg]``, ``g`` ``[B, Lq, H, C, 1]``, a gate per group of ``Dg`` channels) for a
future API; these tests drive the internal autograd function with several groups so that the grouped
kernels stay correct.
"""
import importlib

import pytest
import torch
import torch.nn.functional as F

from .reference import group_softdelta_fp64_oracle
from .test_group_softdelta_attention import (
    NEG_IDS,
    _assert_gradients,
    _assert_output,
    _requires_sm90_extension,
    _segments,
)

# (D, groups, group_dim): fused reads with segmented gate-gradient sums, and the composition (group
# widths that are not a multiple of 8).
GROUP_DIMS = {"d128c2": (128, 2, 64), "d192c3": (192, 3, 64), "d64c4-composed": (64, 4, 12)}
SUPPORTS = {"causal": (None, "full", None), "window40": ("SLIDING_WINDOW", "window", 40),
            "chunk32": ("SLIDING_CHUNK", "chunk", 32)}
MODES = ((False, 0), (True, 1), (False, 2))


def _module():
    return importlib.import_module("xattn.ops.group_softdelta_attention")


def _inputs(*, batch, length, q_heads, kv_heads, qk_dim, groups, group_dim, dtype, with_neg_q):
    def normal(*shape):
        return torch.randn(*shape, dtype=dtype, device="cuda")

    with torch.no_grad():
        tensors = dict(
            q=F.normalize(normal(batch, length, q_heads, qk_dim), dim=-1),
            k=F.normalize(normal(batch, length, kv_heads, qk_dim), dim=-1),
            v=F.silu(normal(batch, length, kv_heads, groups, group_dim)) + 0.1,
            g=normal(batch, length, q_heads, groups, 1) * 0.75,
        )
        if with_neg_q:
            tensors["neg_q"] = F.normalize(F.silu(normal(batch, length, kv_heads, qk_dim)), dim=-1)
    return tensors


@pytest.mark.parametrize("dims", list(GROUP_DIMS), ids=list(GROUP_DIMS))
@pytest.mark.parametrize("support", list(SUPPORTS), ids=list(SUPPORTS))
@pytest.mark.parametrize("heads", ((8, 2), (16, 1)), ids=("h8kv2", "h16kv1"))
@pytest.mark.parametrize("segments", (False, True), ids=("dense", "segments"))
@pytest.mark.parametrize("with_neg_q", (False, True), ids=NEG_IDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("deterministic,high_precision_level", MODES, ids=("level0", "det-level1", "level2"))
def test_value_groups_match_fp64_oracle(dims, support, heads, segments, with_neg_q, dtype, deterministic,
                                        high_precision_level):
    _requires_sm90_extension()
    module = _module()
    torch.manual_seed(20261025)
    qk_dim, groups, group_dim = GROUP_DIMS[dims]
    kind, visibility, span = SUPPORTS[support]
    local = None if kind is None else module._Local(getattr(module.AttentionVisibility, kind), span)
    batch, length = 2, 150
    tensors = _inputs(batch=batch, length=length, q_heads=heads[0], kv_heads=heads[1], qk_dim=qk_dim,
                      groups=groups, group_dim=group_dim, dtype=dtype, with_neg_q=with_neg_q)
    segment_idx = _segments(batch, length, (1, 2, 40, 3, 64, 7)) if segments else None

    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}
    actual = module._GroupSoftDeltaFunc.apply(
        leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves.get("neg_q"), None, None, segment_idx,
        segment_idx, qk_dim ** -0.5, deterministic, high_precision_level, None, local,
    )
    expected_leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    expected, read, neg_read = group_softdelta_fp64_oracle(
        **expected_leaves, q_segment_idx=segment_idx, k_segment_idx=segment_idx, visibility=visibility,
        span=span, return_aux=True,
    )
    _assert_output(actual, expected, read, neg_read, tensors["g"], dtype)
    y_grad = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, list(leaves.values()), y_grad)
    expected_grads = torch.autograd.grad(expected, list(expected_leaves.values()), y_grad.double())
    _assert_gradients(list(leaves), actual_grads, expected_grads, dtype)

# Author: Shicheng Wen

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ._backward_precision import (
    allocate_backward_output_state,
    use_high_precision_output,
)
from ._bos_routing_heuristics import (
    select_flash_sca_bos_routes as _select_bos_routes,
)
from ._segment_metadata import (
    resolve_segment_indices as _resolve_segment_indices,
)
from ._varlen_metadata import validate_packed_qkv
from ._sca_segment_alignment import (
    needs_matched_reset as _needs_matched_reset,
    matched_forward as _matched_reset_forward,
    matched_backward as _matched_reset_backward,
)

__all__ = [
    "flash_sca",
    "flash_sca_bwd",
    "flash_sca_fwd",
    "flash_sca_sm90_available",
]


@dataclass(frozen=True)
class _FlashSCAFwdMetadataPlan:
    _route: str
    _chunk_size: int
    _reset_chunk_pos_per_seq: bool
    _q_segment_idx: Optional[Tensor] = None
    _k_segment_idx: Optional[Tensor] = None
    _cu_seqlens_q: Optional[Tensor] = None
    _cu_seqlens_k: Optional[Tensor] = None
    _max_seqlen_q: Optional[int] = None
    _max_seqlen_k: Optional[int] = None
    _position_offsets: Optional[Tensor] = None
    _k_run_starts: Optional[Tensor] = None
    _k_run_lengths: Optional[Tensor] = None


def _extension_ops():
    try:
        import xattn_cuda
    except ImportError as exc:
        raise ImportError(
            "xattn CUDA extension is not installed. Install xattn with "
            "`pip install --no-build-isolation .` with a CUDA compiler available."
        ) from exc
    return xattn_cuda.ops


class _FlashSCAFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        scale: float,
        prev_k: Optional[Tensor] = None,
        prev_v: Optional[Tensor] = None,
        q_segment_idx: Optional[Tensor] = None,
        k_segment_idx: Optional[Tensor] = None,
        backend: str = "auto",
        reset_chunk_pos_per_seq: bool = False,
        deterministic: bool = False,
        high_precision_output: bool = False,
    ) -> Tensor:
        route = "segment" if q_segment_idx is not None else "dense"
        ctx.fwd_route = route
        ctx.bwd_route = route
        output_state = (
            allocate_backward_output_state(q, v)
            if high_precision_output
            else None
        )
        args = (
            q,
            k,
            v,
            chunk_size,
            scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            backend,
            reset_chunk_pos_per_seq,
        )
        if output_state is not None:
            args = (*args, output_state)
        y, lse = _extension_ops().flash_sca_fwd(*args)
        ctx.save_for_backward(
            q,
            k,
            v,
            output_state if output_state is not None else y,
            lse,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
        )
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.backend = backend
        ctx.reset_chunk_pos_per_seq = reset_chunk_pos_per_seq
        ctx.deterministic = deterministic
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: Tensor,
    ) -> tuple:
        (
            q,
            k,
            v,
            y,
            lse,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
        ) = ctx.saved_tensors
        q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad = (
            _extension_ops().flash_sca_bwd(
                y_grad,
                q,
                k,
                v,
                y,
                lse,
                ctx.chunk_size,
                ctx.scale,
                prev_k,
                prev_v,
                q_segment_idx,
                k_segment_idx,
                ctx.deterministic,
                ctx.backend,
                ctx.reset_chunk_pos_per_seq,
            )
        )
        return (
            q_grad,
            k_grad,
            v_grad,
            None,
            None,
            prev_k_grad,
            prev_v_grad,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FlashSCAVarlenFunc(torch.autograd.Function):
    """Autograd bridge for packed varlen Q/K/V."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        scale: float,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        position_offsets: Optional[Tensor],
        backend: str,
        reset_chunk_pos_per_seq: bool,
        deterministic: bool,
        k_run_starts: Optional[Tensor],
        k_run_lengths: Optional[Tensor],
        k_prefix_ends: Optional[Tensor],
        k_row_length: int,
        high_precision_output: bool,
    ) -> Tensor:
        output_state = (
            allocate_backward_output_state(q, v)
            if high_precision_output
            else None
        )
        y, lse = _flash_sca_varlen_fwd(
            q,
            k,
            v,
            chunk_size,
            scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            position_offsets=position_offsets,
            backend=backend,
            reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
            _k_run_starts=k_run_starts,
            _k_run_lengths=k_run_lengths,
            output_state=output_state,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            output_state if output_state is not None else y,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            position_offsets,
            k_run_starts,
            k_run_lengths,
            k_prefix_ends,
        )
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.backend = backend
        ctx.reset_chunk_pos_per_seq = reset_chunk_pos_per_seq
        ctx.deterministic = deterministic
        ctx.k_row_length = k_row_length
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y_grad: Tensor) -> tuple:
        (
            q,
            k,
            v,
            y,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            position_offsets,
            k_run_starts,
            k_run_lengths,
            k_prefix_ends,
        ) = ctx.saved_tensors
        q_grad, k_grad, v_grad = _flash_sca_varlen_bwd(
            y_grad,
            q,
            k,
            v,
            y,
            lse,
            ctx.chunk_size,
            ctx.scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            deterministic=ctx.deterministic,
            position_offsets=position_offsets,
            backend=ctx.backend,
            reset_chunk_pos_per_seq=ctx.reset_chunk_pos_per_seq,
            _k_run_starts=k_run_starts,
            _k_run_lengths=k_run_lengths,
            _k_prefix_ends=k_prefix_ends,
            _k_row_length=ctx.k_row_length,
        )
        return (
            q_grad,
            k_grad,
            v_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FlashSCABosFunc(torch.autograd.Function):
    """Autograd bridge for BOS plans whose FWD/BWD routes may differ."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        scale: float,
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        q_segment_idx: Optional[Tensor],
        k_segment_idx: Optional[Tensor],
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        backend: str,
        reset_chunk_pos_per_seq: bool,
        deterministic: bool,
        fwd_use_varlen: bool,
        bwd_use_varlen: bool,
        high_precision_output: bool,
    ) -> Tensor:
        output_state = (
            allocate_backward_output_state(q, v)
            if high_precision_output
            else None
        )
        if fwd_use_varlen:
            y, lse = _flash_sca_fwd(
                q,
                k,
                v,
                chunk_size,
                scale,
                prev_k,
                prev_v,
                None,
                None,
                backend,
                reset_chunk_pos_per_seq,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                output_state=output_state,
            )
        else:
            args = (
                q,
                k,
                v,
                chunk_size,
                scale,
                prev_k,
                prev_v,
                q_segment_idx,
                k_segment_idx,
                backend,
                reset_chunk_pos_per_seq,
            )
            if output_state is not None:
                args = (*args, output_state)
            y, lse = _extension_ops().flash_sca_fwd(*args)
        ctx.save_for_backward(
            q,
            k,
            v,
            output_state if output_state is not None else y,
            lse,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.backend = backend
        ctx.reset_chunk_pos_per_seq = reset_chunk_pos_per_seq
        ctx.deterministic = deterministic
        ctx.fwd_route = "varlen" if fwd_use_varlen else "segment"
        ctx.bwd_route = "varlen" if bwd_use_varlen else "segment"
        ctx.bwd_use_varlen = bwd_use_varlen
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y_grad: Tensor) -> tuple:
        (
            q,
            k,
            v,
            y,
            lse,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = ctx.saved_tensors
        use_varlen = ctx.bwd_use_varlen
        if use_varlen:
            q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad = (
                _flash_sca_bwd(
                    y_grad,
                    q,
                    k,
                    v,
                    y,
                    lse,
                    ctx.chunk_size,
                    ctx.scale,
                    prev_k,
                    prev_v,
                    None,
                    None,
                    ctx.deterministic,
                    ctx.backend,
                    ctx.reset_chunk_pos_per_seq,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                )
            )
        else:
            q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad = (
                _extension_ops().flash_sca_bwd(
                    y_grad,
                    q,
                    k,
                    v,
                    y,
                    lse,
                    ctx.chunk_size,
                    ctx.scale,
                    prev_k,
                    prev_v,
                    q_segment_idx,
                    k_segment_idx,
                    ctx.deterministic,
                    ctx.backend,
                    ctx.reset_chunk_pos_per_seq,
                )
            )
        return (
            q_grad,
            k_grad,
            v_grad,
            None,
            None,
            prev_k_grad,
            prev_v_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _resolve_scale(q: Tensor, scale: Optional[float]) -> float:
    return 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)


def _resolve_attn_method(attn_method: str) -> str:
    if not isinstance(attn_method, str):
        raise TypeError("attn_method must be a string")
    method = attn_method.strip().lower().replace("-", "_")
    if method in ("default", "sda"):
        return method
    raise ValueError("attn_method must be one of: default, sda")


def _check_attn_method_supported(attn_method: str) -> None:
    method = _resolve_attn_method(attn_method)
    if method == "sda":
        raise NotImplementedError(
            "attn_method='sda' is reserved for SoftDelta attention output; "
            "use sliding_chunk_softdelta_attention(q, k, v, g, ...)"
        )


def _validate_qkv_heads(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    packed: bool,
    api_name: str,
) -> None:
    if packed:
        validate_packed_qkv(q, k, v, api_name=api_name)
        return
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"{api_name} expects batched 4D q/k/v tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"{api_name} q/k/v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"{api_name} q/k/v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"{api_name} supports only torch.float16 and torch.bfloat16 input; "
            f"got {q.dtype}"
        )

    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError(f"{api_name} batch dimensions and k/v lengths must match")
    if not 0 < q.shape[1] <= k.shape[1]:
        raise ValueError(f"{api_name} requires 0 < q length <= k/v length")
    head_axis = 2

    q_heads = q.shape[head_axis]
    k_heads = k.shape[head_axis]
    v_heads = v.shape[head_axis]
    if k_heads != v_heads:
        raise ValueError(f"{api_name} k/v head counts must match")
    if q_heads <= 0 or k_heads <= 0:
        raise ValueError(f"{api_name} q and kv head counts must be positive")
    if q_heads % k_heads != 0:
        raise ValueError(
            f"{api_name} q head count must be divisible by kv head count; "
            f"got Hq={q_heads}, Hkv={k_heads}"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"{api_name} q/k head dims must match")


def _validate_prev_kv_heads(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    chunk_size: int,
    *,
    api_name: str,
) -> None:
    if prev_k is None:
        return
    assert prev_v is not None
    if prev_k.dim() != 4 or prev_v.dim() != 4:
        raise ValueError(f"{api_name} prev_k/prev_v must be batched 4D tensors")
    if prev_k.device != k.device or prev_v.device != v.device:
        raise ValueError(f"{api_name} previous and current K/V must share a device")
    if prev_k.dtype != k.dtype or prev_v.dtype != v.dtype:
        raise ValueError(f"{api_name} previous and current K/V must share a dtype")
    if prev_k.shape[0] != q.shape[0] or prev_v.shape[0] != q.shape[0]:
        raise ValueError(f"{api_name} prev_k/prev_v batch size must match q")
    if prev_k.shape[1] != chunk_size or prev_v.shape[1] != chunk_size:
        raise ValueError(
            f"{api_name} prev_k/prev_v sequence length must equal chunk_size"
        )
    if prev_k.shape[2:] != k.shape[2:]:
        raise ValueError(f"{api_name} prev_k head count/dim must match current k")
    if prev_v.shape[2:] != v.shape[2:]:
        raise ValueError(f"{api_name} prev_v head count/dim must match current v")


def _validate_metadata_representation(
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    position_offsets: Optional[Tensor],
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    reset_chunk_pos_per_seq: bool,
) -> bool:
    if (prev_k is None) != (prev_v is None):
        raise ValueError("prev_k and prev_v must be provided together")
    if (q_segment_idx is None) != (k_segment_idx is None):
        raise ValueError("q_segment_idx and k_segment_idx must be provided together")
    has_segment = q_segment_idx is not None
    has_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if has_segment and has_varlen:
        raise ValueError(
            "segment metadata and varlen metadata are mutually exclusive; "
            "pass either seg idx or cu_seqlens, not both"
        )
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
    if has_varlen:
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError(
                "max_seqlen_q and max_seqlen_k are required with cu_seqlens"
            )
        if reset_chunk_pos_per_seq and position_offsets is not None:
            raise ValueError(
                "position_offsets are only used by global-chunk varlen input"
            )
    else:
        if max_seqlen_q is not None or max_seqlen_k is not None:
            raise ValueError("max_seqlen_q/max_seqlen_k require cu_seqlens")
        if position_offsets is not None:
            raise ValueError("position_offsets require cu_seqlens")
    return has_varlen


def _pack_batched_varlen_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pack batched Q and optional previous K/V for the varlen kernel."""
    has_prev = prev_k is not None
    if has_prev:
        assert prev_v is not None
    q_packed = q.flatten(0, 1)
    if not has_prev:
        return q_packed, k.flatten(0, 1), v.flatten(0, 1)
    return (
        q_packed,
        torch.cat([prev_k, k], dim=1).flatten(0, 1),
        torch.cat([prev_v, v], dim=1).flatten(0, 1),
    )


def _prepare_varlen_row_alignment_metadata(
    batch: int,
    q_row_length: int,
    k_row_length: int,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    position_offsets: Optional[Tensor],
    reset_chunk_pos_per_seq: bool,
) -> Tuple[
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
]:
    effective_position_offsets = position_offsets
    k_run_starts = None
    k_run_lengths = None
    k_prefix_ends = None
    if batch > 1 and cu_seqlens_k.numel() != cu_seqlens_q.numel():
        (
            k_run_starts,
            k_run_lengths,
            k_prefix_ends,
            q_row_positions,
        ) = _extension_ops()._flash_sca_sm90_build_row_aligned_metadata(
            cu_seqlens_q,
            cu_seqlens_k,
            batch,
            q_row_length,
            k_row_length,
        )
        if position_offsets is None and not reset_chunk_pos_per_seq:
            effective_position_offsets = q_row_positions
    elif (
        batch > 1
        and position_offsets is None
        and not reset_chunk_pos_per_seq
    ):
        effective_position_offsets = torch.remainder(
            cu_seqlens_q[:-1], q_row_length
        ).contiguous()
    return (
        effective_position_offsets,
        k_run_starts,
        k_run_lengths,
        k_prefix_ends,
    )


def _prepare_batched_varlen_metadata(
    q: Tensor,
    k_combined: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    position_offsets: Optional[Tensor],
    reset_chunk_pos_per_seq: bool,
) -> Tuple[
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    int,
]:
    """Prepare row-alignment metadata for batched varlen input."""
    batch = q.shape[0]
    k_row_length = k_combined.shape[0] // batch
    return (
        *_prepare_varlen_row_alignment_metadata(
            batch,
            q.shape[1],
            k_row_length,
            cu_seqlens_q,
            cu_seqlens_k,
            position_offsets,
            reset_chunk_pos_per_seq,
        ),
        k_row_length,
    )


def _restore_varlen_output(y: Tensor, q: Tensor, v: Tensor) -> Tensor:
    return y.reshape(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])


def _restore_varlen_lse(lse: Tensor, q: Tensor) -> Tensor:
    return (
        lse.reshape(q.shape[2], q.shape[0], q.shape[1])
        .permute(1, 0, 2)
        .contiguous()
    )


def _validate_bos_mask(
    bos: Tensor,
    reference: Tensor,
    expected_length: int,
    name: str,
) -> None:
    expected_shape = (reference.shape[0], expected_length)
    if (
        bos.device != reference.device
        or bos.dtype != torch.bool
        or bos.dim() != 2
        or tuple(bos.shape) != expected_shape
    ):
        raise ValueError(
            f"{name} must be a bool tensor on the Q/K/V device with shape "
            f"{list(expected_shape)}"
        )


def _bos_to_cu_seqlens(bos: Tensor) -> Tuple[Tensor, int]:
    ops = _extension_ops()
    if not hasattr(ops, "_flash_sca_sm90_bos_to_cu_seqlens"):
        raise RuntimeError(
            "BOS-mask FlashSCA requires an extension built with SM90 support"
        )
    cu_seqlens, max_seqlen = ops._flash_sca_sm90_bos_to_cu_seqlens(bos)
    return cu_seqlens, int(max_seqlen)


def _build_bos_plan(bos: Tensor) -> Tuple[Tensor, int, int]:
    """Build compact BOS metadata and return its run statistics."""
    ops = _extension_ops()
    if hasattr(ops, "_flash_sca_sm90_bos_plan"):
        cu_seqlens, max_seqlen, num_runs = ops._flash_sca_sm90_bos_plan(bos)
        return cu_seqlens, int(max_seqlen), int(num_runs)
    cu_seqlens, max_seqlen = _bos_to_cu_seqlens(bos)
    return cu_seqlens, max_seqlen, cu_seqlens.numel() - 1


def _bos_to_segment_idx(bos: Tensor) -> Tensor:
    ops = _extension_ops()
    if not hasattr(ops, "_flash_sca_sm90_bos_to_segment_idx"):
        raise RuntimeError(
            "BOS-mask FlashSCA requires an extension built with SM90 support"
        )
    return ops._flash_sca_sm90_bos_to_segment_idx(bos)


def _sm90_metadata_dense_fast_path_available(
    reference: Tensor,
    op_name: Optional[str] = None,
) -> bool:
    if not reference.is_cuda:
        return False
    major, minor = torch.cuda.get_device_capability(reference.device)
    if (major, minor) != (9, 0):
        return False
    return op_name is None or hasattr(_extension_ops(), op_name)


def _build_bos_metadata(
    q: Tensor,
    bos_mask: Tensor,
    k_length: int,
) -> Tuple[Tensor, Tensor, int, int, int, int, bool]:
    q_aligned_bos = (
        bos_mask
        if q.shape[1] == k_length
        else bos_mask[:, k_length - q.shape[1] :]
    )
    cu_seqlens_q, max_seqlen_q, num_q_runs = _build_bos_plan(
        q_aligned_bos
    )
    if bos_mask is q_aligned_bos:
        cu_seqlens_k, max_seqlen_k, num_k_runs = (
            cu_seqlens_q,
            max_seqlen_q,
            num_q_runs,
        )
    else:
        cu_seqlens_k, max_seqlen_k, num_k_runs = _build_bos_plan(
            bos_mask
        )
    metadata_is_dense = (
        num_q_runs == q.shape[0]
        and num_k_runs == q.shape[0]
        and _sm90_metadata_dense_fast_path_available(q)
    )
    return (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_q_runs,
        num_k_runs,
        metadata_is_dense,
    )


def _build_fwd_bos_metadata(
    q: Tensor,
    bos_mask: Tensor,
    k_length: int,
) -> Tuple[Tensor, Tensor, int, int, int, int, bool]:
    if q.shape[1] == k_length:
        return _build_bos_metadata(q, bos_mask, k_length)
    paired_plan = getattr(
        _extension_ops(),
        "_flash_sca_sm90_paired_bos_plan",
        None,
    )
    if paired_plan is None:
        return _build_bos_metadata(q, bos_mask, k_length)
    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_q_runs,
        num_k_runs,
    ) = paired_plan(bos_mask, q.shape[1])
    metadata_is_dense = (
        num_q_runs == q.shape[0]
        and num_k_runs == q.shape[0]
        and _sm90_metadata_dense_fast_path_available(q)
    )
    return (
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        int(num_q_runs),
        int(num_k_runs),
        metadata_is_dense,
    )


def _prepare_bos_metadata(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    bos_mask: Tensor,
    backend: str,
    *,
    api_name: str,
) -> Tuple[Tensor, Tensor, int, int, int, int, bool]:
    """Validate a public BOS mask and build both query/key run plans."""
    if backend.strip().lower() not in ("", "auto", "sm90"):
        raise ValueError("BOS-mask FlashSCA requires backend='auto'/'sm90'")
    if (prev_k is None) != (prev_v is None):
        raise ValueError("prev_k and prev_v must be provided together")

    k_length = k.shape[1] + (0 if prev_k is None else prev_k.shape[1])
    _validate_bos_mask(bos_mask, q, k_length, "bos_mask")
    if q.shape[1] > k_length:
        raise ValueError("bos_mask cannot right-align Q beyond its K layout")
    _validate_qkv_heads(q, k, v, packed=False, api_name=api_name)
    _validate_prev_kv_heads(
        q,
        k,
        v,
        prev_k,
        prev_v,
        chunk_size,
        api_name=api_name,
    )
    return _build_bos_metadata(q, bos_mask, k_length)


def _resolve_public_bos_route(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    bos_mask: Tensor,
    backend: str,
    reset_chunk_pos_per_seq: bool,
    *,
    deterministic: bool,
    for_backward: bool,
    api_name: str,
    high_precision_output: bool = False,
) -> Tuple[
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[int],
    Optional[int],
]:
    """Resolve a public BOS mask to dense, segment, or varlen metadata."""
    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_q_runs,
        num_k_runs,
        metadata_is_dense,
    ) = _prepare_bos_metadata(
        q,
        k,
        v,
        chunk_size,
        prev_k,
        prev_v,
        bos_mask,
        backend,
        api_name=api_name,
    )
    if metadata_is_dense:
        return None, None, None, None, None, None

    fwd_use_varlen, bwd_use_varlen = _select_bos_routes(
        q,
        k,
        v,
        chunk_size,
        num_q_runs,
        max_seqlen_q,
        deterministic,
        for_backward,
        reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        has_previous=prev_k is not None,
        high_precision_output=high_precision_output,
        previous_length=0 if prev_k is None else prev_k.shape[1],
        num_k_runs=num_k_runs,
        max_seqlen_k=max_seqlen_k,
        standalone=True,
    )
    use_varlen = bwd_use_varlen if for_backward else fwd_use_varlen
    if use_varlen:
        return (
            None,
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )

    k_segment_idx = _bos_to_segment_idx(bos_mask)
    q_segment_idx = k_segment_idx[:, -q.shape[1] :]
    return q_segment_idx, k_segment_idx, None, None, None, None


def _explicit_metadata_is_dense(
    q: Tensor,
    k: Tensor,
    prev_k: Optional[Tensor],
    q_segment_idx: Optional[Tensor],
    k_segment_idx: Optional[Tensor],
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    position_offsets: Optional[Tensor],
) -> bool:
    if q_segment_idx is not None:
        expected_k_length = k.shape[1] + (
            0 if prev_k is None else prev_k.shape[1]
        )
        if (
            q_segment_idx.dim() != 2
            or k_segment_idx is None
            or k_segment_idx.dim() != 2
            or tuple(q_segment_idx.shape) != (q.shape[0], q.shape[1])
            or tuple(k_segment_idx.shape)
            != (q.shape[0], expected_k_length)
        ):
            return False
        op_name = "_flash_sca_sm90_segment_metadata_is_dense"
        if not _sm90_metadata_dense_fast_path_available(q, op_name):
            return False
        return bool(
            getattr(_extension_ops(), op_name)(
                q_segment_idx, k_segment_idx
            )
        )

    if cu_seqlens_q is None:
        return False
    assert cu_seqlens_k is not None
    q_row_length = q.shape[1]
    k_row_length = k.shape[1] + (
        0 if prev_k is None else prev_k.shape[1]
    )
    if (
        max_seqlen_q != q_row_length
        or max_seqlen_k != k_row_length
    ):
        return False
    op_name = "_flash_sca_sm90_varlen_metadata_is_dense"
    if not _sm90_metadata_dense_fast_path_available(q, op_name):
        return False
    return bool(
        getattr(_extension_ops(), op_name)(
            cu_seqlens_q,
            cu_seqlens_k,
            q.shape[0],
            q_row_length,
            k_row_length,
            position_offsets,
        )
    )


def _uniform_packed_dense_shape(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    position_offsets: Optional[Tensor],
    chunk_size: int,
    reset_chunk_pos_per_seq: bool,
    has_row_alignment_metadata: bool,
) -> Optional[Tuple[int, int]]:
    """Return ``(num_sequences, length)`` for zero-copy dense packed input."""
    if (
        has_row_alignment_metadata
        or not q.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
        or q.shape[0] != k.shape[0]
        or k.shape[0] != v.shape[0]
        or cu_seqlens_q.dim() != 1
        or cu_seqlens_k.dim() != 1
        or cu_seqlens_q.numel() < 2
        or cu_seqlens_q.numel() != cu_seqlens_k.numel()
    ):
        return None
    num_sequences = cu_seqlens_q.numel() - 1
    if q.shape[0] % num_sequences != 0:
        return None
    sequence_length = q.shape[0] // num_sequences
    if (
        sequence_length <= 0
        or max_seqlen_q != sequence_length
        or max_seqlen_k != sequence_length
        or (
            reset_chunk_pos_per_seq
            and position_offsets is not None
        )
        or (
            not reset_chunk_pos_per_seq
            and position_offsets is None
            and sequence_length % chunk_size != 0
        )
    ):
        return None
    op_name = "_flash_sca_sm90_varlen_metadata_is_dense"
    if not _sm90_metadata_dense_fast_path_available(q, op_name):
        return None
    if not bool(
        getattr(_extension_ops(), op_name)(
            cu_seqlens_q,
            cu_seqlens_k,
            num_sequences,
            sequence_length,
            sequence_length,
            position_offsets,
        )
    ):
        return None
    return num_sequences, sequence_length


def _pack_varlen_backward_tensors(
    y_grad: Tensor, y: Tensor, lse: Tensor, q: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    return (
        y_grad.flatten(0, 1),
        y.flatten(0, 1),
        lse.permute(1, 0, 2).reshape(q.shape[2], -1).contiguous(),
    )


def _unpack_batched_varlen_grads(
    dq_packed: Tensor,
    dk_combined: Tensor,
    dv_combined: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    dq = dq_packed.reshape_as(q)
    if prev_k is None:
        return (
            dq,
            dk_combined.reshape_as(k),
            dv_combined.reshape_as(v),
            None,
            None,
        )

    assert prev_v is not None
    dk_batched = dk_combined.reshape(
        q.shape[0], prev_k.shape[1] + k.shape[1], *k.shape[2:]
    )
    dv_batched = dv_combined.reshape(
        q.shape[0], prev_v.shape[1] + v.shape[1], *v.shape[2:]
    )
    return (
        dq,
        dk_batched[:, prev_k.shape[1] :].contiguous(),
        dv_batched[:, prev_v.shape[1] :].contiguous(),
        dk_batched[:, : prev_k.shape[1]].contiguous(),
        dv_batched[:, : prev_v.shape[1]].contiguous(),
    )


def _flash_sca_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    position_offsets: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    deterministic: bool = False,
    attn_method: str = "default",
    high_precision_output: bool = False,
) -> Tensor:
    """Run autograd on packed varlen Q/K/V."""
    _check_attn_method_supported(attn_method)
    _validate_qkv_heads(
        q, k, v, packed=True, api_name="_flash_sca_varlen"
    )
    _validate_metadata_representation(
        None,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
        None,
        None,
        reset_chunk_pos_per_seq,
    )
    use_fp32_state = use_high_precision_output(
        high_precision_output,
        (q, k, v),
        api_name="_flash_sca_varlen",
    )
    return _FlashSCAVarlenFunc.apply(
        q,
        k,
        v,
        chunk_size,
        _resolve_scale(q, scale),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
        backend,
        reset_chunk_pos_per_seq,
        deterministic,
        None,
        None,
        None,
        0,
        use_fp32_state,
    )


def _flash_sca(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    deterministic: bool = False,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    position_offsets: Optional[Tensor] = None,
    attn_method: str = "default",
    high_precision_output: bool = False,
) -> Tensor:
    """Run FlashSCA with batched Q/K/V and autograd."""
    _check_attn_method_supported(attn_method)
    use_fp32_state = use_high_precision_output(
        high_precision_output,
        (q, k, v, prev_k, prev_v),
        api_name="flash_sca",
    )
    if bos_mask is not None:
        if (
            q_segment_idx is not None
            or k_segment_idx is not None
            or segment_idx is not None
            or cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or max_seqlen_q is not None
            or max_seqlen_k is not None
            or position_offsets is not None
        ):
            raise ValueError(
                "BOS masks, segment indices, and cu-seqlens are mutually exclusive"
            )
        grad_inputs = (q, k, v, prev_k, prev_v)
        needs_backward = torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad
            for tensor in grad_inputs
        )
        if (
            torch.is_inference(bos_mask)
            and not needs_backward
            and q.is_cuda
            and bos_mask.is_cuda
        ):
            resolved_scale = _resolve_scale(q, scale)
            cached_result = _try_inference_bos_plan_fwd(
                q,
                k,
                v,
                chunk_size,
                resolved_scale,
                prev_k,
                prev_v,
                bos_mask,
                backend,
                reset_chunk_pos_per_seq,
            )
            if cached_result is not None:
                return cached_result[0]
            result = _prepare_store_and_execute_inference_bos_plan(
                q,
                k,
                v,
                chunk_size,
                resolved_scale,
                prev_k,
                prev_v,
                bos_mask,
                backend,
                reset_chunk_pos_per_seq,
                attn_method,
            )
            if result is not None:
                return result[0]
        if backend.strip().lower() not in ("", "auto", "sm90"):
            raise ValueError("BOS-mask FlashSCA requires backend='auto'/'sm90'")
        if (prev_k is None) != (prev_v is None):
            raise ValueError("prev_k and prev_v must be provided together")
        k_length = k.shape[1] + (0 if prev_k is None else prev_k.shape[1])
        _validate_bos_mask(bos_mask, q, k_length, "bos_mask")
        if q.shape[1] > k_length:
            raise ValueError("bos_mask cannot right-align Q beyond its K layout")
        q_aligned_bos = (
            bos_mask
            if q.shape[1] == k_length
            else bos_mask[:, k_length - q.shape[1] :]
        )
        _validate_qkv_heads(q, k, v, packed=False, api_name="flash_sca")
        _validate_prev_kv_heads(
            q,
            k,
            v,
            prev_k,
            prev_v,
            chunk_size,
            api_name="flash_sca",
        )
        cu_seqlens_q, max_seqlen_q, num_q_runs = _build_bos_plan(
            q_aligned_bos
        )
        if bos_mask is q_aligned_bos:
            cu_seqlens_k, max_seqlen_k, num_k_runs = (
                cu_seqlens_q,
                max_seqlen_q,
                num_q_runs,
            )
        else:
            (
                cu_seqlens_k,
                max_seqlen_k,
                num_k_runs,
            ) = _build_bos_plan(bos_mask)
        if (
            num_q_runs == q.shape[0]
            and num_k_runs == q.shape[0]
            and _sm90_metadata_dense_fast_path_available(q)
        ):
            return _FlashSCAFunc.apply(
                q,
                k,
                v,
                chunk_size,
                _resolve_scale(q, scale),
                prev_k,
                prev_v,
                None,
                None,
                backend,
                reset_chunk_pos_per_seq,
                deterministic,
                use_fp32_state,
            )
        fwd_use_varlen, bwd_use_varlen = _select_bos_routes(
            q,
            k,
            v,
            chunk_size,
            num_q_runs,
            max_seqlen_q,
            deterministic,
            needs_backward,
            reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
            has_previous=prev_k is not None,
            high_precision_output=use_fp32_state,
            previous_length=0 if prev_k is None else prev_k.shape[1],
            num_k_runs=num_k_runs,
            max_seqlen_k=max_seqlen_k,
        )
        if not fwd_use_varlen or not bwd_use_varlen:
            k_segment_idx = _bos_to_segment_idx(bos_mask)
            q_segment_idx = k_segment_idx[:, -q.shape[1] :]
        return _FlashSCABosFunc.apply(
            q,
            k,
            v,
            chunk_size,
            _resolve_scale(q, scale),
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            backend,
            reset_chunk_pos_per_seq,
            deterministic,
            fwd_use_varlen,
            bwd_use_varlen,
            use_fp32_state,
        )
    k_length = k.shape[1] + (0 if prev_k is None else prev_k.shape[1])
    q_segment_idx, k_segment_idx = _resolve_segment_indices(
        segment_idx,
        q_segment_idx,
        k_segment_idx,
        batch_size=q.shape[0],
        q_length=q.shape[1],
        k_length=k_length,
    )
    is_varlen = _validate_metadata_representation(
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
        prev_k,
        prev_v,
        reset_chunk_pos_per_seq,
    )
    _validate_qkv_heads(q, k, v, packed=False, api_name="flash_sca")
    _validate_prev_kv_heads(
        q,
        k,
        v,
        prev_k,
        prev_v,
        chunk_size,
        api_name="flash_sca",
    )
    if _explicit_metadata_is_dense(
        q,
        k,
        prev_k,
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
    ):
        q_segment_idx = None
        k_segment_idx = None
        cu_seqlens_q = None
        cu_seqlens_k = None
        max_seqlen_q = None
        max_seqlen_k = None
        position_offsets = None
        is_varlen = False
    if is_varlen:
        q_packed, k_combined, v_combined = _pack_batched_varlen_inputs(
            q, k, v, prev_k, prev_v
        )
        (
            effective_position_offsets,
            k_run_starts,
            k_run_lengths,
            k_prefix_ends,
            k_row_length,
        ) = _prepare_batched_varlen_metadata(
            q,
            k_combined,
            cu_seqlens_q,
            cu_seqlens_k,
            position_offsets,
            reset_chunk_pos_per_seq,
        )
        y_packed = _FlashSCAVarlenFunc.apply(
            q_packed,
            k_combined,
            v_combined,
            chunk_size,
            _resolve_scale(q, scale),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            effective_position_offsets,
            backend,
            reset_chunk_pos_per_seq,
            deterministic,
            k_run_starts,
            k_run_lengths,
            k_prefix_ends,
            k_row_length,
            use_fp32_state,
        )
        return _restore_varlen_output(y_packed, q, v)
    return _FlashSCAFunc.apply(
        q,
        k,
        v,
        chunk_size,
        _resolve_scale(q, scale),
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        backend,
        reset_chunk_pos_per_seq,
        deterministic,
        use_fp32_state,
    )


def _flash_sca_varlen_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    position_offsets: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    attn_method: str = "default",
    _k_run_starts: Optional[Tensor] = None,
    _k_run_lengths: Optional[Tensor] = None,
    strict_past: bool = False,
    output_state: Optional[Tensor] = None,
    output_fp32: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Run forward on packed varlen Q/K/V."""
    output_options = {"output_fp32": True} if output_fp32 else {}
    _check_attn_method_supported(attn_method)
    _validate_qkv_heads(
        q, k, v, packed=True, api_name="_flash_sca_varlen_fwd"
    )
    if backend.strip().lower() not in ("", "auto", "sm90"):
        raise ValueError(
            "_flash_sca_varlen_fwd requires backend='auto'/'sm90'"
        )
    dense_shape = _uniform_packed_dense_shape(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
        chunk_size,
        reset_chunk_pos_per_seq,
        _k_run_starts is not None or _k_run_lengths is not None,
    )
    if dense_shape is not None:
        batch, sequence_length = dense_shape
        q_dense = q.view(batch, sequence_length, *q.shape[1:])
        k_dense = k.view(batch, sequence_length, *k.shape[1:])
        v_dense = v.view(batch, sequence_length, *v.shape[1:])
        y_dense, lse_dense = _flash_sca_fwd(
            q_dense,
            k_dense,
            v_dense,
            chunk_size,
            scale,
            backend=backend,
            reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
            attn_method=attn_method,
            strict_past=strict_past,
            **output_options,
            output_state=(
                None
                if output_state is None
                else output_state.view(
                    batch, sequence_length, *output_state.shape[1:]
                )
            ),
        )
        return (
            y_dense.view(q.shape[0], *y_dense.shape[2:]),
            lse_dense.permute(1, 0, 2)
            .contiguous()
            .view(q.shape[1], q.shape[0]),
        )
    args = (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        chunk_size,
        scale,
        position_offsets,
        reset_chunk_pos_per_seq,
        _k_run_starts,
        _k_run_lengths,
        strict_past,
    )
    if output_state is not None:
        args = (*args, output_state)
    return _extension_ops()._flash_sca_sm90_varlen_fwd(*args, **output_options)


def _flash_sca_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    *,
    attn_method: str = "default",
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    position_offsets: Optional[Tensor] = None,
    strict_past: bool = False,
    output_state: Optional[Tensor] = None,
    output_fp32: bool = False,
) -> Tuple[Tensor, Tensor]:
    output_options = {"output_fp32": True} if output_fp32 else {}
    _check_attn_method_supported(attn_method)
    is_varlen = _validate_metadata_representation(
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
        prev_k,
        prev_v,
        reset_chunk_pos_per_seq,
    )
    _validate_qkv_heads(q, k, v, packed=False, api_name="flash_sca_fwd")
    _validate_prev_kv_heads(
        q,
        k,
        v,
        prev_k,
        prev_v,
        chunk_size,
        api_name="flash_sca_fwd",
    )
    if _explicit_metadata_is_dense(
        q,
        k,
        prev_k,
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
    ):
        q_segment_idx = None
        k_segment_idx = None
        cu_seqlens_q = None
        cu_seqlens_k = None
        max_seqlen_q = None
        max_seqlen_k = None
        position_offsets = None
        is_varlen = False
    if is_varlen:
        q_packed, k_combined, v_combined = _pack_batched_varlen_inputs(
            q, k, v, prev_k, prev_v
        )
        (
            effective_position_offsets,
            k_run_starts,
            k_run_lengths,
            _,
            _,
        ) = _prepare_batched_varlen_metadata(
            q,
            k_combined,
            cu_seqlens_q,
            cu_seqlens_k,
            position_offsets,
            reset_chunk_pos_per_seq,
        )
        y_packed, lse_packed = _flash_sca_varlen_fwd(
            q_packed,
            k_combined,
            v_combined,
            chunk_size,
            scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            position_offsets=effective_position_offsets,
            backend=backend,
            reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
            _k_run_starts=k_run_starts,
            _k_run_lengths=k_run_lengths,
            strict_past=strict_past,
            **output_options,
            output_state=(
                None
                if output_state is None
                else output_state.flatten(0, 1)
            ),
        )
        return (
            _restore_varlen_output(y_packed, q, v),
            _restore_varlen_lse(lse_packed, q),
        )
    operation = (
        _extension_ops()._flash_sca_sm90_strict_past_fwd
        if strict_past
        else _extension_ops().flash_sca_fwd
    )
    args = (
        q,
        k,
        v,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        backend,
        reset_chunk_pos_per_seq,
    )
    if output_state is not None:
        args = (*args, output_state)
    return operation(*args, **output_options)


def _flash_sca_varlen_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    chunk_size: int,
    scale: float,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    deterministic: bool = False,
    position_offsets: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    attn_method: str = "default",
    _k_run_starts: Optional[Tensor] = None,
    _k_run_lengths: Optional[Tensor] = None,
    _k_prefix_ends: Optional[Tensor] = None,
    _k_row_length: int = 0,
    strict_past: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run backward on packed varlen Q/K/V."""
    _check_attn_method_supported(attn_method)
    _validate_qkv_heads(
        q, k, v, packed=True, api_name="_flash_sca_varlen_bwd"
    )
    if backend.strip().lower() not in ("", "auto", "sm90"):
        raise ValueError(
            "_flash_sca_varlen_bwd requires backend='auto'/'sm90'"
        )
    dense_shape = _uniform_packed_dense_shape(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
        chunk_size,
        reset_chunk_pos_per_seq,
        (
            _k_run_starts is not None
            or _k_run_lengths is not None
            or _k_prefix_ends is not None
            or _k_row_length != 0
        ),
    )
    if (
        dense_shape is not None
        and y_grad.is_contiguous()
        and y.is_contiguous()
        and lse.is_contiguous()
    ):
        batch, sequence_length = dense_shape
        q_dense = q.view(batch, sequence_length, *q.shape[1:])
        k_dense = k.view(batch, sequence_length, *k.shape[1:])
        v_dense = v.view(batch, sequence_length, *v.shape[1:])
        y_grad_dense = y_grad.view(
            batch, sequence_length, *y_grad.shape[1:]
        )
        y_dense = y.view(batch, sequence_length, *y.shape[1:])
        lse_dense = (
            lse.view(q.shape[1], batch, sequence_length)
            .permute(1, 0, 2)
            .contiguous()
        )
        dq, dk, dv, _, _ = _flash_sca_bwd(
            y_grad_dense,
            q_dense,
            k_dense,
            v_dense,
            y_dense,
            lse_dense,
            chunk_size,
            scale,
            deterministic=deterministic,
            backend=backend,
            reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
            attn_method=attn_method,
            strict_past=strict_past,
        )
        return (
            dq.view_as(q),
            dk.view_as(k),
            dv.view_as(v),
        )
    return _extension_ops()._flash_sca_sm90_varlen_bwd(
        y_grad,
        q,
        k,
        v,
        y,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        chunk_size,
        scale,
        deterministic,
        position_offsets,
        reset_chunk_pos_per_seq,
        _k_run_starts,
        _k_run_lengths,
        _k_prefix_ends,
        _k_row_length,
        strict_past,
    )


def _flash_sca_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    deterministic: bool = False,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    *,
    attn_method: str = "default",
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    position_offsets: Optional[Tensor] = None,
    strict_past: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    _check_attn_method_supported(attn_method)
    is_varlen = _validate_metadata_representation(
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
        prev_k,
        prev_v,
        reset_chunk_pos_per_seq,
    )
    _validate_qkv_heads(q, k, v, packed=False, api_name="flash_sca_bwd")
    _validate_prev_kv_heads(
        q,
        k,
        v,
        prev_k,
        prev_v,
        chunk_size,
        api_name="flash_sca_bwd",
    )
    if _explicit_metadata_is_dense(
        q,
        k,
        prev_k,
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
    ):
        q_segment_idx = None
        k_segment_idx = None
        cu_seqlens_q = None
        cu_seqlens_k = None
        max_seqlen_q = None
        max_seqlen_k = None
        position_offsets = None
        is_varlen = False
    if is_varlen:
        q_packed, k_combined, v_combined = _pack_batched_varlen_inputs(
            q, k, v, prev_k, prev_v
        )
        (
            effective_position_offsets,
            k_run_starts,
            k_run_lengths,
            k_prefix_ends,
            k_row_length,
        ) = _prepare_batched_varlen_metadata(
            q,
            k_combined,
            cu_seqlens_q,
            cu_seqlens_k,
            position_offsets,
            reset_chunk_pos_per_seq,
        )
        y_grad_packed, y_packed, lse_packed = _pack_varlen_backward_tensors(
            y_grad, y, lse, q
        )
        dq_packed, dk_combined, dv_combined = _flash_sca_varlen_bwd(
            y_grad_packed,
            q_packed,
            k_combined,
            v_combined,
            y_packed,
            lse_packed,
            chunk_size,
            scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            deterministic=deterministic,
            position_offsets=effective_position_offsets,
            backend=backend,
            reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
            _k_run_starts=k_run_starts,
            _k_run_lengths=k_run_lengths,
            _k_prefix_ends=k_prefix_ends,
            _k_row_length=k_row_length,
            strict_past=strict_past,
        )
        return _unpack_batched_varlen_grads(
            dq_packed,
            dk_combined,
            dv_combined,
            q,
            k,
            v,
            prev_k,
            prev_v,
        )
    operation = (
        _extension_ops()._flash_sca_sm90_strict_past_bwd
        if strict_past
        else _extension_ops().flash_sca_bwd
    )
    return operation(
        y_grad,
        q,
        k,
        v,
        y,
        lse,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        deterministic,
        backend,
        reset_chunk_pos_per_seq,
    )


def _make_fwd_metadata_plan(
    route: str,
    chunk_size: int,
    reset_chunk_pos_per_seq: bool,
    *,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    position_offsets: Optional[Tensor] = None,
    k_run_starts: Optional[Tensor] = None,
    k_run_lengths: Optional[Tensor] = None,
) -> _FlashSCAFwdMetadataPlan:
    return _FlashSCAFwdMetadataPlan(
        route,
        chunk_size,
        reset_chunk_pos_per_seq,
        q_segment_idx,
        k_segment_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        position_offsets,
        k_run_starts,
        k_run_lengths,
    )


def _prepare_fwd_metadata_plan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    bos_mask: Tensor,
    backend: str,
    reset_chunk_pos_per_seq: bool,
) -> _FlashSCAFwdMetadataPlan:
    if backend.strip().lower() not in ("", "auto", "sm90"):
        raise ValueError(
            "FlashSCA FWD metadata plans require backend='auto'/'sm90'"
        )
    if (prev_k is None) != (prev_v is None):
        raise ValueError("prev_k and prev_v must be provided together")
    _validate_qkv_heads(
        q, k, v, packed=False, api_name="flash_sca_fwd"
    )
    _validate_prev_kv_heads(
        q,
        k,
        v,
        prev_k,
        prev_v,
        chunk_size,
        api_name="flash_sca_fwd",
    )
    if not flash_sca_sm90_available(q):
        raise RuntimeError("FlashSCA FWD metadata plans require SM90")
    k_length = k.shape[1] + (0 if prev_k is None else prev_k.shape[1])
    _validate_bos_mask(bos_mask, q, k_length, "bos_mask")
    if q.shape[1] > k_length:
        raise ValueError("bos_mask cannot right-align Q beyond its K layout")
    with torch.inference_mode(False):
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            num_q_runs,
            num_k_runs,
            metadata_is_dense,
        ) = _build_fwd_bos_metadata(q, bos_mask, k_length)
        if metadata_is_dense:
            return _make_fwd_metadata_plan(
                "dense",
                chunk_size,
                reset_chunk_pos_per_seq,
            )
        fwd_use_varlen, _ = _select_bos_routes(
            q,
            k,
            v,
            chunk_size,
            num_q_runs,
            max_seqlen_q,
            False,
            False,
            reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
            has_previous=prev_k is not None,
            high_precision_output=False,
            previous_length=0 if prev_k is None else prev_k.shape[1],
            num_k_runs=num_k_runs,
            max_seqlen_k=max_seqlen_k,
        )
        if not fwd_use_varlen:
            k_segment_idx = _bos_to_segment_idx(bos_mask)
            q_segment_idx = k_segment_idx[:, -q.shape[1] :]
            return _make_fwd_metadata_plan(
                "segment",
                chunk_size,
                reset_chunk_pos_per_seq,
                q_segment_idx=q_segment_idx,
                k_segment_idx=k_segment_idx,
            )
        k_row_length = k.shape[1] + (
            0 if prev_k is None else prev_k.shape[1]
        )
        (
            effective_position_offsets,
            k_run_starts,
            k_run_lengths,
            _,
        ) = _prepare_varlen_row_alignment_metadata(
            q.shape[0],
            q.shape[1],
            k_row_length,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            reset_chunk_pos_per_seq,
        )
    return _make_fwd_metadata_plan(
        "varlen",
        chunk_size,
        reset_chunk_pos_per_seq,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        position_offsets=effective_position_offsets,
        k_run_starts=k_run_starts,
        k_run_lengths=k_run_lengths,
    )


def _flash_sca_planned_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    backend: str,
    reset_chunk_pos_per_seq: bool,
    attn_method: str,
    plan: _FlashSCAFwdMetadataPlan,
) -> Tuple[Tensor, Tensor]:
    if attn_method != "default":
        _check_attn_method_supported(attn_method)
    if backend not in ("", "auto", "sm90") and backend.strip().lower() not in (
        "",
        "auto",
        "sm90",
    ):
        raise ValueError(
            "FlashSCA FWD metadata plans require backend='auto'/'sm90'"
        )
    if chunk_size != plan._chunk_size:
        raise ValueError("chunk_size does not match the metadata plan")
    if (
        reset_chunk_pos_per_seq
        != plan._reset_chunk_pos_per_seq
    ):
        raise ValueError(
            "reset_chunk_pos_per_seq does not match the metadata plan"
        )
    if plan._route != "varlen":
        return _extension_ops().flash_sca_fwd(
            q,
            k,
            v,
            chunk_size,
            scale,
            prev_k,
            prev_v,
            plan._q_segment_idx,
            plan._k_segment_idx,
            backend,
            reset_chunk_pos_per_seq,
        )

    q_packed, k_combined, v_combined = _pack_batched_varlen_inputs(
        q, k, v, prev_k, prev_v
    )
    y_packed, lse_packed = _extension_ops()._flash_sca_sm90_varlen_fwd(
        q_packed,
        k_combined,
        v_combined,
        plan._cu_seqlens_q,
        plan._cu_seqlens_k,
        plan._max_seqlen_q,
        plan._max_seqlen_k,
        chunk_size,
        scale,
        plan._position_offsets,
        reset_chunk_pos_per_seq,
        plan._k_run_starts,
        plan._k_run_lengths,
    )
    return (
        _restore_varlen_output(y_packed, q, v),
        _restore_varlen_lse(lse_packed, q),
    )


def _clear_inference_bos_plan_cache() -> None:
    ops = _extension_ops()
    clear = getattr(
        ops,
        "_flash_sca_sm90_clear_inference_bos_fwd_plan_cache",
        None,
    )
    if clear is not None:
        clear()


def _try_inference_bos_plan_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    bos_mask: Tensor,
    backend: str,
    reset_chunk_pos_per_seq: bool,
) -> Optional[Tuple[Tensor, Tensor]]:
    normalized_backend = backend.strip().lower()
    if normalized_backend not in ("", "auto", "sm90"):
        raise ValueError("BOS-mask FlashSCA requires backend='auto'/'sm90'")
    op = getattr(
        _extension_ops(),
        "_flash_sca_sm90_try_inference_bos_fwd_plan",
        None,
    )
    if op is None:
        return None
    return op(
        q,
        k,
        v,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        bos_mask,
        backend,
        reset_chunk_pos_per_seq,
    )


def _prepare_store_and_execute_inference_bos_plan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor],
    prev_v: Optional[Tensor],
    bos_mask: Tensor,
    backend: str,
    reset_chunk_pos_per_seq: bool,
    attn_method: str,
) -> Optional[Tuple[Tensor, Tensor]]:
    if not flash_sca_sm90_available(q):
        return None
    plan = _prepare_fwd_metadata_plan(
        q,
        k,
        v,
        chunk_size,
        prev_k,
        prev_v,
        bos_mask=bos_mask,
        backend=backend,
        reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
    )
    route = {"dense": 0, "segment": 1, "varlen": 2}[plan._route]
    plan_args = (
        route,
        plan._q_segment_idx,
        plan._k_segment_idx,
        plan._cu_seqlens_q,
        plan._cu_seqlens_k,
        -1 if plan._max_seqlen_q is None else plan._max_seqlen_q,
        -1 if plan._max_seqlen_k is None else plan._max_seqlen_k,
        plan._position_offsets,
        plan._k_run_starts,
        plan._k_run_lengths,
    )
    store_and_execute = getattr(
        _extension_ops(),
        "_flash_sca_sm90_store_and_execute_inference_bos_fwd_plan",
        None,
    )
    if store_and_execute is not None:
        return store_and_execute(
            q,
            k,
            v,
            chunk_size,
            scale,
            prev_k,
            prev_v,
            bos_mask,
            backend,
            reset_chunk_pos_per_seq,
            *plan_args,
        )
    store = getattr(
        _extension_ops(),
        "_flash_sca_sm90_store_inference_bos_fwd_plan",
        None,
    )
    if store is not None:
        store(
            q,
            k,
            v,
            chunk_size,
            prev_k,
            prev_v,
            bos_mask,
            reset_chunk_pos_per_seq,
            *plan_args,
        )
    return _flash_sca_planned_fwd(
        q,
        k,
        v,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        backend,
        reset_chunk_pos_per_seq,
        attn_method,
        plan,
    )


def flash_sca(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    deterministic: bool = False,
    attn_method: str = "default",
    high_precision_output: bool = False,
) -> Tensor:
    """Run FlashSCA with dense, segment-index, or BOS metadata.

    On SM90, Q may be shorter than current K/V. Positions align to the end
    of total K/V (including previous K/V). With reset positions, matching
    segment IDs instead align at their respective segment ends. Common
    ``segment_idx`` and ``bos_mask`` describe total K/V; Q uses its suffix.

    ``scale`` multiplies QK logits before softmax and defaults to
    ``1 / sqrt(q.shape[-1])``.

    ``high_precision_output=True`` saves FP32 attention state for backward.
    The public output keeps the input dtype.
    """
    if _needs_matched_reset(q, q_segment_idx, k_segment_idx,
                            reset_chunk_pos_per_seq, segment_idx, bos_mask):
        return _matched_reset_forward(
            q, k, v, chunk_size, scale, prev_k, prev_v,
            q_segment_idx, k_segment_idx, backend, attn_method,
            high_precision_output, deterministic, autograd=True,
        )
    return _flash_sca(
        q,
        k,
        v,
        chunk_size,
        scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        segment_idx=segment_idx,
        bos_mask=bos_mask,
        backend=backend,
        reset_chunk_pos_per_seq=reset_chunk_pos_per_seq,
        deterministic=deterministic,
        attn_method=attn_method,
        high_precision_output=high_precision_output,
    )


def flash_sca_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    attn_method: str = "default",
    high_precision_output: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Return ``(y, y_fp32, lse)``; ``y_fp32`` is None without high precision.

    ``y`` keeps the input dtype; ``y_fp32`` has the same shape and FP32 dtype.
    LSE is FP32. Both outputs are written by one forward, including in no_grad
    and inference mode. Pass ``y_fp32`` to backward for high-precision state;
    ``y_grad`` must retain the Q/K/V dtype. No autograd bridge is created.
    """
    if _needs_matched_reset(q, q_segment_idx, k_segment_idx,
                            reset_chunk_pos_per_seq, segment_idx, bos_mask):
        return _matched_reset_forward(
            q, k, v, chunk_size, scale, prev_k, prev_v,
            q_segment_idx, k_segment_idx, backend, attn_method,
            high_precision_output,
        )
    if not isinstance(high_precision_output, bool):
        raise TypeError("flash_sca_fwd high_precision_output must be a bool")
    if high_precision_output and q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("flash_sca_fwd FP32 output requires fp16/bf16 inputs")
    output_state = allocate_backward_output_state(q, v) if high_precision_output else None
    output_options = {"output_state": output_state} if output_state is not None else {}
    resolved_scale = _resolve_scale(q, scale) if scale is None else scale
    if bos_mask is not None:
        _check_attn_method_supported(attn_method)
        if (
            q_segment_idx is not None
            or k_segment_idx is not None
            or segment_idx is not None
        ):
            raise ValueError(
                "BOS masks and segment indices are mutually exclusive"
            )
        if (
            not high_precision_output
            and torch.is_inference(bos_mask)
            and q.is_cuda
            and bos_mask.is_cuda
        ):
            cached_result = _try_inference_bos_plan_fwd(
                q,
                k,
                v,
                chunk_size,
                resolved_scale,
                prev_k,
                prev_v,
                bos_mask,
                backend,
                reset_chunk_pos_per_seq,
            )
            if cached_result is not None:
                return cached_result[0], None, cached_result[1]
            result = _prepare_store_and_execute_inference_bos_plan(
                q,
                k,
                v,
                chunk_size,
                resolved_scale,
                prev_k,
                prev_v,
                bos_mask,
                backend,
                reset_chunk_pos_per_seq,
                attn_method,
            )
            if result is not None:
                return result[0], None, result[1]
        (
            q_segment_idx,
            k_segment_idx,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = _resolve_public_bos_route(
            q,
            k,
            v,
            chunk_size,
            prev_k,
            prev_v,
            bos_mask,
            backend,
            reset_chunk_pos_per_seq,
            deterministic=False,
            for_backward=False,
            api_name="flash_sca_fwd",
            high_precision_output=high_precision_output,
        )
        result = _flash_sca_fwd(
            q,
            k,
            v,
            chunk_size,
            resolved_scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            backend,
            reset_chunk_pos_per_seq,
            attn_method=attn_method,
            **output_options,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        return result[0], output_state, result[1]

    if segment_idx is not None:
        k_length = k.shape[1] + (
            0 if prev_k is None else prev_k.shape[1]
        )
        q_segment_idx, k_segment_idx = _resolve_segment_indices(
            segment_idx,
            q_segment_idx,
            k_segment_idx,
            batch_size=q.shape[0],
            q_length=q.shape[1],
            k_length=k_length,
        )
    result = _flash_sca_fwd(
        q,
        k,
        v,
        chunk_size,
        resolved_scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        backend,
        reset_chunk_pos_per_seq,
        attn_method=attn_method,
        **output_options,
    )

    return result[0], output_state, result[1]


def flash_sca_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    lse: Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    q_segment_idx: Optional[Tensor] = None,
    k_segment_idx: Optional[Tensor] = None,
    *,
    segment_idx: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    backend: str = "auto",
    reset_chunk_pos_per_seq: bool = False,
    deterministic: bool = False,
    attn_method: str = "default",
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Run FlashSCA BWD with dense, segment, or BOS metadata.

    ``y`` may have the input dtype or FP32; its dtype selects state precision.
    ``y_grad`` must retain the Q/K/V dtype.
    """
    resolved_scale = _resolve_scale(q, scale) if scale is None else scale
    if _needs_matched_reset(q, q_segment_idx, k_segment_idx,
                            reset_chunk_pos_per_seq, segment_idx, bos_mask):
        return _matched_reset_backward(
            y_grad, q, k, v, y, lse, chunk_size, scale, prev_k, prev_v,
            q_segment_idx, k_segment_idx, backend, attn_method, deterministic,
        )
    if bos_mask is not None:
        _check_attn_method_supported(attn_method)
        if (
            q_segment_idx is not None
            or k_segment_idx is not None
            or segment_idx is not None
        ):
            raise ValueError(
                "BOS masks and segment indices are mutually exclusive"
            )
        (
            q_segment_idx,
            k_segment_idx,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = _resolve_public_bos_route(
            q,
            k,
            v,
            chunk_size,
            prev_k,
            prev_v,
            bos_mask,
            backend,
            reset_chunk_pos_per_seq,
            deterministic=deterministic,
            for_backward=True,
            api_name="flash_sca_bwd",
            high_precision_output=y.dtype == torch.float32,
        )
        return _flash_sca_bwd(
            y_grad,
            q,
            k,
            v,
            y,
            lse,
            chunk_size,
            resolved_scale,
            prev_k,
            prev_v,
            q_segment_idx,
            k_segment_idx,
            deterministic,
            backend,
            reset_chunk_pos_per_seq,
            attn_method=attn_method,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

    if segment_idx is not None:
        k_length = k.shape[1] + (
            0 if prev_k is None else prev_k.shape[1]
        )
        q_segment_idx, k_segment_idx = _resolve_segment_indices(
            segment_idx,
            q_segment_idx,
            k_segment_idx,
            batch_size=q.shape[0],
            q_length=q.shape[1],
            k_length=k_length,
        )
    return _flash_sca_bwd(
        y_grad,
        q,
        k,
        v,
        y,
        lse,
        chunk_size,
        resolved_scale,
        prev_k,
        prev_v,
        q_segment_idx,
        k_segment_idx,
        deterministic,
        backend,
        reset_chunk_pos_per_seq,
        attn_method=attn_method,
    )


def flash_sca_sm90_available(q: Tensor) -> bool:
    return bool(_extension_ops().flash_sca_sm90_available(q))

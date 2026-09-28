import torch
from torch import Tensor


_INT32_MAX = 2**31 - 1


def validate_packed_qkv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    api_name: str,
) -> None:
    """Validate the variant-independent packed Q/K/V contract."""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"{api_name} expects packed 3D q/k/v tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"{api_name} q/k/v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"{api_name} q/k/v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"{api_name} supports only torch.float16 and "
            f"torch.bfloat16 input; got {q.dtype}"
        )
    if k.shape[0] != v.shape[0]:
        raise ValueError(f"{api_name} packed k/v token counts must match")
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"{api_name} k/v head counts must match")
    if q.shape[1] <= 0 or k.shape[1] <= 0:
        raise ValueError(f"{api_name} q and kv head counts must be positive")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"{api_name} q head count must be divisible by kv head count; "
            f"got Hq={q.shape[1]}, Hkv={k.shape[1]}"
        )
    if q.shape[2] != k.shape[2]:
        raise ValueError(f"{api_name} q/k head dims must match")
    if q.shape[2] <= 0 or v.shape[2] <= 0:
        raise ValueError(f"{api_name} q/k and value head dims must be positive")


def validate_varlen_metadata(
    q: Tensor,
    k: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    api_name: str,
) -> None:
    """Validate packed sequence metadata without synchronizing CUDA values."""
    if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
        raise ValueError(
            f"{api_name} cu_seqlens_q/cu_seqlens_k must be 1D"
        )
    if cu_seqlens_q.dtype != cu_seqlens_k.dtype:
        raise ValueError(
            f"{api_name} cu_seqlens_q/cu_seqlens_k must share a dtype"
        )
    if cu_seqlens_q.dtype != torch.int32:
        raise ValueError(f"{api_name} cu_seqlens must be torch.int32")
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        raise ValueError(
            f"{api_name} cu_seqlens must share the Q/K/V device"
        )
    if cu_seqlens_q.numel() < 2:
        raise ValueError(
            f"{api_name} cu_seqlens must contain at least one sequence"
        )
    if cu_seqlens_k.numel() < cu_seqlens_q.numel():
        raise ValueError(
            f"{api_name} right-aligned varlen requires at least as many "
            "K sequences as Q sequences"
        )
    if (
        isinstance(max_seqlen_q, bool)
        or not isinstance(max_seqlen_q, int)
        or isinstance(max_seqlen_k, bool)
        or not isinstance(max_seqlen_k, int)
    ):
        raise TypeError(
            f"{api_name} max_seqlen_q/max_seqlen_k must be integers"
        )
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError(
            f"{api_name} max_seqlen_q/max_seqlen_k must be positive"
        )
    if (
        q.shape[0] > _INT32_MAX
        or k.shape[0] > _INT32_MAX
        or max_seqlen_q > _INT32_MAX
        or max_seqlen_k > _INT32_MAX
    ):
        raise ValueError(f"{api_name} varlen sizes must fit int32")

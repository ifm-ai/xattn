from typing import Iterable, Optional

import torch
from torch import Tensor


def use_high_precision_output(
    requested: bool,
    tensors: Iterable[Optional[Tensor]],
    *,
    api_name: str,
) -> bool:
    """Resolve a public precision request to a training-only kernel switch."""
    if not isinstance(requested, bool):
        raise TypeError(
            f"{api_name} high_precision_output must be a bool"
        )
    return requested and torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def allocate_backward_output_state(q: Tensor, v: Tensor) -> Tensor:
    """Allocate the generic FP32 attention output state consumed by BWD."""
    return torch.empty(
        (*q.shape[:-1], v.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )

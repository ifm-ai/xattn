from typing import Iterable, Optional

import torch
from torch import Tensor


def validate_high_precision_level(level: int, *, api_name: str = "attention") -> int:
    """Validate the algorithm selector without accepting bool as an integer."""
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f"{api_name} high_precision_level must be an int in [0, 4]")
    if not 0 <= level <= 4:
        raise ValueError(f"{api_name} high_precision_level must be in [0, 4], got {level}")
    return level


def use_high_precision_level(
    requested: int,
    tensors: Iterable[Optional[Tensor]],
    *,
    api_name: str,
) -> int:
    """Keep the algorithm level intact, including standalone/inference forward.

    A level is not a state-width boolean: levels 1--4 all save FP32 state,
    but their backwards select different numerical algorithms.
    """
    return validate_high_precision_level(requested, api_name=api_name)


def allocate_backward_output_state(q: Tensor, v: Tensor) -> Tensor:
    """Allocate the generic FP32 attention output state consumed by BWD."""
    return torch.empty(
        (*q.shape[:-1], v.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )


def require_precision_backward(level: int) -> None:
    """Prevent advanced levels from entering a legacy packed backward bridge."""
    validate_high_precision_level(level)
    if level >= 2:
        raise NotImplementedError(
            f"high_precision_level={level} is unsupported by this legacy packed bridge; "
            "use the public precision-level API instead of a legacy packed bridge"
        )


def validate_backward_precision(level: int, state: Tensor, q: Tensor) -> None:
    validate_high_precision_level(level)
    expected = torch.float32 if level else q.dtype
    if state.dtype != expected:
        raise ValueError(
            f"high_precision_level={level} requires saved state dtype {expected}, "
            f"got {state.dtype}; pass the same level to forward and backward"
        )


def validate_attention_saved_state(state: Tensor, lse: Tensor, q: Tensor, v: Tensor) -> None:
    """Validate standalone reader state before dispatching any native kernel."""
    if q.dim() != 4 or v.dim() != 4:
        raise ValueError("attention Q/V must be rank-four tensors")
    if state.shape != (*q.shape[:-1], v.shape[-1]) or state.device != q.device:
        raise ValueError("saved attention output has incompatible shape or device")
    if lse.shape != (q.shape[0], q.shape[2], q.shape[1]) or lse.device != q.device or lse.dtype != torch.float32:
        raise ValueError("attention LSE must be FP32 [B,Hq,Nq] on Q's device")


def _dispatch_nonpositive_scale(function, arguments):
    """Normalize finite nonpositive scales before masked softmax dispatch.

    Masked softmax kernels require a positive scale. Negation and zeroing of
    Q are exact for finite FP16/BF16 inputs. Autograd sees the Q transform;
    standalone backward needs the chain-rule sign applied explicitly.
    """
    options = dict(arguments)
    scale = float(options['scale'])
    q = options['q']
    options['q'] = -q if scale < 0 else q * 0
    options['scale'] = -scale if scale < 0 else 1.0
    result = function(**options)
    if function.__name__.endswith('_bwd'):
        dq = -result[0] if scale < 0 else torch.zeros_like(result[0])
        result = (dq, *result[1:])
    return result

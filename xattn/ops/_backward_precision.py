import torch
from torch import Tensor


def validate_high_precision_level(level: int, *, api_name: str = "attention") -> int:
    """Validate the precision selector: 0, or 1 for the high-precision softmax and
    the precision-repair backward. Bool is not accepted as an integer."""
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f"{api_name} high_precision_level must be 0 or 1")
    if level not in (0, 1):
        raise ValueError(f"{api_name} high_precision_level must be 0 or 1, got {level}")
    return level


def validate_attention_output(y: Tensor, q: Tensor) -> None:
    """A standalone backward takes the forward's output ``y`` in the Q/K/V dtype."""
    if y.dtype != q.dtype:
        raise ValueError(f"attention output y must have the Q/K/V dtype {q.dtype}, got {y.dtype}")


def allocate_backward_output_state(q: Tensor, v: Tensor) -> Tensor:
    """An FP32 output buffer [*q.shape[:-1], V] for a high-precision forward."""
    return torch.empty(
        (*q.shape[:-1], v.shape[-1]),
        dtype=torch.float32,
        device=q.device,
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

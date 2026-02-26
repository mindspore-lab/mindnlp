from ..._functional import abs, pow, sum, sqrt, amax, amin, stack, clamp, isnan, isinf, any, mul
from ..._creation import tensor


def _compute_norm(t, norm_type):
    """Compute norm of a single tensor."""
    if norm_type == float('inf'):
        return amax(abs(t))
    if norm_type == float('-inf'):
        return amin(abs(t))
    # For p-norm: (sum(|x|^p))^(1/p)
    return pow(sum(pow(abs(t), norm_type)), 1.0 / norm_type)


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """
    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters: An iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm: Max norm of the gradients
        norm_type: Type of the used p-norm. Can be 'inf' for infinity norm.
        error_if_nonfinite: If True, an error is thrown if the total norm is nan, inf, or -inf.
        foreach: Not used, for API compatibility only

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if hasattr(parameters, 'grad'):
        parameters = [parameters]
    elif isinstance(parameters, dict):
        parameters = parameters.values()

    parameters = [p for p in parameters if hasattr(p, 'grad') and p.grad is not None]

    if len(parameters) == 0:
        return tensor(0.0)

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Compute per-parameter norms
    norms = []
    for p in parameters:
        param_norm = _compute_norm(p.grad, norm_type)
        norms.append(param_norm)

    # Compute total norm
    if norm_type == float('inf'):
        total_norm = amax(stack(norms))
    elif norm_type == float('-inf'):
        total_norm = amin(stack(norms))
    else:
        # Total norm: (sum of all param_norms^norm_type)^(1/norm_type)
        total_norm = pow(sum(stack([pow(n, norm_type) for n in norms])), 1.0 / norm_type)

    if error_if_nonfinite:
        if any(isnan(total_norm)) or any(isinf(total_norm)):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )

    # Compute clipping coefficient
    # clip_coef = max_norm / (total_norm + 1e-6), clamped to max of 1.0
    inv_norm = pow(total_norm + tensor(1e-6), -1.0)
    clip_coef = clamp(mul(tensor(max_norm), inv_norm), max_val=1.0)

    # Scale all gradients in-place
    for p in parameters:
        p.grad = p.grad * clip_coef

    return total_norm


def clip_grad_value_(parameters, clip_value):
    """
    Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Args:
        parameters: An iterable of Tensors or a single Tensor that will have gradients clipped
        clip_value: Maximum allowed value of the gradients. The gradients are clipped in the
            range [-clip_value, clip_value]
    """
    if hasattr(parameters, 'grad'):
        parameters = [parameters]
    elif isinstance(parameters, dict):
        parameters = parameters.values()

    clip_value = float(clip_value)

    for p in parameters:
        if hasattr(p, 'grad') and p.grad is not None:
            p.grad = clamp(p.grad, min_val=-clip_value, max_val=clip_value)

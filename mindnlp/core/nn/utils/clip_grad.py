"""clip grad"""
# mypy: allow-untyped-defs
import functools
from typing import Union, Iterable, Optional
from typing_extensions import deprecated

import mindspore
from ... import ops
from ...autograd import no_grad

_tensor_or_tensors = Union[mindspore.Tensor, Iterable[mindspore.Tensor]]

__all__ = ['clip_grad_norm_', 'clip_grad_norm', 'clip_grad_value_']

inf = float('inf')

def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """
    def _no_grad_wrapper(*args, **kwargs):
        with no_grad():
            return func(*args, **kwargs)
    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


@_no_grad
def clip_grad_norm_(
        gradients: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> mindspore.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    grads = gradients
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return mindspore.tensor(0.)
    if norm_type == inf:
        norms = [g.abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else ops.max(ops.stack(norms))
    else:
        total_norm = ops.norm(ops.stack([ops.norm(g, norm_type) for g in grads]), norm_type)
    if error_if_nonfinite and ops.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = ops.clamp(clip_coef, max=1.0)
    for g in grads:
        ops.assign(g, ops.mul(g, clip_coef_clamped))
    return total_norm



@deprecated(
    "`nn.utils.clip_grad_norm` is now deprecated "
    "in favor of `nn.utils.clip_grad_norm_`.",
    category=FutureWarning,
)
def clip_grad_norm(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.,
        error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> mindspore.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`nn.utils.clip_grad_norm_`.
    """
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)




@_no_grad
def clip_grad_value_(gradients: _tensor_or_tensors, clip_value: float, foreach: Optional[bool] = None) -> None:
    r"""Clip the gradients of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        foreach (bool): use the faster foreach-based implementation
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and
            silently fall back to the slow implementation for other device types.
            Default: ``None``
    """
    clip_value = float(clip_value)
    for grad in gradients:
        ops.assign(grad, ops.clamp(grad, -clip_value, clip_value))

"""distribution utils"""
# mypy: allow-untyped-defs
from functools import update_wrapper
from numbers import Number
from typing import Any, Dict

import mindspore
from .. import ops
from ..autograd import enable_grad
from ..utils import get_default_dtype
from ..nn import functional as F


euler_constant = 0.57721566490153286060  # Euler Mascheroni Constant

__all__ = [
    "broadcast_all",
    "logits_to_probs",
    "clamp_probs",
    "probs_to_logits",
    "lazy_property",
    "tril_matrix_to_vec",
    "vec_to_tril_matrix",
]


def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number`, `torch.*Tensor` or objects implementing __torch_function__)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` instance,
            a `torch.*Tensor` instance, or an instance implementing __torch_function__
    """
    if not all(isinstance(v, (mindspore.Tensor, Number)) for v in values):
        raise ValueError(
            "Input arguments must all be instances of numbers.Number, "
            "mindspore.Tensor or objects implementing __torch_function__."
        )
    if not all(isinstance(v, mindspore.Tensor) for v in values):
        options: Dict[str, Any] = {"dtype": get_default_dtype()}
        for value in values:
            if isinstance(value, mindspore.Tensor):
                options = {"dtype": value.dtype}
                break
        new_values = [
            v if isinstance(v, mindspore.Tensor) else mindspore.tensor(v, **options) for v in values
        ]
        return ops.broadcast_tensors(*new_values)
    return ops.broadcast_tensors(*values)


def _standard_normal(shape, dtype):
    return ops.normal(size = shape).to(dtype)


def _sum_rightmost(value, dim):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits, is_binary=False):
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return ops.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def clamp_probs(probs):
    """Clamps the probabilities to be in the open interval `(0, 1)`.

    The probabilities would be clamped between `eps` and `1 - eps`,
    and `eps` would be the smallest representable positive number for the input data type.

    Args:
        probs (Tensor): A tensor of probabilities.

    Returns:
        Tensor: The clamped probabilities.

    Examples:
        >>> probs = mindspore.tensor([0.0, 0.5, 1.0])
        >>> clamp_probs(probs)
        tensor([1.1921e-07, 5.0000e-01, 1.0000e+00])

        >>> probs = mindspore.tensor([0.0, 0.5, 1.0], dtype=mindspore.float64)
        >>> clamp_probs(probs)
        tensor([2.2204e-16, 5.0000e-01, 1.0000e+00], dtype=mindspore.float64)

    """
    eps = ops.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def probs_to_logits(probs, is_binary=False):
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return ops.log(ps_clamped) - ops.log1p(-ps_clamped)
    return ops.log(ps_clamped)


class lazy_property:
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)  # type:ignore[arg-type]

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return _lazy_property_and_property(self.wrapped)
        with enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


class _lazy_property_and_property(lazy_property, property):
    """We want lazy properties to look like multiple things.

    * property when Sphinx autodoc looks
    * lazy_property when Distribution validate_args looks
    """


def tril_matrix_to_vec(mat: mindspore.Tensor, diag: int = 0) -> mindspore.Tensor:
    r"""
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    n = mat.shape[-1]
    # if not torch._C._get_tracing_state() and (diag < -n or diag >= n):
    #     raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n-1}].")
    arange = ops.arange(n)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    vec = mat[..., tril_mask]
    return vec


def vec_to_tril_matrix(vec: mindspore.Tensor, diag: int = 0) -> mindspore.Tensor:
    r"""
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # +ve root of D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0
    n = (
        -(1 + 2 * diag)
        + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5
    ) / 2
    eps = ops.finfo(vec.dtype).eps
    # if not torch._C._get_tracing_state() and (round(n) - n > eps):
    #     raise ValueError(
    #         f"The size of last dimension is {vec.shape[-1]} which cannot be expressed as "
    #         + "the lower triangular part of a square D x D matrix."
    #     )
    n = round(n.item()) if isinstance(n, mindspore.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + (n, n))
    arange = ops.arange(n)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat

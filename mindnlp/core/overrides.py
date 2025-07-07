from typing import Callable, Iterable, Any
from mindnlp import core

def is_tensor_like(inp):
    """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__torch_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(torch.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __torch_function__.

    >>> class TensorLike:
    ...     @classmethod
    ...     def __torch_function__(cls, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """
    return type(inp) is core.Tensor or hasattr(inp, "__torch_function__")

def handle_torch_function(
    public_api: Callable,
    relevant_args: Iterable[Any],
    *args,
    **kwargs,
) -> Any:
    """Implement a function with checks for ``__torch_function__`` overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
    public_api : function
        Function exposed by the public torch API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __torch_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    object
        Result from calling ``implementation`` or an ``__torch_function__``
        method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    Example
    -------
    >>> def func(a):
    ...     if has_torch_function_unary(a):
    ...         return handle_torch_function(func, (a,), a)
    ...     return a + 0
    """
    # Check for __torch_function__ methods.
    pass

def has_torch_function(inp):
    return hasattr(inp, "__torch_function__")
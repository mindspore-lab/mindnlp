"""PyTorch-compatible tensor creation functions."""

import numpy as np
import mindspore
from . import _dtype as dtype_mod
from ._tensor import Tensor
from ._storage import TypedStorage


def _resolve_dtype(dtype, default=None):
    """Resolve dtype or return default."""
    if dtype is not None:
        return dtype
    return default or dtype_mod.float32


def _resolve_shape(args):
    """Accept shape as *args or tuple: zeros(3, 4) or zeros((3, 4))."""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return tuple(args)


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    """Create tensor from data (always copies). Equivalent to torch.tensor()."""
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with zeros."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.zeros(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def ones(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with ones."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.ones(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def empty(*size, dtype=None, device=None, requires_grad=False):
    """Create uninitialized tensor."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.empty(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def full(size, fill_value, *, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with fill_value."""
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.full(size, fill_value, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def arange(*args, dtype=None, device=None, requires_grad=False):
    """arange(end), arange(start, end), arange(start, end, step)."""
    arr = np.arange(*args)
    if dtype is not None:
        np_dtype = dtype_mod.dtype_to_numpy(dtype)
        arr = arr.astype(np_dtype)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)


def linspace(start, end, steps, *, dtype=None, device=None, requires_grad=False):
    """Create evenly spaced tensor."""
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.linspace(start, end, steps, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def eye(n, m=None, *, dtype=None, device=None, requires_grad=False):
    """Create identity matrix."""
    if m is None:
        m = n
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.eye(n, m, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def randn(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor with random normal values."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
    arr = np.random.randn(*shape).astype(np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def rand(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor with random uniform [0, 1) values."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
    arr = np.random.rand(*shape).astype(np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def zeros_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create zero tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return zeros(*input.shape, dtype=dt, device=device or str(input.device),
                 requires_grad=requires_grad)


def ones_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create ones tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return ones(*input.shape, dtype=dt, device=device or str(input.device),
                requires_grad=requires_grad)


def empty_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create empty tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return empty(*input.shape, dtype=dt, device=device or str(input.device),
                 requires_grad=requires_grad)


def from_numpy(ndarray):
    """Create tensor from numpy array (shares memory where possible)."""
    arr = np.ascontiguousarray(ndarray)
    dt = dtype_mod.numpy_to_dtype(arr.dtype)
    return Tensor(arr, dtype=dt)


def randint(low, high=None, size=None, *, dtype=None, device=None, requires_grad=False):
    """Create tensor with random integers in [low, high).

    Args:
        low: Lowest integer (inclusive), or high if high is None
        high: One above highest integer (exclusive)
        size: Shape of output tensor
    """
    if high is None:
        high = low
        low = 0
    if size is None:
        size = ()
    if isinstance(size, int):
        size = (size,)

    dt = dtype or dtype_mod.int64
    arr = np.random.randint(low, high, size=size)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)

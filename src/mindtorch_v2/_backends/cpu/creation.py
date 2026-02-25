import numpy as np

from ..._dtype import to_numpy_dtype
from ..._storage import typed_storage_from_numpy
from ..._tensor import Tensor


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def tensor_create(data, dtype=None, device=None, requires_grad=False, memory_format=None):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride, requires_grad=requires_grad)


def zeros_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.zeros(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def ones_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.ones(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def empty_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.empty(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def arange_create(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def linspace_create(start, end, steps, dtype=None, device=None):
    arr = np.linspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def full_create(shape, fill_value, dtype=None, device=None):
    shape = tuple(shape)
    storage = typed_storage_from_numpy(
        np.full(shape, fill_value, dtype=to_numpy_dtype(dtype)),
        dtype,
        device=device,
    )
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)


def logspace_create(start, end, steps, dtype=None, device=None):
    arr = np.logspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def eye_create(n, m=None, dtype=None, device=None):
    if m is None:
        m = n
    arr = np.eye(n, m, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def range_create(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end + step, step, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def randn_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = rng.randn(*shape).astype(to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)

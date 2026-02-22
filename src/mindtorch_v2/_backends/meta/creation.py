import numpy as np

from ..._dtype import to_numpy_dtype
from ..._storage import meta_typed_storage_from_shape
from ..._tensor import Tensor


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def tensor_create_meta(data, dtype=None, device=None, requires_grad=False):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return Tensor(storage, arr.shape, stride, requires_grad=requires_grad)


def zeros_create_meta(shape, dtype=None, device=None, requires_grad=False):
    shape = tuple(shape)
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def ones_create_meta(shape, dtype=None, device=None, requires_grad=False):
    shape = tuple(shape)
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def empty_create_meta(shape, dtype=None, device=None, requires_grad=False):
    shape = tuple(shape)
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def arange_create_meta(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return Tensor(storage, arr.shape, stride)


def linspace_create_meta(start, end, steps, dtype=None, device=None):
    arr = np.linspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return Tensor(storage, arr.shape, stride)


def full_create_meta(shape, fill_value, dtype=None, device=None):
    shape = tuple(shape)
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return Tensor(storage, shape, stride)


__all__ = [
    "tensor_create_meta",
    "zeros_create_meta",
    "ones_create_meta",
    "empty_create_meta",
    "arange_create_meta",
    "linspace_create_meta",
    "full_create_meta",
]

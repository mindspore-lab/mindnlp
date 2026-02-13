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


def tensor_create(data, dtype=None, device=None):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def zeros_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.zeros(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)


def ones_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.ones(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)


def empty_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.empty(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)

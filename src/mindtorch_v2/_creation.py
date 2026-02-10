import numpy as np

from ._dtype import float32, to_numpy_dtype
from ._storage import Storage
from ._tensor import Tensor


def tensor(data, dtype=float32):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    storage = Storage(arr, dtype=dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def zeros(shape, dtype=float32):
    arr = np.zeros(shape, dtype=to_numpy_dtype(dtype))
    storage = Storage(arr, dtype=dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def ones(shape, dtype=float32):
    arr = np.ones(shape, dtype=to_numpy_dtype(dtype))
    storage = Storage(arr, dtype=dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)

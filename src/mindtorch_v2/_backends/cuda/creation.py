import numpy as np

from ..._dtype import to_numpy_dtype
from ..._storage import cuda_typed_storage_from_numpy, empty_cuda_typed_storage
from ..._tensor import Tensor


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


def tensor_create(data, dtype=None, device=None, requires_grad=False, memory_format=None):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    storage = cuda_typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride, requires_grad=requires_grad)


def zeros_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    arr = np.zeros(shape, dtype=to_numpy_dtype(dtype))
    storage = cuda_typed_storage_from_numpy(arr, dtype, device=device)
    return Tensor(storage, shape, _contiguous_stride(shape), requires_grad=requires_grad)


def ones_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    arr = np.ones(shape, dtype=to_numpy_dtype(dtype))
    storage = cuda_typed_storage_from_numpy(arr, dtype, device=device)
    return Tensor(storage, shape, _contiguous_stride(shape), requires_grad=requires_grad)


def empty_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    storage = empty_cuda_typed_storage(shape, dtype, device=device)
    return Tensor(storage, shape, _contiguous_stride(shape), requires_grad=requires_grad)


def full_create(shape, fill_value, dtype=None, device=None):
    shape = tuple(shape)
    arr = np.full(shape, fill_value, dtype=to_numpy_dtype(dtype))
    storage = cuda_typed_storage_from_numpy(arr, dtype, device=device)
    return Tensor(storage, shape, _contiguous_stride(shape))

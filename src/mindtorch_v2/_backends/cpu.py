import numpy as np

from .._dispatch.registry import registry
from .._storage import Storage
from .._tensor import Tensor


def _from_numpy(arr, dtype):
    storage = Storage(arr, dtype=dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def add(a, b):
    return _from_numpy(a.storage.data + b.storage.data, a.dtype)


def mul(a, b):
    return _from_numpy(a.storage.data * b.storage.data, a.dtype)


def matmul(a, b):
    return _from_numpy(a.storage.data @ b.storage.data, a.dtype)


def relu(a):
    return _from_numpy(np.maximum(a.storage.data, 0), a.dtype)


def sum_(a, dim=None, keepdim=False):
    return _from_numpy(a.storage.data.sum(axis=dim, keepdims=keepdim), a.dtype)


registry.register("add", "cpu", add)
registry.register("mul", "cpu", mul)
registry.register("matmul", "cpu", matmul)
registry.register("relu", "cpu", relu)
registry.register("sum", "cpu", sum_)

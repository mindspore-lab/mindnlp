import numpy as np

from .._storage import Storage


def reduce_grad(grad, shape):
    arr = grad.storage.data
    while arr.ndim > len(shape):
        arr = arr.sum(axis=0)
    for i, (g_dim, s_dim) in enumerate(zip(arr.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            arr = arr.sum(axis=i, keepdims=True)
    storage = Storage(arr, device=grad.device, dtype=grad.dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    from .._tensor import Tensor
    return Tensor(storage, arr.shape, stride)

import numpy as np

from ..._storage import cuda_typed_storage_from_numpy, cuda_typed_storage_to_numpy
from ..._tensor import Tensor


def _from_numpy(arr, dtype, device):
    storage = cuda_typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def add(a, b):
    a_np = cuda_typed_storage_to_numpy(a.storage(), a.shape, a.dtype)
    b_np = cuda_typed_storage_to_numpy(b.storage(), b.shape, b.dtype) if isinstance(b, Tensor) else b
    out = np.ascontiguousarray(a_np + b_np)
    return _from_numpy(out, a.dtype, a.device)

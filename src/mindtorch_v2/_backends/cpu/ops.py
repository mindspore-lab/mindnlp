import numpy as np

from ..._storage import typed_storage_from_numpy
from ..._tensor import Tensor


def _to_numpy(t):
    return t._numpy_view()


def _from_numpy(arr, dtype, device):
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def add(a, b):
    return _from_numpy(_to_numpy(a) + _to_numpy(b), a.dtype, a.device)


def mul(a, b):
    return _from_numpy(_to_numpy(a) * _to_numpy(b), a.dtype, a.device)


def matmul(a, b):
    return _from_numpy(_to_numpy(a) @ _to_numpy(b), a.dtype, a.device)


def relu(a):
    return _from_numpy(np.maximum(_to_numpy(a), 0), a.dtype, a.device)


def sum_(a, dim=None, keepdim=False):
    return _from_numpy(_to_numpy(a).sum(axis=dim, keepdims=keepdim), a.dtype, a.device)


def add_(a, b):
    arr = _to_numpy(a)
    arr += _to_numpy(b)
    return a


def mul_(a, b):
    arr = _to_numpy(a)
    arr *= _to_numpy(b)
    return a


def relu_(a):
    arr = _to_numpy(a)
    np.maximum(arr, 0, out=arr)
    return a


def zero_(a):
    arr = _to_numpy(a)
    arr.fill(0)
    return a

def contiguous(a):
    if a.device.type != "cpu":
        raise ValueError("CPU contiguous expects CPU tensors")
    arr = np.ascontiguousarray(_to_numpy(a))
    return _from_numpy(arr, a.dtype, a.device)


def abs(a):
    return _from_numpy(np.abs(_to_numpy(a)), a.dtype, a.device)


def neg(a):
    return _from_numpy(np.negative(_to_numpy(a)), a.dtype, a.device)


def exp(a):
    return _from_numpy(np.exp(_to_numpy(a)), a.dtype, a.device)


def log(a):
    return _from_numpy(np.log(_to_numpy(a)), a.dtype, a.device)


def sqrt(a):
    return _from_numpy(np.sqrt(_to_numpy(a)), a.dtype, a.device)


def sin(a):
    return _from_numpy(np.sin(_to_numpy(a)), a.dtype, a.device)


def cos(a):
    return _from_numpy(np.cos(_to_numpy(a)), a.dtype, a.device)


def tan(a):
    return _from_numpy(np.tan(_to_numpy(a)), a.dtype, a.device)


def tanh(a):
    return _from_numpy(np.tanh(_to_numpy(a)), a.dtype, a.device)


def sigmoid(a):
    arr = _to_numpy(a)
    out = 1.0 / (1.0 + np.exp(-arr))
    return _from_numpy(out, a.dtype, a.device)

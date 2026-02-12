import numpy as np

from .._dispatch.registry import registry
from .._dtype import to_numpy_dtype
from .._storage import (
    meta_typed_storage_from_shape,
    typed_storage_from_numpy,
)
from .._tensor import Tensor


def _to_numpy(t):
    return t._numpy_view()


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def tensor_create_meta(data, dtype=None, device=None):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype)
    return Tensor(storage, arr.shape, stride)


def zeros_create_meta(shape, dtype=None, device=None):
    shape = tuple(shape)
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype)
    return Tensor(storage, shape, stride)


def ones_create_meta(shape, dtype=None, device=None):
    shape = tuple(shape)
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype)
    return Tensor(storage, shape, stride)


def empty_create_meta(shape, dtype=None, device=None):
    shape = tuple(shape)
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype)
    return Tensor(storage, shape, stride)


def tensor_create(data, dtype=None, device=None):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def zeros_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.zeros(shape, dtype=to_numpy_dtype(dtype)), dtype)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)


def ones_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.ones(shape, dtype=to_numpy_dtype(dtype)), dtype)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)


def empty_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.empty(shape, dtype=to_numpy_dtype(dtype)), dtype)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)


def _from_numpy(arr, dtype):
    storage = typed_storage_from_numpy(arr, dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def _broadcast_shape(a_shape, b_shape):
    try:
        return np.broadcast_shapes(a_shape, b_shape)
    except AttributeError:
        return np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape


def _meta_tensor(shape, dtype, device):
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype)
    return Tensor(storage, shape, stride)


def _meta_binary(a, b):
    shape = _broadcast_shape(a.shape, b.shape)
    arr = np.empty(shape, dtype=to_numpy_dtype(a.dtype))
    return _from_numpy(arr, a.dtype)


def _meta_unary(a):
    arr = np.empty(a.shape, dtype=to_numpy_dtype(a.dtype))
    return _from_numpy(arr, a.dtype)


def _meta_sum(a, dim=None, keepdim=False):
    shape = list(a.shape)
    if dim is None:
        dims = list(range(len(shape)))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    for d in sorted(dims):
        shape[d] = 1
    if not keepdim:
        shape = [s for i, s in enumerate(shape) if i not in dims]
    arr = np.empty(tuple(shape), dtype=to_numpy_dtype(a.dtype))
    return _from_numpy(arr, a.dtype)


def _meta_view(a, shape):
    arr = np.empty(tuple(shape), dtype=to_numpy_dtype(a.dtype))
    return _from_numpy(arr, a.dtype)


def _meta_transpose(a, dim0, dim1):
    shape = list(a.shape)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    arr = np.empty(tuple(shape), dtype=to_numpy_dtype(a.dtype))
    return _from_numpy(arr, a.dtype)


def _meta_binary_meta(a, b):
    shape = _broadcast_shape(a.shape, b.shape)
    return _meta_tensor(shape, a.dtype, a.device)


def _meta_unary_meta(a):
    return _meta_tensor(a.shape, a.dtype, a.device)


def _meta_sum_meta(a, dim=None, keepdim=False):
    shape = list(a.shape)
    if dim is None:
        dims = list(range(len(shape)))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    for d in sorted(dims):
        shape[d] = 1
    if not keepdim:
        shape = [s for i, s in enumerate(shape) if i not in dims]
    return _meta_tensor(tuple(shape), a.dtype, a.device)


def _meta_view_meta(a, shape):
    return _meta_tensor(tuple(shape), a.dtype, a.device)


def _meta_transpose_meta(a, dim0, dim1):
    shape = list(a.shape)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    return _meta_tensor(tuple(shape), a.dtype, a.device)


def _meta_matmul_meta(a, b):
    a_shape = a.shape
    b_shape = b.shape
    a_dim = len(a_shape)
    b_dim = len(b_shape)

    if a_dim == 1 and b_dim == 1:
        if a_shape[0] != b_shape[0]:
            raise ValueError("matmul shape mismatch")
        out_shape = ()
    elif a_dim == 1:
        k = a_shape[0]
        if b_dim < 2 or b_shape[-2] != k:
            raise ValueError("matmul shape mismatch")
        batch = b_shape[:-2]
        out_shape = batch + (b_shape[-1],)
    elif b_dim == 1:
        k = b_shape[0]
        if a_shape[-1] != k:
            raise ValueError("matmul shape mismatch")
        batch = a_shape[:-2]
        out_shape = batch + (a_shape[-2],)
    else:
        if a_shape[-1] != b_shape[-2]:
            raise ValueError("matmul shape mismatch")
        batch = _broadcast_shape(a_shape[:-2], b_shape[:-2])
        out_shape = batch + (a_shape[-2], b_shape[-1])
    return _meta_tensor(out_shape, a.dtype, a.device)


def add(a, b):
    return _from_numpy(_to_numpy(a) + _to_numpy(b), a.dtype)


def mul(a, b):
    return _from_numpy(_to_numpy(a) * _to_numpy(b), a.dtype)


def matmul(a, b):
    return _from_numpy(_to_numpy(a) @ _to_numpy(b), a.dtype)


def relu(a):
    return _from_numpy(np.maximum(_to_numpy(a), 0), a.dtype)


def sum_(a, dim=None, keepdim=False):
    return _from_numpy(_to_numpy(a).sum(axis=dim, keepdims=keepdim), a.dtype)


registry.register("add", "cpu", add, meta=_meta_binary)
registry.register("mul", "cpu", mul, meta=_meta_binary)
registry.register("matmul", "cpu", matmul)
registry.register("relu", "cpu", relu, meta=_meta_unary)
registry.register("sum", "cpu", sum_, meta=_meta_sum)
from .common import convert as convert_backend
from .common import view as view_backend


registry.register("reshape", "cpu", view_backend.reshape, meta=_meta_view)
registry.register("view", "cpu", view_backend.view, meta=_meta_view)
registry.register("transpose", "cpu", view_backend.transpose, meta=_meta_transpose)
registry.register("to", "cpu", convert_backend.to_device)
registry.register("add", "meta", _meta_binary_meta)
registry.register("mul", "meta", _meta_binary_meta)
registry.register("matmul", "meta", _meta_matmul_meta)
registry.register("relu", "meta", _meta_unary_meta)
registry.register("sum", "meta", _meta_sum_meta)
registry.register("reshape", "meta", view_backend.reshape, meta=_meta_view_meta)
registry.register("view", "meta", view_backend.view, meta=_meta_view_meta)
registry.register("transpose", "meta", view_backend.transpose, meta=_meta_transpose_meta)
registry.register("to", "meta", convert_backend.to_device)

registry.register("tensor", "cpu", tensor_create)
registry.register("zeros", "cpu", zeros_create)
registry.register("ones", "cpu", ones_create)
registry.register("empty", "cpu", empty_create)

registry.register("tensor", "meta", tensor_create_meta)
registry.register("zeros", "meta", zeros_create_meta)
registry.register("ones", "meta", ones_create_meta)
registry.register("empty", "meta", empty_create_meta)

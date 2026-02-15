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


def _meta_tensor(shape, dtype, device):
    stride = _contiguous_stride(shape)
    storage = meta_typed_storage_from_shape(shape, dtype)
    return Tensor(storage, shape, stride)


def _broadcast_shape(a_shape, b_shape):
    try:
        return np.broadcast_shapes(a_shape, b_shape)
    except AttributeError:
        return np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape


def _meta_binary_meta(a, b):
    shape = _broadcast_shape(a.shape, b.shape)
    return _meta_tensor(shape, a.dtype, a.device)


def _meta_binary_or_scalar_meta(a, b):
    if hasattr(b, "shape"):
        shape = _broadcast_shape(a.shape, b.shape)
    else:
        shape = a.shape
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



def _meta_contiguous_meta(a):
    return _meta_tensor(a.shape, a.dtype, a.device)


__all__ = [
    "_meta_binary_meta",
    "_meta_binary_or_scalar_meta",
    "_meta_matmul_meta",
    "_meta_sum_meta",
    "_meta_transpose_meta",
    "_meta_unary_meta",
    "_meta_view_meta",
    "_meta_contiguous_meta",
]

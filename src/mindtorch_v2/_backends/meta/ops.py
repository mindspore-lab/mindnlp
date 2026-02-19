import numpy as np

from ..._dtype import bool as bool_dtype
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


def _meta_where_meta(cond, x, y):
    if hasattr(cond, "shape"):
        cond_shape = cond.shape
    else:
        cond_shape = ()
    if hasattr(x, "shape"):
        x_shape = x.shape
    else:
        x_shape = ()
    if hasattr(y, "shape"):
        y_shape = y.shape
    else:
        y_shape = ()
    shape = _broadcast_shape(_broadcast_shape(cond_shape, x_shape), y_shape)
    return _meta_tensor(shape, x.dtype, x.device)


def _meta_unary_meta(a):
    return _meta_tensor(a.shape, a.dtype, a.device)


def _meta_unary_bool_meta(a):
    return _meta_tensor(a.shape, bool_dtype, a.device)


def _meta_clamp_meta(a, min_val=None, max_val=None):
    return _meta_unary_meta(a)


def _meta_clamp_min_meta(a, min_val):
    return _meta_unary_meta(a)


def _meta_clamp_max_meta(a, max_val):
    return _meta_unary_meta(a)


def _meta_hardtanh_meta(a, min_val=-1.0, max_val=1.0):
    return _meta_unary_meta(a)


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
    "_meta_where_meta",
    "_meta_matmul_meta",
    "_meta_sum_meta",
    "_meta_transpose_meta",
    "_meta_unary_meta",
    "_meta_unary_bool_meta",
    "_meta_clamp_meta",
    "_meta_clamp_min_meta",
    "_meta_clamp_max_meta",
    "_meta_hardtanh_meta",
    "_meta_view_meta",
    "_meta_contiguous_meta",
]

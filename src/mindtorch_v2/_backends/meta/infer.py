from dataclasses import dataclass
import numpy as np

from ..._dtype import bool as bool_dtype
from ..._dtype import int64 as int64_dtype


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    stride: tuple
    dtype: object
    offset: int = 0


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _broadcast_shape(a_shape, b_shape):
    try:
        return np.broadcast_shapes(a_shape, b_shape)
    except AttributeError:
        return np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape


def infer_binary(a, b):
    shape = _broadcast_shape(a.shape, b.shape)
    return TensorSpec(shape=tuple(shape), stride=_contiguous_stride(shape), dtype=a.dtype)


def infer_unary(a):
    return TensorSpec(shape=tuple(a.shape), stride=_contiguous_stride(a.shape), dtype=a.dtype)


def infer_unary_bool(a):
    return TensorSpec(shape=tuple(a.shape), stride=_contiguous_stride(a.shape), dtype=bool_dtype)


def infer_sum(a, dim=None, keepdim=False):
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
    shape = tuple(shape)
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype)


def infer_reduce_bool(a, dim=None, keepdim=False):
    spec = infer_sum(a, dim=dim, keepdim=keepdim)
    return TensorSpec(shape=spec.shape, stride=spec.stride, dtype=bool_dtype)


def infer_argmax(a, dim=None, keepdim=False):
    spec = infer_sum(a, dim=dim, keepdim=keepdim)
    return TensorSpec(shape=spec.shape, stride=spec.stride, dtype=int64_dtype)


def infer_cummax(a, dim=0):
    spec = infer_unary(a)
    indices = TensorSpec(shape=spec.shape, stride=spec.stride, dtype=int64_dtype)
    return (spec, indices)


def infer_view(a, shape):
    shape = tuple(shape)
    size = 1
    for d in a.shape:
        size *= d
    new_size = 1
    for d in shape:
        new_size *= d
    if size != new_size:
        raise ValueError("reshape size mismatch")
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype, offset=a.offset)


def infer_transpose(a, dim0, dim1):
    shape = list(a.shape)
    stride = list(a.stride)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    return TensorSpec(shape=tuple(shape), stride=tuple(stride), dtype=a.dtype, offset=a.offset)


def infer_matmul(a, b):
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
    return TensorSpec(shape=tuple(out_shape), stride=_contiguous_stride(out_shape), dtype=a.dtype)

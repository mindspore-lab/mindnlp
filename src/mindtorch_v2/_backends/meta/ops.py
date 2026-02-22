import numpy as np

from ..._dtype import bool as bool_dtype
from ..._dtype import int64 as int64_dtype
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


def _meta_binary_meta(a, b, **kwargs):
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


def _meta_lerp_meta(a, b, weight):
    if hasattr(weight, "shape"):
        w_shape = weight.shape
    else:
        w_shape = ()
    shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), w_shape)
    return _meta_tensor(shape, a.dtype, a.device)


def _meta_addcmul_meta(a, b, c, value=1.0):
    shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), c.shape)
    return _meta_tensor(shape, a.dtype, a.device)


def _meta_addcdiv_meta(a, b, c, value=1.0):
    shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), c.shape)
    return _meta_tensor(shape, a.dtype, a.device)


def _meta_unary_meta(a, *args, **kwargs):
    return _meta_tensor(a.shape, a.dtype, a.device)


def _meta_unary_bool_meta(a, *args, **kwargs):
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


def _meta_reduce_bool_meta(a, b=None, dim=None, keepdim=False, **kwargs):
    if dim is None and b is not None:
        return _meta_tensor((), bool_dtype, a.device)
    if dim is None:
        return _meta_tensor((), bool_dtype, a.device)
    out = _meta_sum_meta(a, dim=dim, keepdim=keepdim)
    return _meta_tensor(out.shape, bool_dtype, a.device)


def _meta_equal_meta(a, b, **kwargs):
    return _meta_tensor((), bool_dtype, a.device)


def _meta_cummax_meta(a, dim=0):
    return (
        _meta_tensor(a.shape, a.dtype, a.device),
        _meta_tensor(a.shape, int64_dtype, a.device),
    )


def _meta_argsort_meta(a, dim=-1, descending=False, stable=False):
    return _meta_tensor(a.shape, int64_dtype, a.device)


def _meta_sort_meta(a, dim=-1, descending=False, stable=False):
    return (
        _meta_tensor(a.shape, a.dtype, a.device),
        _meta_tensor(a.shape, int64_dtype, a.device),
    )


def _meta_topk_meta(a, k, dim=-1, largest=True, sorted=True):
    shape = list(a.shape)
    shape[dim] = k
    shape = tuple(shape)
    return (
        _meta_tensor(shape, a.dtype, a.device),
        _meta_tensor(shape, int64_dtype, a.device),
    )


def _meta_stack_meta(tensors, dim=0):
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    return _meta_tensor(tuple(shape), tensors[0].dtype, tensors[0].device)


def _meta_cat_meta(tensors, dim=0):
    shape = list(tensors[0].shape)
    shape[dim] = sum(t.shape[dim] for t in tensors)
    return _meta_tensor(tuple(shape), tensors[0].dtype, tensors[0].device)


def _meta_hstack_meta(tensors):
    if len(tensors[0].shape) == 1:
        shape = (sum(t.shape[0] for t in tensors),)
    else:
        shape = list(tensors[0].shape)
        shape[1] = sum(t.shape[1] for t in tensors)
        shape = tuple(shape)
    return _meta_tensor(shape, tensors[0].dtype, tensors[0].device)


def _meta_vstack_meta(tensors):
    if len(tensors[0].shape) == 1:
        shape = (len(tensors), tensors[0].shape[0])
    else:
        shape = list(tensors[0].shape)
        shape[0] = sum(t.shape[0] for t in tensors)
        shape = tuple(shape)
    return _meta_tensor(shape, tensors[0].dtype, tensors[0].device)


def _meta_column_stack_meta(tensors):
    if len(tensors[0].shape) == 1:
        shape = (tensors[0].shape[0], len(tensors))
    else:
        shape = list(tensors[0].shape)
        shape[1] = sum(t.shape[1] for t in tensors)
        shape = tuple(shape)
    return _meta_tensor(shape, tensors[0].dtype, tensors[0].device)


def _meta_pad_sequence_meta(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    max_len = max(t.shape[0] for t in seqs)
    batch = len(seqs)
    trailing = seqs[0].shape[1:]
    shape = (batch, max_len, *trailing) if batch_first else (max_len, batch, *trailing)
    return _meta_tensor(tuple(shape), seqs[0].dtype, seqs[0].device)


def _meta_block_diag_meta(*tensors):
    rows = sum(t.shape[0] for t in tensors)
    cols = sum(t.shape[1] for t in tensors)
    return _meta_tensor((rows, cols), tensors[0].dtype, tensors[0].device)


def _meta_diag_meta(a, diagonal=0):
    if len(a.shape) == 1:
        size = a.shape[0] + abs(diagonal)
        shape = (size, size)
    elif len(a.shape) == 2:
        m, n = a.shape
        if diagonal >= 0:
            length = max(0, min(m, n - diagonal))
        else:
            length = max(0, min(m + diagonal, n))
        shape = (length,)
    else:
        raise ValueError("diag expects 1D or 2D tensor")
    return _meta_tensor(shape, a.dtype, a.device)


def _meta_cartesian_prod_meta(*tensors):
    rows = 1
    for t in tensors:
        rows *= t.shape[0]
    cols = len(tensors)
    return _meta_tensor((rows, cols), tensors[0].dtype, tensors[0].device)





def _meta_chunk_meta(a, chunks, dim=0):
    dim_size = a.shape[dim]
    actual_chunks = min(chunks, dim_size) if dim_size > 0 else chunks
    if actual_chunks == 0:
        return tuple()
    chunk_size = (dim_size + actual_chunks - 1) // actual_chunks
    specs = []
    for i in range(actual_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, dim_size)
        if start >= end:
            break
        shape = list(a.shape)
        shape[dim] = end - start
        specs.append(_meta_tensor(tuple(shape), a.dtype, a.device))
    return tuple(specs)


def _meta_split_meta(a, split_size_or_sections, dim=0):
    dim_size = a.shape[dim]
    specs = []
    if isinstance(split_size_or_sections, int):
        step = split_size_or_sections
        for start in range(0, dim_size, step):
            end = min(start + step, dim_size)
            shape = list(a.shape)
            shape[dim] = end - start
            specs.append(_meta_tensor(tuple(shape), a.dtype, a.device))
    else:
        sizes = list(split_size_or_sections)
        if sum(sizes) != dim_size:
            raise ValueError("split sections must sum to dim size")
        for size in sizes:
            shape = list(a.shape)
            shape[dim] = size
            specs.append(_meta_tensor(tuple(shape), a.dtype, a.device))
    return tuple(specs)


def _meta_unbind_meta(a, dim=0):
    dim_size = a.shape[dim]
    specs = []
    for _ in range(dim_size):
        shape = list(a.shape)
        shape.pop(dim)
        specs.append(_meta_tensor(tuple(shape), a.dtype, a.device))
    return tuple(specs)


def _meta_argmax_meta(a, dim=None, keepdim=False):
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
    # PyTorch argmax returns int64
    return _meta_tensor(tuple(shape), int64_dtype, a.device)


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
    "_meta_lerp_meta",
    "_meta_addcmul_meta",
    "_meta_addcdiv_meta",
    "_meta_matmul_meta",
    "_meta_sum_meta",
    "_meta_reduce_bool_meta",
    "_meta_argmax_meta",
    "_meta_equal_meta",
    "_meta_cummax_meta",
    "_meta_argsort_meta",
    "_meta_sort_meta",
    "_meta_topk_meta",
    "_meta_stack_meta",
    "_meta_cat_meta",
    "_meta_hstack_meta",
    "_meta_vstack_meta",
    "_meta_column_stack_meta",
    "_meta_pad_sequence_meta",
    "_meta_block_diag_meta",
    "_meta_diag_meta",
    "_meta_diag_meta",
    "_meta_cartesian_prod_meta",
    "_meta_chunk_meta",
    "_meta_split_meta",
    "_meta_unbind_meta",
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

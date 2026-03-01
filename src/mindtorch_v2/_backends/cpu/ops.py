import math
import numpy as np

from ..._dtype import bool as bool_dtype
from ..._dtype import int64 as int64_dtype
from ..._dtype import to_numpy_dtype
from ..._storage import typed_storage_from_numpy
from ..._tensor import Tensor


def _to_numpy(t):
    return t._numpy_view()


def _normalize_tensor_sequence_args(tensors):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        return tuple(tensors[0])
    return tuple(tensors)


def _from_numpy(arr, dtype, device):
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def add(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(a_np + b_np, a.dtype, a.device)


def mul(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(a_np * b_np, a.dtype, a.device)


def div(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    out = np.true_divide(a_np, b_np)
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype), copy=False), a.dtype, a.device)


def true_divide(a, b):
    return div(a, b)


def matmul(a, b):
    return _from_numpy(_to_numpy(a) @ _to_numpy(b), a.dtype, a.device)


def relu(a):
    return _from_numpy(np.maximum(_to_numpy(a), 0), a.dtype, a.device)


def gelu(a):
    arr = _to_numpy(a)
    out = 0.5 * arr * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def sum_(a, dim=None, keepdim=False, dtype=None):
    if dtype is not None:
        raise NotImplementedError("sum dtype not supported yet")
    return _from_numpy(_to_numpy(a).sum(axis=dim, keepdims=keepdim), a.dtype, a.device)


def mean_(a, dim=None, keepdim=False):
    return _from_numpy(_to_numpy(a).mean(axis=dim, keepdims=keepdim), a.dtype, a.device)


def std_(a, dim=None, keepdim=False, unbiased=True):
    if not a.dtype.is_floating_point and not a.dtype.is_complex:
        raise RuntimeError("std and var only support floating point and complex dtypes")
    ddof = 1 if unbiased else 0
    out = np.std(_to_numpy(a), axis=dim, keepdims=keepdim, ddof=ddof)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def all_(a, dim=None, keepdim=False):
    return _from_numpy(np.all(_to_numpy(a), axis=dim, keepdims=keepdim), bool_dtype, a.device)


def any_(a, dim=None, keepdim=False):
    return _from_numpy(np.any(_to_numpy(a), axis=dim, keepdims=keepdim), bool_dtype, a.device)


def argmax(a, dim=None, keepdim=False):
    arr = _to_numpy(a)
    if dim is None:
        out = np.array(np.argmax(arr), dtype=np.int64)
    else:
        out = np.argmax(arr, axis=dim)
        if keepdim:
            out = np.expand_dims(out, axis=dim)
        out = out.astype(np.int64)
    return _from_numpy(out, int64_dtype, a.device)


def argmin(a, dim=None, keepdim=False):
    arr = _to_numpy(a)
    if dim is None:
        out = np.array(np.argmin(arr), dtype=np.int64)
    else:
        out = np.argmin(arr, axis=dim)
        if keepdim:
            out = np.expand_dims(out, axis=dim)
        out = out.astype(np.int64)
    return _from_numpy(out, int64_dtype, a.device)


def count_nonzero(a, dim=None, keepdim=False):
    arr = _to_numpy(a)
    if dim is None:
        count = np.count_nonzero(arr)
        if keepdim:
            out = np.array(count, dtype=np.int64).reshape((1,) * arr.ndim)
        else:
            out = np.array(count, dtype=np.int64)
    else:
        out = np.count_nonzero(arr, axis=dim, keepdims=keepdim).astype(np.int64)
    return _from_numpy(out, int64_dtype, a.device)


def masked_select(a, mask):
    arr = _to_numpy(a)
    mask_arr = _to_numpy(mask).astype(bool)
    out = arr[mask_arr]
    return _from_numpy(out, a.dtype, a.device)


def flip(a, dims):
    arr = _to_numpy(a)
    out = np.flip(arr, axis=dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def roll(a, shifts, dims=None):
    arr = _to_numpy(a)
    out = np.roll(arr, shift=shifts, axis=dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def rot90(a, k=1, dims=(0, 1)):
    arr = _to_numpy(a)
    out = np.rot90(arr, k=k, axes=dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def repeat(a, repeats):
    arr = _to_numpy(a)
    out = np.tile(arr, repeats)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def repeat_interleave(a, repeats, dim=None):
    arr = _to_numpy(a)
    axis = None if dim is None else dim
    out = np.repeat(arr, repeats, axis=axis)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def tile(a, dims):
    arr = _to_numpy(a)
    out = np.tile(arr, dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def nonzero(a, as_tuple=False):
    idx = np.nonzero(_to_numpy(a))
    if as_tuple:
        return tuple(
            _from_numpy(
                np.ascontiguousarray(dim_idx, dtype=np.int64), int64_dtype, a.device
            )
            for dim_idx in idx
        )
    stacked = np.stack(idx, axis=1).astype(np.int64, copy=False)
    return _from_numpy(np.ascontiguousarray(stacked), int64_dtype, a.device)


def cumsum(a, dim=0):
    return _from_numpy(np.cumsum(_to_numpy(a), axis=dim), a.dtype, a.device)


def cumprod(a, dim=0):
    return _from_numpy(np.cumprod(_to_numpy(a), axis=dim), a.dtype, a.device)


def cummax(a, dim=0):
    arr = _to_numpy(a)
    if dim < 0:
        dim += arr.ndim
    moved = np.moveaxis(arr, dim, 0)
    values = np.empty_like(moved)
    indices = np.empty(moved.shape, dtype=np.int64)
    max_vals = moved[0].copy()
    values[0] = max_vals
    indices[0] = 0
    for i in range(1, moved.shape[0]):
        mask = moved[i] > max_vals
        max_vals = np.where(mask, moved[i], max_vals)
        values[i] = max_vals
        indices[i] = np.where(mask, i, indices[i - 1])
    values = np.ascontiguousarray(np.moveaxis(values, 0, dim))
    indices = np.ascontiguousarray(np.moveaxis(indices, 0, dim))
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(indices, int64_dtype, a.device),
    )


def argsort(a, dim=-1, descending=False, stable=False):
    arr = _to_numpy(a)
    kind = "stable" if stable else "quicksort"
    if descending:
        idx = np.argsort(-arr, axis=dim, kind=kind)
    else:
        idx = np.argsort(arr, axis=dim, kind=kind)
    return _from_numpy(idx.astype(np.int64), int64_dtype, a.device)


def sort(a, dim=-1, descending=False, stable=False):
    arr = _to_numpy(a)
    kind = "stable" if stable else "quicksort"
    if descending:
        idx = np.argsort(-arr, axis=dim, kind=kind)
    else:
        idx = np.argsort(arr, axis=dim, kind=kind)
    values = np.take_along_axis(arr, idx, axis=dim)
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(idx.astype(np.int64), int64_dtype, a.device),
    )


def topk(a, k, dim=-1, largest=True, sorted=True):
    arr = _to_numpy(a)
    if largest:
        idx = np.argsort(-arr, axis=dim)
    else:
        idx = np.argsort(arr, axis=dim)
    idx = np.take(idx, np.arange(k), axis=dim)
    values = np.take_along_axis(arr, idx, axis=dim)
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(idx.astype(np.int64), int64_dtype, a.device),
    )


def stack(tensors, dim=0):
    arrays = [_to_numpy(t) for t in tensors]
    return _from_numpy(np.stack(arrays, axis=dim), tensors[0].dtype, tensors[0].device)


def cat(tensors, dim=0):
    arrays = [_to_numpy(t) for t in tensors]
    return _from_numpy(np.concatenate(arrays, axis=dim), tensors[0].dtype, tensors[0].device)


def concatenate(tensors, dim=0):
    return cat(tensors, dim=dim)


def hstack(tensors):
    if tensors[0].dim() == 1:
        return cat(tensors, dim=0)
    return cat(tensors, dim=1)


def vstack(tensors):
    if tensors[0].dim() == 1:
        expanded = [t.reshape((1, t.shape[0])) for t in tensors]
        return cat(expanded, dim=0)
    return cat(tensors, dim=0)


def row_stack(tensors):
    return vstack(tensors)


def dstack(tensors):
    arrays = [_to_numpy(t) for t in tensors]
    expanded = []
    for arr in arrays:
        if arr.ndim == 1:
            expanded.append(arr.reshape(1, arr.shape[0], 1))
        elif arr.ndim == 2:
            expanded.append(arr.reshape(arr.shape[0], arr.shape[1], 1))
        else:
            expanded.append(arr)
    out = np.concatenate(expanded, axis=2)
    return _from_numpy(out, tensors[0].dtype, tensors[0].device)


def column_stack(tensors):
    if tensors[0].dim() == 1:
        expanded = [t.reshape((t.shape[0], 1)) for t in tensors]
        return cat(expanded, dim=1)
    return cat(tensors, dim=1)


def _check_indices_layout(layout):
    if layout is None:
        return
    if isinstance(layout, str):
        if layout != "strided":
            raise ValueError("layout must be strided")
        return
    raise ValueError("layout must be strided")


def _indices_device(device):
    if device is None:
        return None
    if isinstance(device, str):
        return device
    return str(device)


def _ensure_integer_indices(arr, name):
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be integer dtype")
    return arr


def take(a, index):
    arr = _to_numpy(a).reshape(-1)
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    out = np.take(arr, idx)
    return _from_numpy(out, a.dtype, a.device)


def take_along_dim(a, indices, dim):
    arr = _to_numpy(a)
    idx = _ensure_integer_indices(_to_numpy(indices), "indices").astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    if idx.ndim != arr.ndim:
        raise ValueError("indices shape mismatch")
    for i, size in enumerate(idx.shape):
        if i != dim and size != arr.shape[i]:
            raise ValueError("indices shape mismatch")
    out = np.take_along_axis(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)


def index_select(a, dim, index):
    arr = _to_numpy(a)
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    if idx.ndim != 1:
        raise ValueError("index must be 1D")
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    out = np.take(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)


def _check_index_range(index, dim_size):
    if (index < 0).any() or (index >= dim_size).any():
        raise IndexError("index out of range")


def gather(a, dim, index):
    arr = _to_numpy(a)
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    if idx.ndim != arr.ndim:
        raise ValueError("index shape mismatch")
    for i, size in enumerate(idx.shape):
        if i != dim and size != arr.shape[i]:
            raise ValueError("index shape mismatch")
    _check_index_range(idx, arr.shape[dim])
    out = np.take_along_axis(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)


def scatter(a, dim, index, src):
    arr = _to_numpy(a).copy()
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    if idx.ndim != arr.ndim:
        raise ValueError("index shape mismatch")
    for i, size in enumerate(idx.shape):
        if i != dim and size != arr.shape[i]:
            raise ValueError("index shape mismatch")
    _check_index_range(idx, arr.shape[dim])
    if hasattr(src, "shape"):
        src_arr = _to_numpy(src)
    else:
        src_arr = np.array(src, dtype=arr.dtype)
    src_arr = np.broadcast_to(src_arr, idx.shape)
    np.put_along_axis(arr, idx, src_arr, axis=dim)
    return _from_numpy(arr, a.dtype, a.device)


def tril_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    _check_indices_layout(layout)
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype
    dev = _indices_device(device)
    r, c = np.tril_indices(row, k=offset, m=col)
    out = np.stack([r, c], axis=0).astype(to_numpy_dtype(dtype), copy=False)
    return _from_numpy(out, dtype, dev)


def triu_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    _check_indices_layout(layout)
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype
    dev = _indices_device(device)
    r, c = np.triu_indices(row, k=offset, m=col)
    out = np.stack([r, c], axis=0).astype(to_numpy_dtype(dtype), copy=False)
    return _from_numpy(out, dtype, dev)


def pad_sequence(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    arrays = [_to_numpy(t) for t in seqs]
    max_len = max(a.shape[0] for a in arrays)
    batch = len(arrays)
    trailing = arrays[0].shape[1:]
    out_shape = (batch, max_len, *trailing) if batch_first else (max_len, batch, *trailing)
    out = np.full(out_shape, padding_value, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        length = a.shape[0]
        start = max_len - length if padding_side == "left" else 0
        if batch_first:
            out[i, start:start + length, ...] = a
        else:
            out[start:start + length, i, ...] = a
    return _from_numpy(out, seqs[0].dtype, seqs[0].device)


def block_diag(*tensors):
    tensors = _normalize_tensor_sequence_args(tensors)
    arrays = [_to_numpy(t) for t in tensors]
    rows = sum(a.shape[0] for a in arrays)
    cols = sum(a.shape[1] for a in arrays)
    out = np.zeros((rows, cols), dtype=arrays[0].dtype)
    r = c = 0
    for a in arrays:
        out[r:r + a.shape[0], c:c + a.shape[1]] = a
        r += a.shape[0]
        c += a.shape[1]
    return _from_numpy(out, tensors[0].dtype, tensors[0].device)


def tril(a, diagonal=0):
    out = np.tril(_to_numpy(a), k=diagonal)
    return _from_numpy(out, a.dtype, a.device)


def triu(a, diagonal=0):
    out = np.triu(_to_numpy(a), k=diagonal)
    return _from_numpy(out, a.dtype, a.device)


def diag(a, diagonal=0):
    arr = _to_numpy(a)
    if arr.ndim not in (1, 2):
        raise ValueError("diag expects 1D or 2D tensor")
    out = np.diag(arr, k=diagonal).copy()
    return _from_numpy(out, a.dtype, a.device)


def cartesian_prod(*tensors):
    tensors = _normalize_tensor_sequence_args(tensors)
    arrays = [_to_numpy(t) for t in tensors]
    grids = np.meshgrid(*arrays, indexing="ij")
    stacked = np.stack([g.reshape(-1) for g in grids], axis=1)
    return _from_numpy(stacked, tensors[0].dtype, tensors[0].device)


def chunk(a, chunks, dim=0):
    arr = _to_numpy(a)
    if dim < 0:
        dim += arr.ndim
    dim_size = arr.shape[dim]
    if chunks <= 0:
        raise ValueError("chunks must be > 0")
    actual_chunks = min(chunks, dim_size) if dim_size > 0 else chunks
    if actual_chunks == 0:
        return tuple()
    chunk_size = (dim_size + actual_chunks - 1) // actual_chunks
    outputs = []
    for i in range(actual_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, dim_size)
        if start >= end:
            break
        slices = [slice(None)] * arr.ndim
        slices[dim] = slice(start, end)
        out = np.ascontiguousarray(arr[tuple(slices)])
        outputs.append(_from_numpy(out, a.dtype, a.device))
    return tuple(outputs)


def split(a, split_size_or_sections, dim=0):
    arr = _to_numpy(a)
    if dim < 0:
        dim += arr.ndim
    dim_size = arr.shape[dim]
    outputs = []
    if isinstance(split_size_or_sections, int):
        if split_size_or_sections <= 0:
            raise ValueError("split_size must be > 0")
        step = split_size_or_sections
        for start in range(0, dim_size, step):
            end = min(start + step, dim_size)
            slices = [slice(None)] * arr.ndim
            slices[dim] = slice(start, end)
            out = np.ascontiguousarray(arr[tuple(slices)])
            outputs.append(_from_numpy(out, a.dtype, a.device))
    else:
        sizes = list(split_size_or_sections)
        if sum(sizes) != dim_size:
            raise ValueError("split sections must sum to dim size")
        start = 0
        for size in sizes:
            end = start + size
            slices = [slice(None)] * arr.ndim
            slices[dim] = slice(start, end)
            out = np.ascontiguousarray(arr[tuple(slices)])
            outputs.append(_from_numpy(out, a.dtype, a.device))
            start = end
    return tuple(outputs)


def _split_sections_from_count(dim_size, sections):
    if sections <= 0:
        raise ValueError("sections must be > 0")
    size, extra = divmod(dim_size, sections)
    return [size + 1] * extra + [size] * (sections - extra)


def vsplit(a, split_size_or_sections):
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[0], split_size_or_sections)
        return split(a, sizes, dim=0)
    return split(a, split_size_or_sections, dim=0)


def hsplit(a, split_size_or_sections):
    dim = 0 if a.dim() == 1 else 1
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[dim], split_size_or_sections)
        return split(a, sizes, dim=dim)
    return split(a, split_size_or_sections, dim=dim)


def dsplit(a, split_size_or_sections):
    if a.dim() < 3:
        raise ValueError("dsplit expects input with at least 3 dimensions")
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[2], split_size_or_sections)
        return split(a, sizes, dim=2)
    return split(a, split_size_or_sections, dim=2)


def unbind(a, dim=0):
    arr = _to_numpy(a)
    if dim < 0:
        dim += arr.ndim
    dim_size = arr.shape[dim]
    outputs = []
    for i in range(dim_size):
        slices = [slice(None)] * arr.ndim
        slices[dim] = i
        out = np.ascontiguousarray(arr[tuple(slices)])
        outputs.append(_from_numpy(out, a.dtype, a.device))
    return tuple(outputs)


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return np.allclose(
        _to_numpy(a),
        _to_numpy(b),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    out = np.isclose(
        _to_numpy(a),
        _to_numpy(b),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    return _from_numpy(out, bool_dtype, a.device)


def equal(a, b):
    return np.array_equal(_to_numpy(a), _to_numpy(b))


def eq(a, b):
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.equal(_to_numpy(a), b_np), bool_dtype, a.device)


def ne(a, b):
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.not_equal(_to_numpy(a), b_np), bool_dtype, a.device)


def lt(a, b):
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.less(_to_numpy(a), b_np), bool_dtype, a.device)


def le(a, b):
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.less_equal(_to_numpy(a), b_np), bool_dtype, a.device)


def gt(a, b):
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.greater(_to_numpy(a), b_np), bool_dtype, a.device)


def ge(a, b):
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.greater_equal(_to_numpy(a), b_np), bool_dtype, a.device)


def add_(a, b):
    arr = _to_numpy(a)
    arr += _to_numpy(b) if isinstance(b, Tensor) else b
    return a


def mul_(a, b):
    arr = _to_numpy(a)
    arr *= _to_numpy(b) if isinstance(b, Tensor) else b
    return a


def relu_(a):
    arr = _to_numpy(a)
    np.maximum(arr, 0, out=arr)
    return a


def zero_(a):
    arr = _to_numpy(a)
    arr.fill(0)
    return a


def uniform_(a, low=0.0, high=1.0):
    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = _to_numpy(a)
    arr[:] = rng.uniform(low, high, arr.shape).astype(arr.dtype)
    return a


def normal_(a, mean=0.0, std=1.0):
    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = _to_numpy(a)
    arr[:] = rng.normal(mean, std, arr.shape).astype(arr.dtype)
    return a


def fill_(a, value):
    arr = _to_numpy(a)
    arr.fill(value)
    return a


def clamp_(a, min_val=None, max_val=None):
    arr = _to_numpy(a)
    np.clip(arr, min_val, max_val, out=arr)
    return a


def copy_(a, src):
    arr = _to_numpy(a)
    src_arr = _to_numpy(src)
    np.copyto(arr, src_arr)
    return a


def erfinv_(a):
    arr = _to_numpy(a)
    arr[:] = _ndtr_inv((arr + 1.0) / 2.0) / np.sqrt(2.0)
    return a


def _ndtr_inv(p):
    """Inverse normal CDF (probit function) using rational approximation.
    Used to compute erfinv: erfinv(x) = ndtr_inv((x+1)/2) / sqrt(2)."""
    p = np.asarray(p, dtype=np.float64)
    result = np.zeros_like(p)

    # Central region: |p - 0.5| <= 0.425
    q = p - 0.5
    mask_central = np.abs(q) <= 0.425
    if np.any(mask_central):
        r = q[mask_central]
        r2 = r * r
        # Rational approximation coefficients (Beasley-Springer-Moro)
        a = np.array([
            2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637
        ])
        b = np.array([
            -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833
        ])
        num = ((a[3] * r2 + a[2]) * r2 + a[1]) * r2 + a[0]
        den = (((b[3] * r2 + b[2]) * r2 + b[1]) * r2 + b[0]) * r2 + 1.0
        result[mask_central] = r * num / den

    # Tail regions
    mask_tail = ~mask_central & (p > 0) & (p < 1)
    if np.any(mask_tail):
        pp = np.where(p[mask_tail] < 0.5, p[mask_tail], 1.0 - p[mask_tail])
        r = np.sqrt(-2.0 * np.log(pp))
        c = np.array([
            2.515517, 0.802853, 0.010328
        ])
        d = np.array([
            1.432788, 0.189269, 0.001308
        ])
        num = (c[2] * r + c[1]) * r + c[0]
        den = ((d[2] * r + d[1]) * r + d[0]) * r + 1.0
        val = r - num / den
        val = np.where(p[mask_tail] < 0.5, -val, val)
        result[mask_tail] = val

    # Boundary cases
    result[p <= 0] = -np.inf
    result[p >= 1] = np.inf
    return result


def sub_(a, b):
    arr = _to_numpy(a)
    arr -= _to_numpy(b) if isinstance(b, Tensor) else b
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


def floor(a):
    return _from_numpy(np.floor(_to_numpy(a)), a.dtype, a.device)


def ceil(a):
    return _from_numpy(np.ceil(_to_numpy(a)), a.dtype, a.device)


def round(a):
    return _from_numpy(np.round(_to_numpy(a)), a.dtype, a.device)


def trunc(a):
    return _from_numpy(np.trunc(_to_numpy(a)), a.dtype, a.device)


def frac(a):
    arr = _to_numpy(a)
    out = arr - np.trunc(arr)
    return _from_numpy(out, a.dtype, a.device)


def pow(a, b):
    arr_a = _to_numpy(a)
    if isinstance(b, Tensor):
        arr_b = _to_numpy(b)
    else:
        arr_b = b
    return _from_numpy(np.power(arr_a, arr_b), a.dtype, a.device)


def log2(a):
    return _from_numpy(np.log2(_to_numpy(a)), a.dtype, a.device)


def log10(a):
    return _from_numpy(np.log10(_to_numpy(a)), a.dtype, a.device)


def exp2(a):
    return _from_numpy(np.exp2(_to_numpy(a)), a.dtype, a.device)


def rsqrt(a):
    arr = _to_numpy(a)
    out = 1.0 / np.sqrt(arr)
    return _from_numpy(out, a.dtype, a.device)


def sign(a):
    return _from_numpy(np.sign(_to_numpy(a)), a.dtype, a.device)


def signbit(a):
    arr = np.signbit(_to_numpy(a))
    return _from_numpy(arr, bool_dtype, a.device)


def isnan(a):
    arr = np.isnan(_to_numpy(a))
    return _from_numpy(arr, bool_dtype, a.device)


def isinf(a):
    arr = np.isinf(_to_numpy(a))
    return _from_numpy(arr, bool_dtype, a.device)


def isfinite(a):
    arr = np.isfinite(_to_numpy(a))
    return _from_numpy(arr, bool_dtype, a.device)


def sinh(a):
    return _from_numpy(np.sinh(_to_numpy(a)), a.dtype, a.device)


def cosh(a):
    return _from_numpy(np.cosh(_to_numpy(a)), a.dtype, a.device)


def asinh(a):
    return _from_numpy(np.arcsinh(_to_numpy(a)), a.dtype, a.device)


def acosh(a):
    return _from_numpy(np.arccosh(_to_numpy(a)), a.dtype, a.device)


def atanh(a):
    return _from_numpy(np.arctanh(_to_numpy(a)), a.dtype, a.device)


def erf(a):
    arr = _to_numpy(a)
    out = np.vectorize(math.erf)(arr)
    return _from_numpy(out, a.dtype, a.device)


def erfc(a):
    arr = _to_numpy(a)
    out = np.vectorize(math.erfc)(arr)
    return _from_numpy(out, a.dtype, a.device)


def softplus(a):
    arr = _to_numpy(a)
    out = np.log1p(np.exp(arr))
    return _from_numpy(out, a.dtype, a.device)


def silu(a):
    arr = _to_numpy(a)
    out = arr / (1.0 + np.exp(-arr))
    return _from_numpy(out, a.dtype, a.device)


def leaky_relu(a, negative_slope=0.01):
    arr = _to_numpy(a)
    out = np.where(arr > 0, arr, negative_slope * arr)
    return _from_numpy(out, a.dtype, a.device)


def elu(a, alpha=1.0):
    arr = _to_numpy(a)
    out = np.where(arr > 0, arr, alpha * (np.exp(arr) - 1))
    return _from_numpy(out, a.dtype, a.device)


def mish(a):
    arr = _to_numpy(a)
    out = arr * np.tanh(np.log1p(np.exp(arr)))
    return _from_numpy(out, a.dtype, a.device)


def prelu(a, weight):
    arr = _to_numpy(a)
    weight_arr = _to_numpy(weight)
    out = np.where(arr > 0, arr, arr * weight_arr)
    return _from_numpy(out, a.dtype, a.device)


def clamp(a, min_val=None, max_val=None):
    arr = _to_numpy(a)
    out = np.clip(arr, min_val, max_val)
    return _from_numpy(out, a.dtype, a.device)


def clamp_min(a, min_val):
    arr = _to_numpy(a)
    out = np.maximum(arr, min_val)
    return _from_numpy(out, a.dtype, a.device)


def clamp_max(a, max_val):
    arr = _to_numpy(a)
    out = np.minimum(arr, max_val)
    return _from_numpy(out, a.dtype, a.device)


def relu6(a):
    arr = _to_numpy(a)
    out = np.minimum(np.maximum(arr, 0.0), 6.0)
    return _from_numpy(out, a.dtype, a.device)


def hardtanh(a, min_val=-1.0, max_val=1.0):
    arr = _to_numpy(a)
    out = np.clip(arr, min_val, max_val)
    return _from_numpy(out, a.dtype, a.device)


def min_(a, b):
    return _from_numpy(np.minimum(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def max_(a, b):
    return _from_numpy(np.maximum(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def amin(a, dim=None, keepdim=False):
    arr = _to_numpy(a)
    out = np.amin(arr, axis=dim, keepdims=keepdim)
    return _from_numpy(out, a.dtype, a.device)


def amax(a, dim=None, keepdim=False):
    arr = _to_numpy(a)
    out = np.amax(arr, axis=dim, keepdims=keepdim)
    return _from_numpy(out, a.dtype, a.device)


def fmin(a, b):
    return _from_numpy(np.fmin(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def fmax(a, b):
    return _from_numpy(np.fmax(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def where(cond, x, y):
    cond_arr = _to_numpy(cond)
    x_arr = _to_numpy(x)
    if isinstance(y, Tensor):
        y_arr = _to_numpy(y)
    else:
        y_arr = y
    out = np.where(cond_arr, x_arr, y_arr)
    return _from_numpy(out, x.dtype, x.device)


def atan(a):
    return _from_numpy(np.arctan(_to_numpy(a)), a.dtype, a.device)


def atan2(a, b):
    return _from_numpy(np.arctan2(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def asin(a):
    return _from_numpy(np.arcsin(_to_numpy(a)), a.dtype, a.device)


def acos(a):
    return _from_numpy(np.arccos(_to_numpy(a)), a.dtype, a.device)


def lerp(a, b, weight):
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)
    if isinstance(weight, Tensor):
        w = _to_numpy(weight)
    else:
        w = weight
    out = arr_a + w * (arr_b - arr_a)
    return _from_numpy(out, a.dtype, a.device)


def addcmul(a, b, c, value=1.0):
    return _from_numpy(
        _to_numpy(a) + value * (_to_numpy(b) * _to_numpy(c)),
        a.dtype,
        a.device,
    )


def addcdiv(a, b, c, value=1.0):
    return _from_numpy(
        _to_numpy(a) + value * (_to_numpy(b) / _to_numpy(c)),
        a.dtype,
        a.device,
    )


def logaddexp(a, b):
    return _from_numpy(np.logaddexp(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def logaddexp2(a, b):
    return _from_numpy(np.logaddexp2(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def hypot(a, b):
    return _from_numpy(np.hypot(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def remainder(a, b):
    return _from_numpy(np.remainder(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def fmod(a, b):
    return _from_numpy(np.fmod(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def _normalize_index_key(key):
    if isinstance(key, Tensor):
        arr = _to_numpy(key)
        if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
            return arr
        raise IndexError("tensors used as indices must be integer or boolean")
    if isinstance(key, tuple):
        return tuple(_normalize_index_key(k) for k in key)
    if isinstance(key, list):
        return [_normalize_index_key(k) for k in key]
    return key


def _is_int_index(key):
    return isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_))


def _is_basic_index_key(key):
    keys = key if isinstance(key, tuple) else (key,)
    for item in keys:
        if item is Ellipsis or item is None:
            continue
        if isinstance(item, slice):
            continue
        if _is_int_index(item):
            continue
        return False
    return True


def _expand_ellipsis(keys, ndim):
    ellipsis_count = sum(1 for item in keys if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if ellipsis_count == 0:
        return keys

    specified_dims = sum(1 for item in keys if item is not None and item is not Ellipsis)
    fill = ndim - specified_dims
    if fill < 0:
        raise IndexError("too many indices for tensor")

    expanded = []
    for item in keys:
        if item is Ellipsis:
            expanded.extend([slice(None)] * fill)
        else:
            expanded.append(item)
    return expanded


def _basic_getitem_view(tensor, key):
    keys = list(key) if isinstance(key, tuple) else [key]
    keys = _expand_ellipsis(keys, tensor.dim())

    in_dim = 0
    out_shape = []
    out_stride = []
    out_offset = tensor.offset

    for item in keys:
        if item is None:
            out_shape.append(1)
            out_stride.append(0)
            continue

        if in_dim >= tensor.dim():
            raise IndexError("too many indices for tensor")

        dim_size = tensor.shape[in_dim]
        dim_stride = tensor.stride[in_dim]

        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += dim_size
            if idx < 0 or idx >= dim_size:
                raise IndexError("index out of range")
            out_offset += idx * dim_stride
            in_dim += 1
            continue

        if isinstance(item, slice):
            start, stop, step = item.indices(dim_size)
            out_offset += start * dim_stride
            out_shape.append(len(range(start, stop, step)))
            out_stride.append(dim_stride * step)
            in_dim += 1
            continue

        return None

    while in_dim < tensor.dim():
        out_shape.append(tensor.shape[in_dim])
        out_stride.append(tensor.stride[in_dim])
        in_dim += 1

    # Keep fallback copy path for negative strides until Tensor._numpy_view
    # supports them safely.
    if any(s < 0 for s in out_stride):
        return None

    return Tensor(tensor.storage(), tuple(out_shape), tuple(out_stride), out_offset)


def getitem(tensor, key):
    norm_key = _normalize_index_key(key)
    arr = _to_numpy(tensor)
    result = arr[norm_key]

    if _is_basic_index_key(norm_key):
        view = _basic_getitem_view(tensor, norm_key)
        if view is not None:
            return view

    if isinstance(result, np.generic) or (isinstance(result, np.ndarray) and result.ndim == 0):
        # Return 0-dim tensor (matches PyTorch behavior)
        scalar_arr = np.array(result)
        return _from_numpy(scalar_arr, tensor.dtype, tensor.device)
    return _from_numpy(np.ascontiguousarray(result), tensor.dtype, tensor.device)


def setitem(tensor, key, value):
    arr = _to_numpy(tensor)
    norm_key = _normalize_index_key(key)
    if hasattr(value, 'numpy'):
        arr[norm_key] = value.numpy()
    else:
        arr[norm_key] = value
    return tensor


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    arr = _to_numpy(input)
    ndim = len(arr.shape)

    if training:
        axes = (0,) + tuple(range(2, ndim))
        mean = arr.mean(axis=axes)
        var = arr.var(axis=axes)
        if running_mean is not None:
            rm = _to_numpy(running_mean)
            rm[:] = (1 - momentum) * rm + momentum * mean
        if running_var is not None:
            rv = _to_numpy(running_var)
            rv[:] = (1 - momentum) * rv + momentum * var
    else:
        mean = _to_numpy(running_mean)
        var = _to_numpy(running_var)

    shape = [1, -1] + [1] * (ndim - 2)
    normalized = (arr - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)

    if weight is not None:
        normalized = normalized * _to_numpy(weight).reshape(shape)
    if bias is not None:
        normalized = normalized + _to_numpy(bias).reshape(shape)

    return _from_numpy(normalized, input.dtype, input.device)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    arr = _to_numpy(input)
    N, C = arr.shape[:2]
    spatial = arr.shape[2:]

    grouped = arr.reshape(N, num_groups, C // num_groups, *spatial)
    axes = tuple(range(2, len(grouped.shape)))
    mean = grouped.mean(axis=axes, keepdims=True)
    var = grouped.var(axis=axes, keepdims=True)
    normalized = (grouped - mean) / np.sqrt(var + eps)
    normalized = normalized.reshape(arr.shape)

    if weight is not None:
        shape = [1, C] + [1] * len(spatial)
        normalized = normalized * _to_numpy(weight).reshape(shape)
    if bias is not None:
        shape = [1, C] + [1] * len(spatial)
        normalized = normalized + _to_numpy(bias).reshape(shape)

    return _from_numpy(normalized, input.dtype, input.device)


def dropout(a, p=0.5, training=True):
    if not training or p == 0:
        return a
    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = _to_numpy(a)
    mask = (rng.random(arr.shape) >= p).astype(arr.dtype)
    return _from_numpy(arr * mask / (1.0 - p), a.dtype, a.device)


def pad(a, pad_widths, mode='constant', value=0):
    arr = _to_numpy(a)
    ndim = len(arr.shape)

    if len(pad_widths) % 2 != 0:
        raise ValueError("Padding length must be divisible by 2")

    n_pairs = len(pad_widths) // 2
    if n_pairs > ndim:
        raise ValueError("Padding length too large for input dimensions")

    pads = [(0, 0)] * ndim
    # PyTorch pad format: (left, right, top, bottom, front, back, ...)
    # Applied to last dims first.
    for i in range(n_pairs):
        dim = ndim - 1 - i
        pads[dim] = (int(pad_widths[2 * i]), int(pad_widths[2 * i + 1]))

    # Negative padding crops first, then positive padding extends.
    slices = [slice(None)] * ndim
    for dim, (left, right) in enumerate(pads):
        start = max(-left, 0)
        end = arr.shape[dim] - max(-right, 0)
        length = end - start
        if length < 0:
            raise RuntimeError("narrow(): length must be non-negative.")
        slices[dim] = slice(start, end)
    result = arr[tuple(slices)]

    np_pad = [(max(left, 0), max(right, 0)) for left, right in pads]
    if any(left or right for left, right in np_pad):
        if mode == 'constant':
            result = np.pad(result, np_pad, mode='constant', constant_values=value)
        elif mode == 'reflect':
            result = np.pad(result, np_pad, mode='reflect')
        elif mode == 'replicate':
            result = np.pad(result, np_pad, mode='edge')
        elif mode == 'circular':
            result = np.pad(result, np_pad, mode='wrap')
        else:
            raise ValueError(f"Unsupported pad mode: {mode}")
    else:
        result = np.ascontiguousarray(result)

    return _from_numpy(result, a.dtype, a.device)


def softmax(a, dim):
    arr = _to_numpy(a)
    exp_arr = np.exp(arr - np.max(arr, axis=dim, keepdims=True))
    result = exp_arr / np.sum(exp_arr, axis=dim, keepdims=True)
    return _from_numpy(result, a.dtype, a.device)


def log_softmax(a, dim):
    arr = _to_numpy(a)
    max_arr = np.max(arr, axis=dim, keepdims=True)
    exp_arr = np.exp(arr - max_arr)
    log_sum_exp = np.log(np.sum(exp_arr, axis=dim, keepdims=True))
    result = arr - max_arr - log_sum_exp
    return _from_numpy(result, a.dtype, a.device)


def one_hot(a, num_classes=-1):
    arr = _to_numpy(a)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("one_hot is only applicable to index tensor")
    flat = arr.astype(np.int64, copy=False).reshape(-1)
    if num_classes is None or int(num_classes) < 0:
        num_classes = int(flat.max()) + 1 if flat.size > 0 else 0
    num_classes = int(num_classes)
    if (flat < 0).any():
        raise ValueError("one_hot indices must be non-negative")
    if (flat >= num_classes).any() and flat.size > 0:
        raise ValueError("one_hot indices out of range")
    out = np.eye(num_classes, dtype=np.int64)[flat]
    out = out.reshape(arr.shape + (num_classes,))
    return _from_numpy(np.ascontiguousarray(out), int64_dtype, a.device)


def embedding(weight, indices, padding_idx=None, scale_grad_by_freq=False, sparse=False):
    weight_arr = _to_numpy(weight)
    idx = _ensure_integer_indices(_to_numpy(indices), "indices").astype(np.int64, copy=False)
    if idx.size and (idx.min() < 0 or idx.max() >= weight_arr.shape[0]):
        raise IndexError("index out of range in self")
    out = weight_arr[idx]
    return _from_numpy(np.ascontiguousarray(out), weight.dtype, weight.device)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    arr = _to_numpy(input)
    norm_shape = tuple(normalized_shape)
    if len(norm_shape) == 0:
        raise ValueError("normalized_shape must be non-empty")
    if tuple(arr.shape[-len(norm_shape):]) != norm_shape:
        raise ValueError("normalized_shape mismatch")

    axis = tuple(range(arr.ndim - len(norm_shape), arr.ndim))
    mean = arr.mean(axis=axis, keepdims=True)
    var = arr.var(axis=axis, keepdims=True)
    out = (arr - mean) / np.sqrt(var + eps)

    if weight is not None:
        out = out * _to_numpy(weight).reshape((1,) * (arr.ndim - len(norm_shape)) + norm_shape)
    if bias is not None:
        out = out + _to_numpy(bias).reshape((1,) * (arr.ndim - len(norm_shape)) + norm_shape)

    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)

import math
import ctypes
import struct
import numpy as np

from ..._dtype import bool as bool_dtype
from ..._dtype import int64 as int64_dtype
from ..._dtype import float16 as float16_dtype
from ..._dtype import float32 as float32_dtype
from ..._dtype import float64 as float64_dtype
from ..._dtype import to_numpy_dtype
from ..._storage import mps_typed_storage_from_numpy, _MPSUntypedStorage, TypedStorage
from ..._tensor import Tensor
from . import accelerate as _accel

# ---------------------------------------------------------------------------
# GPU dispatch helpers
# ---------------------------------------------------------------------------
_GPU_FLOAT_DTYPES = frozenset({float32_dtype, float16_dtype})


def _can_use_gpu(t):
    """Check if tensor can use Metal GPU kernels."""
    return (t.dtype in _GPU_FLOAT_DTYPES
            and t.is_contiguous()
            and t.numel() > 0
            and hasattr(t._storage, '_untyped')
            and hasattr(t._storage._untyped, '_metal_buffer')
            and t._storage._untyped._metal_buffer is not None)


def _metal_buf(t):
    """Get the raw Metal buffer from a tensor."""
    return t._storage._untyped._metal_buffer


def _kernel_suffix(dtype):
    """Return MSL kernel suffix for dtype."""
    return "f16" if dtype == float16_dtype else "f32"


def _scalar_fmt(dtype):
    """Return struct format char for scalar encoding."""
    return "e" if dtype == float16_dtype else "f"


def _itemsize(dtype):
    """Return byte size per element."""
    return 2 if dtype == float16_dtype else 4


def _alloc_output_buf(numel, dtype):
    """Allocate a Metal buffer for output."""
    from .runtime import get_runtime
    rt = get_runtime()
    return rt.create_buffer(numel * _itemsize(dtype))


def _from_metal_buffer(metal_buf, shape, stride, dtype, device):
    """Wrap an existing Metal buffer into a Tensor without copying data."""
    from .runtime import buffer_contents
    nbytes = 1
    for s in shape:
        nbytes *= s
    nbytes *= _itemsize(dtype)
    nbytes = max(nbytes, 1)
    untyped = _MPSUntypedStorage(metal_buf, nbytes, device=device)
    ptr = buffer_contents(metal_buf)
    numel = 1
    for s in shape:
        numel *= s
    data = np.frombuffer(
        (ctypes.c_uint8 * nbytes).from_address(ptr),
        dtype=to_numpy_dtype(dtype),
        count=max(numel, 1),
    )
    size = max(numel, 1)
    storage = TypedStorage(untyped, dtype, size, data=data)
    return Tensor(storage, shape, stride)


def _get_dispatcher():
    """Lazy import of the Metal kernel dispatcher singleton."""
    from .metal_compute import get_dispatcher
    return get_dispatcher()


def _to_numpy(t):
    return t._numpy_view()


def _normalize_tensor_sequence_args(tensors):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        return tuple(tensors[0])
    return tuple(tensors)


def _from_numpy(arr, dtype, device):
    storage = mps_typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def add(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"add_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"add_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(a_np + b_np, a.dtype, a.device)


def mul(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"mul_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"mul_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(a_np * b_np, a.dtype, a.device)


def div(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"div_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"div_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    out = np.true_divide(a_np, b_np)
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype), copy=False), a.dtype, a.device)


def true_divide(a, b):
    return div(a, b)


def _can_use_blas(arr):
    """Check if array is contiguous float32 or float64 for BLAS."""
    return (arr.flags['C_CONTIGUOUS'] and
            arr.dtype in (np.float32, np.float64) and
            _accel.available())


def _blas_gemm(a_np, b_np, dtype):
    """Matrix-matrix multiply via Accelerate BLAS."""
    M, K = a_np.shape
    K2, N = b_np.shape
    out = np.empty((M, N), dtype=a_np.dtype)
    if a_np.dtype == np.float32:
        _accel.cblas_sgemm(111, 111, M, N, K, 1.0,
                           a_np.ctypes.data, K, b_np.ctypes.data, N,
                           0.0, out.ctypes.data, N)
    else:
        _accel.cblas_dgemm(111, 111, M, N, K, 1.0,
                           a_np.ctypes.data, K, b_np.ctypes.data, N,
                           0.0, out.ctypes.data, N)
    return out


def matmul(a, b):
    # GPU path: MPSMatrixMultiplication for 2D contiguous float32/float16
    if (_can_use_gpu(a) and _can_use_gpu(b)
            and len(a.shape) == 2 and len(b.shape) == 2):
        from .mps_kernels import mps_matmul_gpu, _mps_dtype_code
        np_dt = to_numpy_dtype(a.dtype)
        dtype_code = _mps_dtype_code(np_dt)
        if dtype_code is not None:
            M, K = a.shape
            K2, N = b.shape
            if K == K2:
                out_buf = mps_matmul_gpu(
                    _metal_buf(a), _metal_buf(b),
                    M, K, N, dtype_code, _itemsize(a.dtype))
                if out_buf is not None:
                    from ..._tensor import _compute_strides
                    out_shape = (M, N)
                    out_stride = _compute_strides(out_shape)
                    return _from_metal_buffer(out_buf, out_shape, out_stride,
                                             a.dtype, a.device)
    # CPU path (Accelerate BLAS or numpy)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    if a_np.ndim == 2 and b_np.ndim == 2:
        a_c = np.ascontiguousarray(a_np)
        b_c = np.ascontiguousarray(b_np)
        if _can_use_blas(a_c) and _can_use_blas(b_c):
            return _from_numpy(_blas_gemm(a_c, b_c, a.dtype), a.dtype, a.device)
    if a_np.ndim == 3 and b_np.ndim == 3:
        a_c = np.ascontiguousarray(a_np)
        b_c = np.ascontiguousarray(b_np)
        if a_c.dtype in (np.float32, np.float64) and _accel.available():
            batch = a_c.shape[0]
            M, K = a_c.shape[1], a_c.shape[2]
            N = b_c.shape[2]
            out = np.empty((batch, M, N), dtype=a_c.dtype)
            for i in range(batch):
                out[i] = _blas_gemm(a_c[i], b_c[i], a.dtype)
            return _from_numpy(out, a.dtype, a.device)
    return _from_numpy(a_np @ b_np, a.dtype, a.device)


def mm(a, b):
    return matmul(a, b)


def bmm(a, b):
    return matmul(a, b)


def relu(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"relu_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.maximum(_to_numpy(a), 0), a.dtype, a.device)


def gelu(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"gelu_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    arr = _to_numpy(a)
    out = 0.5 * arr * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def sum_(a, dim=None, keepdim=False, dtype=None):
    if dtype is not None:
        raise NotImplementedError("sum dtype not supported yet")
    if isinstance(dim, list):
        dim = tuple(dim)
    if isinstance(dim, tuple) and len(dim) == 0:
        dim = None

    ndim = len(a.shape)

    def _check_dim_range(d):
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )

    if isinstance(dim, int):
        _check_dim_range(dim)
    elif isinstance(dim, tuple):
        for d in dim:
            _check_dim_range(d)

    # GPU path: full-tensor reduction (dim=None)
    if dim is None and not keepdim and _can_use_gpu(a) and a.dtype == float32_dtype:
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(1, a.dtype)
        d.dispatch_reduction("sum_partial_f32", "sum_final_f32",
                             _metal_buf(a), out_buf, a.numel())
        from ..._tensor import _compute_strides
        return _from_metal_buffer(out_buf, (), (), a.dtype, a.device)

    return _from_numpy(_to_numpy(a).sum(axis=dim, keepdims=keepdim), a.dtype, a.device)


def mean_(a, dim=None, keepdim=False):
    # GPU path: full-tensor mean (dim=None)
    if dim is None and not keepdim and _can_use_gpu(a) and a.dtype == float32_dtype:
        d = _get_dispatcher()
        # mean = sum / N
        sum_buf = _alloc_output_buf(1, a.dtype)
        d.dispatch_reduction("sum_partial_f32", "sum_final_f32",
                             _metal_buf(a), sum_buf, a.numel())
        out_buf = _alloc_output_buf(1, a.dtype)
        n = float(a.numel())
        d.dispatch_binary_scalar("div_scalar_f32", sum_buf, n, out_buf, 1)
        return _from_metal_buffer(out_buf, (), (), a.dtype, a.device)
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
    # GPU path: full-tensor argmax (dim=None)
    if dim is None and _can_use_gpu(a) and a.dtype == float32_dtype:
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(1, int64_dtype)  # uint output
        d.dispatch_arg_reduction("argmax_partial_f32", "argmax_final_f32",
                                 _metal_buf(a), out_buf, a.numel())
        # Read uint32 result, convert to int64
        from .runtime import buffer_contents
        ptr = buffer_contents(out_buf)
        idx_val = struct.unpack("I", (ctypes.c_char * 4).from_address(ptr))[0]
        out = np.array(int(idx_val), dtype=np.int64)
        return _from_numpy(out, int64_dtype, a.device)
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
    # GPU path: full-tensor argmin (dim=None)
    if dim is None and _can_use_gpu(a) and a.dtype == float32_dtype:
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(1, int64_dtype)
        d.dispatch_arg_reduction("argmin_partial_f32", "argmin_final_f32",
                                 _metal_buf(a), out_buf, a.numel())
        from .runtime import buffer_contents
        ptr = buffer_contents(out_buf)
        idx_val = struct.unpack("I", (ctypes.c_char * 4).from_address(ptr))[0]
        out = np.array(int(idx_val), dtype=np.int64)
        return _from_numpy(out, int64_dtype, a.device)
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
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"add_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"add_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr += _to_numpy(b) if isinstance(b, Tensor) else b
    return a


def mul_(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"mul_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"mul_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr *= _to_numpy(b) if isinstance(b, Tensor) else b
    return a


def relu_(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_inplace_unary(f"relu_inplace_{sfx}", _metal_buf(a), a.numel())
        return a
    arr = _to_numpy(a)
    np.maximum(arr, 0, out=arr)
    return a


def zero_(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_fill(f"fill_{sfx}", _metal_buf(a), 0.0, a.numel(),
                        scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr.fill(0)
    return a


def uniform_(a, low=0.0, high=1.0, generator=None):
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[:] = rng.uniform(low, high, arr.shape).astype(arr.dtype)
    return a


def normal_(a, mean=0.0, std=1.0, generator=None):
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[:] = rng.normal(mean, std, arr.shape).astype(arr.dtype)
    return a


def bernoulli_(a, p=0.5, generator=None):
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    if hasattr(p, '_numpy_view'):
        probs = p._numpy_view()
    elif hasattr(p, 'numpy'):
        probs = p.numpy()
    else:
        probs = float(p)
    arr[...] = (rng.uniform(0, 1, arr.shape) < probs).astype(arr.dtype)
    return a


def exponential_(a, lambd=1.0, generator=None):
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.exponential(1.0 / lambd, arr.shape).astype(arr.dtype)
    return a


def log_normal_(a, mean=1.0, std=2.0, generator=None):
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.lognormal(mean, std, arr.shape).astype(arr.dtype)
    return a


def cauchy_(a, median=0.0, sigma=1.0, generator=None):
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = (median + sigma * np.tan(np.pi * (rng.uniform(0, 1, arr.shape) - 0.5))).astype(arr.dtype)
    return a


def geometric_(a, p, generator=None):
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.geometric(p, arr.shape).astype(arr.dtype)
    return a


def fill_(a, value):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_fill(f"fill_{sfx}", _metal_buf(a), float(value), a.numel(),
                        scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr.fill(value)
    return a


def clamp_(a, min_val=None, max_val=None):
    arr = _to_numpy(a)
    np.clip(arr, min_val, max_val, out=arr)
    return a


def copy_(a, src):
    if _can_use_gpu(a) and _can_use_gpu(src):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = min(a.numel(), src.numel())
        d.dispatch_copy(f"copy_{sfx}", _metal_buf(src), _metal_buf(a), numel)
        return a
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
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"sub_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"sub_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr -= _to_numpy(b) if isinstance(b, Tensor) else b
    return a


def div_(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"div_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"div_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    arr /= b_np
    return a


def contiguous(a):
    if a.device.type != "mps":
        raise ValueError("MPS contiguous expects MPS tensors")
    arr = np.ascontiguousarray(_to_numpy(a))
    return _from_numpy(arr, a.dtype, a.device)


def abs(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"abs_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.abs(_to_numpy(a)), a.dtype, a.device)


def neg(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"neg_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.negative(_to_numpy(a)), a.dtype, a.device)


def exp(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"exp_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.exp(_to_numpy(a)), a.dtype, a.device)


def log(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"log_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.log(_to_numpy(a)), a.dtype, a.device)


def sqrt(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"sqrt_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.sqrt(_to_numpy(a)), a.dtype, a.device)


def sin(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"sin_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.sin(_to_numpy(a)), a.dtype, a.device)


def cos(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"cos_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.cos(_to_numpy(a)), a.dtype, a.device)


def tan(a):
    return _from_numpy(np.tan(_to_numpy(a)), a.dtype, a.device)


def tanh(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"tanh_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.tanh(_to_numpy(a)), a.dtype, a.device)


def sigmoid(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"sigmoid_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    arr = _to_numpy(a)
    out = 1.0 / (1.0 + np.exp(-arr))
    return _from_numpy(out, a.dtype, a.device)


def floor(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"floor_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.floor(_to_numpy(a)), a.dtype, a.device)


def ceil(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"ceil_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.ceil(_to_numpy(a)), a.dtype, a.device)


def round(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"round_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.round(_to_numpy(a)), a.dtype, a.device)


def trunc(a):
    return _from_numpy(np.trunc(_to_numpy(a)), a.dtype, a.device)


def frac(a):
    arr = _to_numpy(a)
    out = arr - np.trunc(arr)
    return _from_numpy(out, a.dtype, a.device)


def pow(a, b):
    if isinstance(a, Tensor) and _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"pow_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"pow_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    if isinstance(a, Tensor):
        arr_a = _to_numpy(a)
        ref = a
    else:
        arr_a = a
        ref = b
    if isinstance(b, Tensor):
        arr_b = _to_numpy(b)
    else:
        arr_b = b
    return _from_numpy(np.power(arr_a, arr_b), ref.dtype, ref.device)


def log2(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"log2_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.log2(_to_numpy(a)), a.dtype, a.device)


def log10(a):
    return _from_numpy(np.log10(_to_numpy(a)), a.dtype, a.device)


def exp2(a):
    return _from_numpy(np.exp2(_to_numpy(a)), a.dtype, a.device)


def rsqrt(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"rsqrt_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    arr = _to_numpy(a)
    out = 1.0 / np.sqrt(arr)
    return _from_numpy(out, a.dtype, a.device)


def sign(a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"sign_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.sign(_to_numpy(a)), a.dtype, a.device)


def square(a):
    arr = _to_numpy(a)
    out = np.square(arr)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


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
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_unary(f"silu_{sfx}", _metal_buf(a), out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
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


def selu(a):
    ALPHA = 1.6732632423543772
    SCALE = 1.0507009873554805
    arr = _to_numpy(a)
    out = SCALE * np.where(arr > 0, arr, ALPHA * (np.exp(arr) - 1))
    return _from_numpy(out, a.dtype, a.device)


def celu(a, alpha=1.0):
    arr = _to_numpy(a)
    out = np.maximum(arr, 0.0) + np.minimum(0.0, alpha * (np.exp(arr / alpha) - 1))
    return _from_numpy(out, a.dtype, a.device)


def threshold(a, threshold_val, value):
    arr = _to_numpy(a)
    out = np.where(arr > threshold_val, arr, value)
    return _from_numpy(out, a.dtype, a.device)


def hardshrink(a, lambd=0.5):
    arr = _to_numpy(a)
    out = np.where(np.abs(arr) > lambd, arr, 0.0)
    return _from_numpy(out, a.dtype, a.device)


def softshrink(a, lambd=0.5):
    arr = _to_numpy(a)
    out = np.sign(arr) * np.maximum(np.abs(arr) - lambd, 0.0)
    return _from_numpy(out, a.dtype, a.device)


def rrelu(a, lower=1.0 / 8, upper=1.0 / 3, training=False):
    arr = _to_numpy(a)
    if training:
        slope = np.random.uniform(lower, upper, size=arr.shape).astype(arr.dtype)
    else:
        slope = np.full_like(arr, (lower + upper) / 2.0)
    out = np.where(arr >= 0, arr, arr * slope)
    result = _from_numpy(out, a.dtype, a.device)
    result._rrelu_slope = _from_numpy(slope, a.dtype, a.device)
    return result


def min_(a, b):
    return _from_numpy(np.minimum(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def max_(a, b):
    return _from_numpy(np.maximum(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def hardswish(a):
    arr = _to_numpy(a).astype(np.float64)
    out = arr * np.clip(arr + 3.0, 0.0, 6.0) / 6.0
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)


def hardsigmoid(a):
    arr = _to_numpy(a).astype(np.float64)
    out = np.clip(arr + 3.0, 0.0, 6.0) / 6.0
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)


def softsign(a):
    arr = _to_numpy(a).astype(np.float64)
    out = arr / (1.0 + np.abs(arr))
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)


def normalize(a, p=2.0, dim=1, eps=1e-12):
    arr = _to_numpy(a).astype(np.float64)
    norm = np.linalg.norm(arr, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)
    out = arr / norm
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)


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
    if isinstance(a, Tensor) and _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"remainder_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"remainder_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    a_np = _to_numpy(a) if isinstance(a, Tensor) else a
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    ref = a if isinstance(a, Tensor) else b
    return _from_numpy(np.remainder(a_np, b_np), ref.dtype, ref.device)


def fmod(a, b):
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_binary(f"fmod_{sfx}", _metal_buf(a), _metal_buf(b),
                          out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
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
    # GPU path: softmax over last dim for 2D contiguous float32
    ndim = len(a.shape)
    actual_dim = dim if dim >= 0 else dim + ndim
    if (_can_use_gpu(a) and a.dtype == float32_dtype
            and actual_dim == ndim - 1 and ndim >= 1):
        d = _get_dispatcher()
        numel = a.numel()
        cols = a.shape[-1]
        rows = numel // cols
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_softmax_2d("softmax_f32", _metal_buf(a), out_buf,
                              rows, cols)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
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
    # Check logical dtype — may be a string or DType object
    dtype_str = str(a.dtype).replace('torch.', '')
    if dtype_str not in ('int8', 'int16', 'int32', 'int64', 'uint8', 'bool'):
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


def linalg_qr(a, mode='reduced'):
    """QR decomposition on CPU via numpy."""
    arr = _to_numpy(a)
    np_mode = mode
    q, r = np.linalg.qr(arr, mode=np_mode)
    return _from_numpy(q, a.dtype, a.device), _from_numpy(r, a.dtype, a.device)



# ---------------------------------------------------------------------------
# Tensor indexing / selection ops
# ---------------------------------------------------------------------------

def narrow(a, dim, start, length):
    from ..._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    return Tensor(a.storage(), tuple(new_shape), a.stride, new_offset)


def select(a, dim, index):
    from ..._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    idx = int(index)
    if idx < 0:
        idx += a.shape[d]
    new_shape = list(a.shape)
    del new_shape[d]
    new_stride = list(a.stride)
    new_offset = a.offset + idx * a.stride[d]
    del new_stride[d]
    return Tensor(a.storage(), tuple(new_shape), tuple(new_stride), new_offset)


def expand(a, sizes):
    from ..._tensor import Tensor
    sizes = tuple(sizes)
    ndiff = len(sizes) - a.dim()
    if ndiff < 0:
        raise RuntimeError("expand: number of sizes must be >= tensor dim")
    src_shape = (1,) * ndiff + a.shape
    src_stride = (0,) * ndiff + a.stride
    out_shape = []
    out_stride = []
    for i, sz in enumerate(sizes):
        if sz == -1:
            out_shape.append(src_shape[i])
            out_stride.append(src_stride[i])
        elif src_shape[i] == 1:
            out_shape.append(sz)
            out_stride.append(0)
        elif src_shape[i] == sz:
            out_shape.append(sz)
            out_stride.append(src_stride[i])
        else:
            raise RuntimeError(
                f"expand: size {sz} not compatible with dim size {src_shape[i]}"
            )
    return Tensor(a.storage(), tuple(out_shape), tuple(out_stride), a.offset)


def masked_fill(a, mask, value):
    arr = _to_numpy(a).copy()
    m = _to_numpy(mask).astype(bool)
    arr[m] = value
    return _from_numpy(arr, a.dtype, a.device)


def masked_fill_(a, mask, value):
    arr = _to_numpy(a)
    m = _to_numpy(mask).astype(bool)
    arr[m] = value
    return a


def index_put_(a, indices, values, accumulate=False):
    arr = _to_numpy(a)
    idx = tuple(_to_numpy(t) if hasattr(t, '_numpy_view') else t for t in indices)
    vals = _to_numpy(values) if hasattr(values, '_numpy_view') else values
    if accumulate:
        np.add.at(arr, idx, vals)
    else:
        arr[idx] = vals
    return a


def index_put(a, indices, values, accumulate=False):
    arr = _to_numpy(a).copy()
    idx = tuple(_to_numpy(t) if hasattr(t, '_numpy_view') else t for t in indices)
    vals = _to_numpy(values) if hasattr(values, '_numpy_view') else values
    if accumulate:
        np.add.at(arr, idx, vals)
    else:
        arr[idx] = vals
    return _from_numpy(arr, a.dtype, a.device)


def index_copy_(a, dim, index, source):
    arr = _to_numpy(a)
    idx = _to_numpy(index).ravel().astype(np.intp)
    src = _to_numpy(source)
    d = dim if dim >= 0 else dim + a.dim()
    for j, i in enumerate(idx):
        slices_dst = [slice(None)] * arr.ndim
        slices_dst[d] = int(i)
        slices_src = [slice(None)] * arr.ndim
        slices_src[d] = j
        arr[tuple(slices_dst)] = src[tuple(slices_src)]
    return a


def index_fill_(a, dim, index, value):
    arr = _to_numpy(a)
    idx = _to_numpy(index).ravel().astype(np.intp)
    d = dim if dim >= 0 else dim + a.dim()
    for i in idx:
        slices = [slice(None)] * arr.ndim
        slices[d] = int(i)
        arr[tuple(slices)] = value
    return a


def index_add_(a, dim, index, source, alpha=1.0):
    arr = _to_numpy(a)
    idx = _to_numpy(index).ravel().astype(np.intp)
    src = _to_numpy(source)
    d = dim if dim >= 0 else dim + a.dim()
    for j, i in enumerate(idx):
        slices_dst = [slice(None)] * arr.ndim
        slices_dst[d] = int(i)
        slices_src = [slice(None)] * arr.ndim
        slices_src[d] = j
        arr[tuple(slices_dst)] += float(alpha) * src[tuple(slices_src)]
    return a


def scatter_(a, dim, index, src):
    arr = _to_numpy(a)
    idx = _to_numpy(index).astype(np.intp)
    d = dim if dim >= 0 else dim + a.dim()
    if hasattr(src, '_numpy_view'):
        src_arr = _to_numpy(src)
    else:
        src_arr = src
    it = np.nditer(idx, flags=['multi_index'])
    while not it.finished:
        mi = it.multi_index
        dst_idx = list(mi)
        dst_idx[d] = int(it[0])
        if hasattr(src, '_numpy_view'):
            arr[tuple(dst_idx)] = src_arr[mi]
        else:
            arr[tuple(dst_idx)] = src_arr
        it.iternext()
    return a


def scatter_add_(a, dim, index, src):
    arr = _to_numpy(a)
    idx = _to_numpy(index).astype(np.intp)
    src_arr = _to_numpy(src)
    d = dim if dim >= 0 else dim + a.dim()
    it = np.nditer(idx, flags=['multi_index'])
    while not it.finished:
        mi = it.multi_index
        dst_idx = list(mi)
        dst_idx[d] = int(it[0])
        arr[tuple(dst_idx)] += src_arr[mi]
        it.iternext()
    return a


def masked_scatter_(a, mask, source):
    arr = _to_numpy(a)
    m = _to_numpy(mask).astype(bool)
    src = _to_numpy(source)
    arr[m] = src.ravel()[:m.sum()]
    return a


def unfold(a, dimension, size, step):
    arr = _to_numpy(a)
    d = dimension if dimension >= 0 else dimension + arr.ndim
    dim_size = arr.shape[d]
    n_windows = max(0, (dim_size - size) // step + 1)
    if n_windows == 0:
        new_shape = list(arr.shape)
        new_shape[d] = 0
        new_shape.append(size)
        return _from_numpy(np.empty(new_shape, dtype=arr.dtype), a.dtype, a.device)
    out_shape = list(arr.shape)
    out_shape[d] = n_windows
    out_shape.append(size)
    out = np.empty(out_shape, dtype=arr.dtype)
    for i in range(n_windows):
        src_s = [slice(None)] * arr.ndim
        src_s[d] = slice(i * step, i * step + size)
        chunk = np.moveaxis(arr[tuple(src_s)], d, -1)
        dst_s = [slice(None)] * (arr.ndim + 1)
        dst_s[d] = i
        out[tuple(dst_s)] = chunk
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def var_(a, dim=None, unbiased=True, keepdim=False):
    arr = _to_numpy(a)
    ddof = 1 if unbiased else 0
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            dim = tuple(dim)
        out = np.var(arr, axis=dim, keepdims=keepdim, ddof=ddof)
    else:
        out = np.var(arr, ddof=ddof)
        if keepdim:
            out = np.full([1] * arr.ndim, out)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(arr.dtype, copy=False)), a.dtype, a.device)


def norm_(a, p=2, dim=None, keepdim=False):
    arr = _to_numpy(a).astype(np.float64)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            dim = tuple(dim)
        out = np.linalg.norm(arr, ord=p, axis=dim, keepdims=keepdim)
    else:
        out = np.linalg.norm(arr.ravel(), ord=p)
        if keepdim:
            out = np.full([1] * arr.ndim, out)
    from ..._dtype import float32 as f32
    out_dtype = a.dtype if a.dtype.is_floating_point else f32
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(out_dtype), copy=False)), out_dtype, a.device)


def prod_(a, dim=None, keepdim=False):
    arr = _to_numpy(a)
    if dim is not None:
        out = np.prod(arr, axis=dim, keepdims=keepdim)
    else:
        out = np.prod(arr)
        if keepdim:
            out = np.full([1] * arr.ndim, out)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(arr.dtype, copy=False)), a.dtype, a.device)


def floor_divide(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    out = np.floor_divide(a_np, b_np)
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype), copy=False), a.dtype, a.device)


def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
    arr = _to_numpy(input)
    norm_shape = tuple(normalized_shape)
    axis = tuple(range(arr.ndim - len(norm_shape), arr.ndim))
    variance = np.mean(arr ** 2, axis=axis, keepdims=True)
    out = arr / np.sqrt(variance + eps)
    if weight is not None:
        out = out * _to_numpy(weight).reshape((1,) * (arr.ndim - len(norm_shape)) + norm_shape)
    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)


def conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Conv2d forward using numpy. Input: (N,C,H,W), Weight: (O,C/g,kH,kW)."""
    inp = _to_numpy(input)
    w = _to_numpy(weight)
    N, C_in, H, W = inp.shape
    C_out, C_in_g, kH, kW = w.shape
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    # Effective kernel size with dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1

    # Pad input
    if pH > 0 or pW > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')

    out = np.zeros((N, C_out, H_out, W_out), dtype=inp.dtype)

    for g in range(groups):
        c_out_per_g = C_out // groups
        c_out_s = g * c_out_per_g
        c_in_s = g * C_in_g
        for co_local in range(c_out_per_g):
            co = c_out_s + co_local
            for ci_local in range(C_in_g):
                ci = c_in_s + ci_local
                kernel = w[co, ci_local]
                # For dilated kernels, use the dilated positions
                for oh in range(H_out):
                    for ow in range(W_out):
                        val = 0.0
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * sH + kh * dH
                                iw = ow * sW + kw * dW
                                # Broadcasting over batch dimension
                                out[:, co, oh, ow] += inp[:, ci, ih, iw] * kernel[kh, kw]

    if bias is not None:
        b = _to_numpy(bias)
        out += b[np.newaxis, :, np.newaxis, np.newaxis]

    return _from_numpy(np.ascontiguousarray(out.astype(inp.dtype)), input.dtype, input.device)


def conv1d(input, weight, bias=None, stride=(1,), padding=(0,), dilation=(1,), groups=1):
    """Conv1d via conv2d: unsqueeze spatial dim."""
    inp = _to_numpy(input)   # (N, C, L)
    w = _to_numpy(weight)    # (O, C/g, kL)
    # Add H=1 dimension
    inp4d = inp[:, :, np.newaxis, :]  # (N, C, 1, L)
    w4d = w[:, :, np.newaxis, :]      # (O, C/g, 1, kL)
    inp_t = _from_numpy(np.ascontiguousarray(inp4d), input.dtype, input.device)
    w_t = _from_numpy(np.ascontiguousarray(w4d), weight.dtype, weight.device)
    out = conv2d(inp_t, w_t, bias, stride=(1, stride[0]),
                 padding=(0, padding[0]), dilation=(1, dilation[0]), groups=groups)
    out_np = _to_numpy(out)[:, :, 0, :]  # (N, C_out, L_out)
    return _from_numpy(np.ascontiguousarray(out_np), input.dtype, input.device)


def conv_transpose2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0),
                     output_padding=(0, 0), groups=1, dilation=(1, 1)):
    """Transposed conv2d using numpy."""
    inp = _to_numpy(input)
    w = _to_numpy(weight)   # (C_in, C_out/g, kH, kW) — note: transposed weight layout
    N, C_in, H_in, W_in = inp.shape
    C_in_w, C_out_g, kH, kW = w.shape
    sH, sW = stride
    pH, pW = padding
    opH, opW = output_padding
    dH, dW = dilation

    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    H_out = (H_in - 1) * sH - 2 * pH + ekH + opH
    W_out = (W_in - 1) * sW - 2 * pW + ekW + opW
    C_out = C_out_g * groups

    # We compute in a padded buffer and then slice
    H_buf = (H_in - 1) * sH + ekH
    W_buf = (W_in - 1) * sW + ekW
    buf = np.zeros((N, C_out, H_buf, W_buf), dtype=inp.dtype)

    for g in range(groups):
        c_in_per_g = C_in // groups
        c_in_s = g * c_in_per_g
        c_out_s = g * C_out_g
        for ci_local in range(c_in_per_g):
            ci = c_in_s + ci_local
            for co_local in range(C_out_g):
                co = c_out_s + co_local
                kernel = w[ci_local, co_local]
                for ih in range(H_in):
                    for iw in range(W_in):
                        oh_start = ih * sH
                        ow_start = iw * sW
                        for kh in range(kH):
                            for kw in range(kW):
                                oh = oh_start + kh * dH
                                ow = ow_start + kw * dW
                                buf[:, co, oh, ow] += inp[:, ci, ih, iw] * kernel[kh, kw]

    # Slice to remove padding and apply output_padding
    out = buf[:, :, pH:pH + H_out, pW:pW + W_out]

    if bias is not None:
        b = _to_numpy(bias)
        out = out + b[np.newaxis, :, np.newaxis, np.newaxis]

    return _from_numpy(np.ascontiguousarray(out.astype(inp.dtype)), input.dtype, input.device)


def conv_transpose1d(input, weight, bias=None, stride=(1,), padding=(0,),
                     output_padding=(0,), groups=1, dilation=(1,)):
    """Conv_transpose1d via conv_transpose2d: unsqueeze spatial dim."""
    inp = _to_numpy(input)   # (N, C, L)
    w = _to_numpy(weight)    # (C_in, C_out/g, kL)
    inp4d = inp[:, :, np.newaxis, :]
    w4d = w[:, :, np.newaxis, :]
    inp_t = _from_numpy(np.ascontiguousarray(inp4d), input.dtype, input.device)
    w_t = _from_numpy(np.ascontiguousarray(w4d), weight.dtype, weight.device)
    out = conv_transpose2d(inp_t, w_t, bias, stride=(1, stride[0]),
                           padding=(0, padding[0]), output_padding=(0, output_padding[0]),
                           groups=groups, dilation=(1, dilation[0]))
    out_np = _to_numpy(out)[:, :, 0, :]
    return _from_numpy(np.ascontiguousarray(out_np), input.dtype, input.device)


def max_pool2d(input, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """MaxPool2d using numpy."""
    inp = _to_numpy(input)  # (N, C, H, W)
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dH, dW = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    N, C, H, W = inp.shape
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    if ceil_mode:
        H_out = math.ceil((H + 2 * pH - ekH) / sH) + 1
        W_out = math.ceil((W + 2 * pW - ekW) / sW) + 1
    else:
        H_out = (H + 2 * pH - ekH) // sH + 1
        W_out = (W + 2 * pW - ekW) // sW + 1

    if pH > 0 or pW > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                     mode='constant', constant_values=-np.inf)

    out = np.full((N, C, H_out, W_out), -np.inf, dtype=inp.dtype)
    for oh in range(H_out):
        for ow in range(W_out):
            for kh in range(kH):
                for kw in range(kW):
                    ih = oh * sH + kh * dH
                    iw = ow * sW + kw * dW
                    if ih < inp.shape[2] and iw < inp.shape[3]:
                        out[:, :, oh, ow] = np.maximum(out[:, :, oh, ow], inp[:, :, ih, iw])

    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)


def avg_pool2d(input, kernel_size, stride, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    """AvgPool2d using numpy."""
    inp = _to_numpy(input)  # (N, C, H, W)
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)
    N, C, H, W = inp.shape
    if ceil_mode:
        H_out = math.ceil((H + 2 * pH - kH) / sH) + 1
        W_out = math.ceil((W + 2 * pW - kW) / sW) + 1
    else:
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

    if pH > 0 or pW > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')

    out = np.zeros((N, C, H_out, W_out), dtype=inp.dtype)
    for oh in range(H_out):
        for ow in range(W_out):
            h_start = oh * sH
            w_start = ow * sW
            h_end = min(h_start + kH, inp.shape[2])
            w_end = min(w_start + kW, inp.shape[3])
            window = inp[:, :, h_start:h_end, w_start:w_end]
            if divisor_override is not None:
                out[:, :, oh, ow] = window.sum(axis=(-2, -1)) / divisor_override
            elif count_include_pad:
                out[:, :, oh, ow] = window.sum(axis=(-2, -1)) / (kH * kW)
            else:
                # Only count non-padded elements
                actual_h = min(h_end, H + pH) - max(h_start, pH)
                actual_w = min(w_end, W + pW) - max(w_start, pW)
                count = max(actual_h * actual_w, 1)
                out[:, :, oh, ow] = window.sum(axis=(-2, -1)) / count

    return _from_numpy(np.ascontiguousarray(out.astype(inp.dtype)), input.dtype, input.device)


def adaptive_avg_pool2d(input, output_size):
    """AdaptiveAvgPool2d using numpy."""
    inp = _to_numpy(input)  # (N, C, H, W)
    N, C, H, W = inp.shape
    if isinstance(output_size, int):
        oH = oW = output_size
    else:
        oH, oW = output_size
    out = np.zeros((N, C, oH, oW), dtype=inp.dtype)
    for oh in range(oH):
        h_start = oh * H // oH
        h_end = (oh + 1) * H // oH
        for ow in range(oW):
            w_start = ow * W // oW
            w_end = (ow + 1) * W // oW
            out[:, :, oh, ow] = inp[:, :, h_start:h_end, w_start:w_end].mean(axis=(-2, -1))
    return _from_numpy(np.ascontiguousarray(out.astype(inp.dtype)), input.dtype, input.device)


# ---------------------------------------------------------------------------
# Group 1: Math ops
# ---------------------------------------------------------------------------

def sub(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"sub_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"sub_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(a_np - b_np, a.dtype, a.device)


def log1p(a):
    return _from_numpy(np.log1p(_to_numpy(a)), a.dtype, a.device)


def expm1(a):
    return _from_numpy(np.expm1(_to_numpy(a)), a.dtype, a.device)


def reciprocal(a):
    return _from_numpy(1.0 / _to_numpy(a), a.dtype, a.device)


def maximum(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"maximum_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"maximum_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.maximum(a_np, b_np), a.dtype, a.device)


def minimum(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"minimum_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"minimum_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.minimum(a_np, b_np), a.dtype, a.device)


def dot(a, b):
    if a.dtype != b.dtype:
        raise RuntimeError("dot: expected both vectors to have same dtype")
    a_np = np.ascontiguousarray(_to_numpy(a))
    b_np = np.ascontiguousarray(_to_numpy(b))
    if _can_use_blas(a_np) and _can_use_blas(b_np):
        n = a_np.size
        if a_np.dtype == np.float32:
            val = _accel.cblas_sdot(n, a_np.ctypes.data, 1, b_np.ctypes.data, 1)
        else:
            val = _accel.cblas_ddot(n, a_np.ctypes.data, 1, b_np.ctypes.data, 1)
        return _from_numpy(np.array(val, dtype=a_np.dtype), a.dtype, a.device)
    return _from_numpy(np.dot(a_np, b_np), a.dtype, a.device)


def outer(a, b):
    return _from_numpy(np.outer(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def inner(a, b):
    return _from_numpy(np.inner(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)


def mv(a, b):
    a_np = np.ascontiguousarray(_to_numpy(a))
    b_np = np.ascontiguousarray(_to_numpy(b))
    if a_np.ndim == 2 and b_np.ndim == 1 and _can_use_blas(a_np) and _can_use_blas(b_np):
        M, N = a_np.shape
        out = np.empty(M, dtype=a_np.dtype)
        if a_np.dtype == np.float32:
            _accel.cblas_sgemv(111, M, N, 1.0,
                               a_np.ctypes.data, N, b_np.ctypes.data, 1,
                               0.0, out.ctypes.data, 1)
        else:
            return _from_numpy(np.dot(a_np, b_np), a.dtype, a.device)
        return _from_numpy(out, a.dtype, a.device)
    return _from_numpy(np.dot(a_np, b_np), a.dtype, a.device)


def cross(a, b, dim=-1):
    if a.dtype != b.dtype:
        raise RuntimeError("cross: expected both inputs to have same dtype")
    a_np = np.moveaxis(_to_numpy(a), dim, -1)
    b_np = np.moveaxis(_to_numpy(b), dim, -1)
    out = np.cross(a_np, b_np)
    out = np.moveaxis(out, -1, dim)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def tensordot(a, b, dims=2):
    if a.dtype != b.dtype:
        raise RuntimeError("tensordot: expected both inputs to have same dtype")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    if isinstance(dims, int):
        out = np.tensordot(a_np, b_np, axes=dims)
    elif isinstance(dims, (list, tuple)) and len(dims) == 2:
        out = np.tensordot(a_np, b_np, axes=dims)
    else:
        out = np.tensordot(a_np, b_np, axes=dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def einsum(equation, *operands):
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    ops_np = [_to_numpy(op) for op in operands]
    out = np.einsum(equation, *ops_np)
    return _from_numpy(np.ascontiguousarray(out), operands[0].dtype, operands[0].device)


# ---------------------------------------------------------------------------
# Group 2: Logical ops
# ---------------------------------------------------------------------------

def logical_and(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.logical_and(a_np, b_np), bool_dtype, a.device)


def logical_or(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.logical_or(a_np, b_np), bool_dtype, a.device)


def logical_not(a):
    return _from_numpy(np.logical_not(_to_numpy(a)), bool_dtype, a.device)


def logical_xor(a, b):
    a_np = _to_numpy(a).astype(bool)
    b_np = (_to_numpy(b) if isinstance(b, Tensor) else np.array(b)).astype(bool)
    return _from_numpy(np.logical_xor(a_np, b_np), bool_dtype, a.device)


# ---------------------------------------------------------------------------
# Group 3: Bitwise ops
# ---------------------------------------------------------------------------

def bitwise_and(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.bitwise_and(a_np, b_np), a.dtype, a.device)


def bitwise_or(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.bitwise_or(a_np, b_np), a.dtype, a.device)


def bitwise_xor(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.bitwise_xor(a_np, b_np), a.dtype, a.device)


def bitwise_not(a):
    return _from_numpy(np.bitwise_not(_to_numpy(a)), a.dtype, a.device)


# ---------------------------------------------------------------------------
# Group 4: Random in-place op
# ---------------------------------------------------------------------------

def randint_(a, low, high=None, generator=None):
    """In-place randint — fills tensor a with random integers from [low, high)."""
    if high is None:
        low, high = 0, low
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.randint(int(low), int(high), size=arr.shape)
    return a


def random_(a, from_=0, to=None, generator=None):
    """In-place random — fills tensor with random values from [from_, to)."""
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    if to is None:
        if np.issubdtype(arr.dtype, np.floating):
            to = 2**24 if arr.dtype == np.float32 else 2**53
        else:
            to = int(np.iinfo(arr.dtype).max) + 1
    arr[...] = rng.randint(int(from_), int(to), size=arr.shape).astype(arr.dtype)
    return a


# ---------------------------------------------------------------------------
# Group 5: Shape ops
# ---------------------------------------------------------------------------

def flatten(a, start_dim=0, end_dim=-1):
    arr = _to_numpy(a)
    ndim = arr.ndim
    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    new_shape = arr.shape[:start] + (-1,) + arr.shape[end + 1:]
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def unflatten(a, dim, sizes):
    arr = _to_numpy(a)
    ndim = arr.ndim
    d = dim if dim >= 0 else dim + ndim
    new_shape = arr.shape[:d] + tuple(sizes) + arr.shape[d + 1:]
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def broadcast_to(a, shape):
    arr = _to_numpy(a)
    out = np.broadcast_to(arr, shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def movedim(a, source, destination):
    arr = _to_numpy(a)
    out = np.moveaxis(arr, source, destination)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def diagonal(a, offset=0, dim1=0, dim2=1):
    arr = _to_numpy(a)
    out = np.diagonal(arr, offset=offset, axis1=dim1, axis2=dim2)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


# ---------------------------------------------------------------------------
# Group 6: Search ops
# ---------------------------------------------------------------------------

def unique(a, sorted=True, return_inverse=False, return_counts=False, dim=None):
    arr = _to_numpy(a)
    if dim is None:
        flat = arr.flatten()
        result = np.unique(flat, return_inverse=return_inverse, return_counts=return_counts)
    else:
        result = np.unique(arr, return_inverse=return_inverse, return_counts=return_counts, axis=dim)
    if isinstance(result, tuple):
        out = []
        for i, r in enumerate(result):
            r_cont = np.ascontiguousarray(r)
            if i == 0:
                out.append(_from_numpy(r_cont, a.dtype, a.device))
            else:
                out.append(_from_numpy(r_cont.astype(np.int64), int64_dtype, a.device))
        return tuple(out)
    return _from_numpy(np.ascontiguousarray(result), a.dtype, a.device)


def searchsorted(sorted_seq, values, out_int32=False, right=False, side=None, sorter=None):
    seq_np = _to_numpy(sorted_seq)
    val_np = _to_numpy(values) if isinstance(values, Tensor) else np.array(values)
    side_str = side if side is not None else ('right' if right else 'left')
    if sorter is not None:
        sorter_np = _to_numpy(sorter).astype(np.int64)
        out = np.searchsorted(seq_np.flatten(), val_np.flatten(), side=side_str, sorter=sorter_np)
    else:
        if seq_np.ndim == 1:
            out = np.searchsorted(seq_np, val_np, side=side_str)
        else:
            out = np.zeros_like(val_np, dtype=np.int64)
            for i in range(seq_np.shape[0]):
                out[i] = np.searchsorted(seq_np[i], val_np[i], side=side_str)
    out_dtype_np = np.int32 if out_int32 else np.int64
    return _from_numpy(out.astype(out_dtype_np), int64_dtype, sorted_seq.device)


def kthvalue(a, k, dim=-1, keepdim=False):
    arr = _to_numpy(a)
    if dim < 0:
        dim = dim + arr.ndim
    sorted_idx = np.argsort(arr, axis=dim)
    kth_idx = np.take(sorted_idx, [k - 1], axis=dim)
    values = np.take_along_axis(arr, kth_idx, axis=dim)
    if not keepdim:
        values = np.squeeze(values, axis=dim)
        kth_idx = np.squeeze(kth_idx, axis=dim)
    return (
        _from_numpy(np.ascontiguousarray(values), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(kth_idx.astype(np.int64)), int64_dtype, a.device),
    )


def instance_norm(input, weight=None, bias=None, running_mean=None, running_var=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=False):
    arr = _to_numpy(input)
    ndim = len(arr.shape)
    N = arr.shape[0]
    C = arr.shape[1] if ndim >= 2 else 1
    spatial_axes = tuple(range(2, ndim))

    if use_input_stats:
        mean = arr.mean(axis=spatial_axes, keepdims=True)
        var = arr.var(axis=spatial_axes, keepdims=True)
        if running_mean is not None:
            rm = _to_numpy(running_mean)
            batch_mean = mean.reshape(N, C).mean(axis=0)
            rm[:] = (1 - momentum) * rm + momentum * batch_mean
        if running_var is not None:
            rv = _to_numpy(running_var)
            batch_var = var.reshape(N, C).mean(axis=0)
            rv[:] = (1 - momentum) * rv + momentum * batch_var
    else:
        rm = _to_numpy(running_mean)
        rv = _to_numpy(running_var)
        shape = [1, C] + [1] * (ndim - 2)
        mean = rm.reshape(shape)
        var = rv.reshape(shape)

    normalized = (arr - mean) / np.sqrt(var + eps)

    if weight is not None:
        shape = [1, C] + [1] * (ndim - 2)
        normalized = normalized * _to_numpy(weight).reshape(shape)
    if bias is not None:
        shape = [1, C] + [1] * (ndim - 2)
        normalized = normalized + _to_numpy(bias).reshape(shape)

    return _from_numpy(normalized, input.dtype, input.device)


def median(a, dim=None, keepdim=False):
    arr = _to_numpy(a)
    if dim is None:
        out = np.median(arr.flatten())
        return _from_numpy(np.array(out, dtype=arr.dtype), a.dtype, a.device)
    else:
        if dim < 0:
            dim = dim + arr.ndim
        sorted_idx = np.argsort(arr, axis=dim)
        n = arr.shape[dim]
        mid = n // 2
        med_idx = np.take(sorted_idx, [mid], axis=dim)
        values = np.take_along_axis(arr, med_idx, axis=dim)
        if not keepdim:
            values = np.squeeze(values, axis=dim)
            med_idx = np.squeeze(med_idx, axis=dim)
        return (
            _from_numpy(np.ascontiguousarray(values), a.dtype, a.device),
            _from_numpy(np.ascontiguousarray(med_idx.astype(np.int64)), int64_dtype, a.device),
        )


# ---------------------------------------------------------------------------
# Group 7: New math ops for Tensor API alignment
# ---------------------------------------------------------------------------

def logsumexp(a, dim, keepdim=False):
    """Numerically stable logsumexp: log(sum(exp(x), dim))."""
    arr = _to_numpy(a)
    max_val = np.max(arr, axis=dim, keepdims=True)
    exp_shifted = np.exp(arr - max_val)
    sum_exp = np.sum(exp_shifted, axis=dim, keepdims=keepdim)
    if keepdim:
        out = np.log(sum_exp) + max_val
    else:
        out = np.log(sum_exp) + np.squeeze(max_val, axis=dim)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def trace(a):
    """Sum of diagonal elements (2D only)."""
    arr = _to_numpy(a)
    if arr.ndim != 2:
        raise RuntimeError(
            f"trace: expected a matrix (2-D tensor), but got {arr.ndim}-D tensor"
        )
    out = np.trace(arr)
    return _from_numpy(np.array(out, dtype=arr.dtype), a.dtype, a.device)


def det(a):
    """Determinant of a square matrix (or batch of square matrices)."""
    arr = _to_numpy(a)
    if arr.ndim < 2:
        raise RuntimeError(f"det: input must be at least 2-D, got {arr.ndim}-D")
    if arr.shape[-2] != arr.shape[-1]:
        raise RuntimeError(
            f"det: input must be a square matrix, got shape {arr.shape}"
        )
    out = np.linalg.det(arr.astype(np.float64))
    return _from_numpy(np.ascontiguousarray(out).astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)


def matrix_power(a, n):
    """Matrix raised to the integer power n."""
    arr = _to_numpy(a)
    if arr.ndim < 2:
        raise RuntimeError(
            f"matrix_power: input must be at least 2-D, got {arr.ndim}-D"
        )
    if arr.shape[-2] != arr.shape[-1]:
        raise RuntimeError(
            f"matrix_power: input must be a square matrix, got shape {arr.shape}"
        )
    out = np.linalg.matrix_power(arr, n)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def dist(a, b, p=2):
    """p-norm distance between two tensors (flattened)."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    diff = a_np.ravel() - b_np.ravel()
    out = np.linalg.norm(diff, ord=p)
    return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)


def renorm(a, p, dim, maxnorm):
    """Renormalize tensor: each sub-tensor along dim has norm <= maxnorm."""
    arr = _to_numpy(a)
    # Compute the norm along all axes except dim
    norm = np.linalg.norm(arr, ord=p, axis=dim, keepdims=True)
    # Scale: if norm > maxnorm, scale down; otherwise keep unchanged
    scale = np.where(norm > maxnorm, maxnorm / (norm + 1e-7), 1.0)
    out = arr * scale
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def nansum(a, dim=None, keepdim=False):
    """Sum ignoring NaN values."""
    arr = _to_numpy(a)
    if dim is None:
        out = np.nansum(arr)
        return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    else:
        out = np.nansum(arr, axis=dim, keepdims=keepdim)
        return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def nanmean(a, dim=None, keepdim=False):
    """Mean ignoring NaN values."""
    arr = _to_numpy(a)
    if dim is None:
        out = np.nanmean(arr)
        return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    else:
        out = np.nanmean(arr, axis=dim, keepdims=keepdim)
        return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def argwhere(a):
    """Returns indices of non-zero elements as a 2D tensor (shape [N, ndim])."""
    arr = _to_numpy(a)
    out = np.argwhere(arr)
    return _from_numpy(out.astype(np.int64), int64_dtype, a.device)


def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    """Batch matrix-matrix product: beta * input + alpha * (batch1 @ batch2)."""
    input_np = _to_numpy(input)
    batch1_np = _to_numpy(batch1)
    batch2_np = _to_numpy(batch2)

    if batch1_np.ndim != 3 or batch2_np.ndim != 3:
        raise RuntimeError("baddbmm: batch1 and batch2 must be 3-D tensors")

    bmm_result = batch1_np @ batch2_np
    out = beta * input_np + alpha * bmm_result
    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)


def cummin(a, dim):
    """Cumulative minimum along a dimension, returns (values, indices) namedtuple."""
    arr = _to_numpy(a)
    ndim = arr.ndim
    if dim < 0:
        dim = dim + ndim

    values = np.minimum.accumulate(arr, axis=dim)

    # Compute indices: for each position i along dim, index is where min first occurred
    n = arr.shape[dim]
    indices = np.zeros_like(arr, dtype=np.int64)

    # Iterate over the dimension to compute the argmin up to each point
    idx_shape = list(arr.shape)
    idx_shape[dim] = 1
    running_min = np.take(arr, [0], axis=dim)
    running_idx = np.zeros(idx_shape, dtype=np.int64)

    slc_base = [slice(None)] * ndim
    for i in range(n):
        slc = slc_base[:]
        slc[dim] = slice(i, i + 1)
        current = arr[tuple(slc)]
        new_min_mask = current < running_min
        running_idx = np.where(new_min_mask, i, running_idx)
        running_min = np.minimum(running_min, current)
        indices_slc = slc_base[:]
        indices_slc[dim] = i
        indices[tuple(indices_slc)] = running_idx.squeeze(axis=dim)

    from collections import namedtuple
    CumminResult = namedtuple("cummin", ["values", "indices"])
    return CumminResult(
        _from_numpy(np.ascontiguousarray(values), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(indices), int64_dtype, a.device),
    )


# ---------------------------------------------------------------------------
# Top-level gap-fill ops (Category C2)
# ---------------------------------------------------------------------------

def diff(a, n=1, dim=-1, prepend=None, append=None):
    """Compute the n-th discrete difference along the given dim."""
    arr = _to_numpy(a)
    if prepend is not None or append is not None:
        pieces = []
        if prepend is not None:
            pieces.append(_to_numpy(prepend))
        pieces.append(arr)
        if append is not None:
            pieces.append(_to_numpy(append))
        arr = np.concatenate(pieces, axis=dim)
    out = np.diff(arr, n=n, axis=dim)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def bincount(a, weights=None, minlength=0):
    """Count number of occurrences of each value in a 1-D int tensor."""
    arr = _to_numpy(a).astype(np.int64).ravel()
    w = _to_numpy(weights).ravel() if weights is not None else None
    out = np.bincount(arr, weights=w, minlength=minlength)
    out_dtype = a.dtype if weights is None else weights.dtype
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(out_dtype))), out_dtype, a.device)


def cdist(x1, x2, p=2.0):
    """Batched pairwise distance between two sets of vectors."""
    a = _to_numpy(x1).astype(np.float64)
    b = _to_numpy(x2).astype(np.float64)
    if a.ndim == 2:
        a = a[np.newaxis]
        b = b[np.newaxis]
        squeeze = True
    else:
        squeeze = False
    batch = a.shape[0]
    results = []
    for i in range(batch):
        # Manual pairwise distance: (M, 1, D) - (1, N, D) -> (M, N, D)
        diff = a[i][:, np.newaxis, :] - b[i][np.newaxis, :, :]
        if p == 2.0:
            d = np.sqrt(np.sum(diff ** 2, axis=-1))
        elif p == 1.0:
            d = np.sum(np.abs(diff), axis=-1)
        elif p == float('inf'):
            d = np.max(np.abs(diff), axis=-1)
        elif p == 0.0:
            d = np.sum(diff != 0, axis=-1).astype(np.float64)
        else:
            d = np.sum(np.abs(diff) ** p, axis=-1) ** (1.0 / p)
        results.append(d)
    out = np.stack(results)
    if squeeze:
        out = out[0]
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(x1.dtype))), x1.dtype, x1.device)


def aminmax(a, dim=None, keepdim=False):
    """Returns the min and max of a tensor."""
    from collections import namedtuple
    arr = _to_numpy(a)
    if dim is None:
        mn = np.min(arr)
        mx = np.max(arr)
        mn_t = _from_numpy(np.array(mn, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
        mx_t = _from_numpy(np.array(mx, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    else:
        mn = np.min(arr, axis=dim, keepdims=keepdim)
        mx = np.max(arr, axis=dim, keepdims=keepdim)
        mn_t = _from_numpy(np.ascontiguousarray(mn), a.dtype, a.device)
        mx_t = _from_numpy(np.ascontiguousarray(mx), a.dtype, a.device)
    AminmaxResult = namedtuple("aminmax", ["min", "max"])
    return AminmaxResult(mn_t, mx_t)


def quantile(a, q, dim=None, keepdim=False):
    """Compute the q-th quantile of the input tensor."""
    arr = _to_numpy(a).astype(np.float64)
    q_val = _to_numpy(q) if hasattr(q, '_numpy_view') else np.asarray(q, dtype=np.float64)
    if dim is None:
        out = np.quantile(arr, q_val)
    else:
        out = np.quantile(arr, q_val, axis=dim, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def nanquantile(a, q, dim=None, keepdim=False):
    """Compute the q-th quantile ignoring NaN values."""
    arr = _to_numpy(a).astype(np.float64)
    q_val = _to_numpy(q) if hasattr(q, '_numpy_view') else np.asarray(q, dtype=np.float64)
    if dim is None:
        out = np.nanquantile(arr, q_val)
    else:
        out = np.nanquantile(arr, q_val, axis=dim, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def nanmedian(a, dim=None, keepdim=False):
    """Median ignoring NaN values. Returns (values, indices) when dim is given."""
    arr = _to_numpy(a).astype(np.float64)
    if dim is None:
        out = np.nanmedian(arr)
        return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    else:
        values = np.nanmedian(arr, axis=dim, keepdims=keepdim)
        # Compute indices: for each slice along dim, find index of the median value
        n = arr.shape[dim]
        sorted_arr = np.sort(arr, axis=dim)
        # Count non-nan along dim
        not_nan = ~np.isnan(arr)
        count = np.sum(not_nan, axis=dim, keepdims=True)
        # Median index in sorted order
        med_idx_sorted = (count - 1) // 2
        # For each position, find the index in the original array
        sorted_indices = np.argsort(arr, axis=dim)
        # Gather the median index from sorted_indices
        indices = np.take_along_axis(sorted_indices, med_idx_sorted.astype(np.intp), axis=dim)
        if not keepdim:
            indices = np.squeeze(indices, axis=dim)
        from collections import namedtuple
        NanmedianResult = namedtuple("nanmedian", ["values", "indices"])
        return NanmedianResult(
            _from_numpy(np.ascontiguousarray(values.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
            _from_numpy(np.ascontiguousarray(indices.astype(np.int64)), int64_dtype, a.device),
        )


def histc(a, bins=100, min=0, max=0):
    """Histogram with equal-width bins (1-D output count tensor)."""
    arr = _to_numpy(a).ravel().astype(np.float64)
    lo = float(min)
    hi = float(max)
    if lo == 0 and hi == 0:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    out, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def histogram(a, bins, range=None, weight=None, density=False):
    """Histogram returning (hist, bin_edges)."""
    arr = _to_numpy(a).ravel().astype(np.float64)
    bins_val = _to_numpy(bins) if hasattr(bins, '_numpy_view') else bins
    w = _to_numpy(weight).ravel().astype(np.float64) if weight is not None else None
    hist, edges = np.histogram(arr, bins=bins_val, range=range, weights=w, density=density)
    return (
        _from_numpy(np.ascontiguousarray(hist.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(edges.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
    )


def bucketize(a, boundaries, out_int32=False, right=False):
    """Maps values to bucket indices using boundaries."""
    arr = _to_numpy(a)
    b = _to_numpy(boundaries).ravel()
    side = 'right' if not right else 'left'
    out = np.searchsorted(b, arr, side=side)
    out_np_dtype = np.int32 if out_int32 else np.int64
    out_dtype = int64_dtype
    return _from_numpy(np.ascontiguousarray(out.astype(out_np_dtype)), out_dtype, a.device)


def isneginf(a):
    """Returns a bool tensor indicating negative infinity."""
    arr = _to_numpy(a)
    out = np.isneginf(arr)
    return _from_numpy(np.ascontiguousarray(out), bool_dtype, a.device)


def isposinf(a):
    """Returns a bool tensor indicating positive infinity."""
    arr = _to_numpy(a)
    out = np.isposinf(arr)
    return _from_numpy(np.ascontiguousarray(out), bool_dtype, a.device)


def isreal(a):
    """Returns a bool tensor indicating real-valued elements."""
    arr = _to_numpy(a)
    out = np.isreal(arr)
    if out.ndim == 0:
        out = np.array(out)
    return _from_numpy(np.ascontiguousarray(out.astype(np.bool_)), bool_dtype, a.device)


def isin(elements, test_elements):
    """Tests if each element is in test_elements."""
    e = _to_numpy(elements)
    te = _to_numpy(test_elements)
    out = np.isin(e, te)
    return _from_numpy(np.ascontiguousarray(out), bool_dtype, elements.device)


def heaviside(a, values):
    """Heaviside step function."""
    a_np = _to_numpy(a)
    v_np = _to_numpy(values)
    out = np.heaviside(a_np, v_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


# ---------------------------------------------------------------------------
# torch.linalg ops
# ---------------------------------------------------------------------------

def linalg_cholesky(a, upper=False):
    """Cholesky decomposition."""
    arr = _to_numpy(a).astype(np.float64)
    L = np.linalg.cholesky(arr)
    if upper:
        # Transpose the last two dims
        L = np.swapaxes(L, -2, -1).conj()
    return _from_numpy(np.ascontiguousarray(L.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_cond(a, p=None):
    """Condition number of a matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.cond(arr, p=p)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_det(a):
    """Determinant of a square matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.det(arr)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_eig(a):
    """Eigenvalue decomposition of a square matrix."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(arr)
    return (
        _from_numpy(np.ascontiguousarray(eigenvalues), complex128_dtype, a.device),
        _from_numpy(np.ascontiguousarray(eigenvectors), complex128_dtype, a.device),
    )


def linalg_eigh(a, UPLO='L'):
    """Eigenvalue decomposition of a symmetric/Hermitian matrix."""
    arr = _to_numpy(a).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(arr, UPLO=UPLO)
    return (
        _from_numpy(np.ascontiguousarray(eigenvalues.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(eigenvectors.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
    )


def linalg_eigvals(a):
    """Eigenvalues of a square matrix."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.eigvals(arr)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def linalg_eigvalsh(a, UPLO='L'):
    """Eigenvalues of a symmetric/Hermitian matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.eigvalsh(arr, UPLO=UPLO)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_householder_product(input, tau):
    """Computes the first n columns of the product of Householder matrices."""
    A = _to_numpy(input).astype(np.float64)
    tau_np = _to_numpy(tau).astype(np.float64)
    m, n = A.shape[-2], A.shape[-1]
    k = tau_np.shape[-1]
    Q = np.eye(m, dtype=np.float64)
    for i in range(k):
        v = np.zeros(m, dtype=np.float64)
        v[i] = 1.0
        v[i + 1:] = A[i + 1:, i]
        Q = Q - tau_np[i] * np.outer(Q @ v, v)
    Q = Q[:, :n]
    return _from_numpy(np.ascontiguousarray(Q.astype(to_numpy_dtype(input.dtype))), input.dtype, input.device)


def linalg_inv(a):
    """Inverse of a square matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.inv(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_lstsq(a, b, rcond=None, driver=None):
    """Least-squares solution to a linear matrix equation."""
    from collections import namedtuple
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    if rcond is None:
        rcond = -1 if np.lib.NumpyVersion(np.__version__) < '2.0.0' else None
    solution, residuals, rank, singular_values = np.linalg.lstsq(a_np, b_np, rcond=rcond)
    LstsqResult = namedtuple("LstsqResult", ["solution", "residuals", "rank", "singular_values"])
    return LstsqResult(
        _from_numpy(np.ascontiguousarray(solution.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(np.atleast_1d(residuals).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        rank,
        _from_numpy(np.ascontiguousarray(singular_values.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
    )


def linalg_lu(a, pivot=True):
    """LU decomposition with partial pivoting."""
    from collections import namedtuple
    from scipy import linalg as scipy_linalg
    arr = _to_numpy(a).astype(np.float64)
    P_mat, L, U = scipy_linalg.lu(arr)
    LUResult = namedtuple("LUResult", ["P", "L", "U"])
    dt = to_numpy_dtype(a.dtype)
    return LUResult(
        _from_numpy(np.ascontiguousarray(P_mat.astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(L.astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(U.astype(dt)), a.dtype, a.device),
    )


def linalg_lu_factor(a, pivot=True):
    """Compact LU factorization."""
    from collections import namedtuple
    from scipy import linalg as scipy_linalg
    arr = _to_numpy(a).astype(np.float64)
    lu, piv = scipy_linalg.lu_factor(arr)
    LUFactorResult = namedtuple("LUFactorResult", ["LU", "pivots"])
    return LUFactorResult(
        _from_numpy(np.ascontiguousarray(lu.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(piv.astype(np.int32)), int64_dtype, a.device),
    )


def linalg_lu_solve(LU, pivots, B, left=True, adjoint=False):
    """Solve using LU factorization."""
    from scipy import linalg as scipy_linalg
    lu_np = _to_numpy(LU).astype(np.float64)
    piv_np = _to_numpy(pivots).astype(np.int32)
    b_np = _to_numpy(B).astype(np.float64)
    if not left:
        b_np = b_np.T
    trans = 1 if adjoint else 0
    out = scipy_linalg.lu_solve((lu_np, piv_np), b_np, trans=trans)
    if not left:
        out = out.T
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(B.dtype))), B.dtype, B.device)


def linalg_matrix_exp(a):
    """Matrix exponential."""
    from scipy import linalg as scipy_linalg
    arr = _to_numpy(a).astype(np.float64)
    out = scipy_linalg.expm(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_matrix_norm(a, ord='fro', dim=(-2, -1), keepdim=False):
    """Matrix norm."""
    arr = _to_numpy(a).astype(np.float64)
    if isinstance(dim, (list, tuple)) and len(dim) == 2:
        axis = tuple(dim)
    else:
        axis = dim
    if ord == 'fro':
        out = np.sqrt(np.sum(arr ** 2, axis=axis, keepdims=keepdim))
    elif ord == 'nuc':
        # Nuclear norm = sum of singular values
        out = np.sum(np.linalg.svd(arr, compute_uv=False), axis=-1, keepdims=keepdim)
    else:
        out = np.linalg.norm(arr, ord=ord, axis=axis, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_matrix_power(a, n):
    """Matrix raised to integer power n."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.matrix_power(arr, n)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_matrix_rank(a, atol=None, rtol=None, hermitian=False):
    """Numerical rank of a matrix."""
    arr = _to_numpy(a).astype(np.float64)
    if atol is not None or rtol is not None:
        s = np.linalg.svd(arr, compute_uv=False)
        tol = 0.0
        if atol is not None:
            tol = max(tol, atol)
        if rtol is not None:
            tol = max(tol, rtol * s[..., 0])
        rank = np.sum(s > tol, axis=-1)
    else:
        rank = np.linalg.matrix_rank(arr)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(rank).astype(np.int64)), int64_dtype, a.device)


def linalg_multi_dot(tensors):
    """Efficiently multiply 2+ matrices using np.linalg.multi_dot."""
    arrays = [_to_numpy(t).astype(np.float64) for t in tensors]
    out = np.linalg.multi_dot(arrays)
    dt = tensors[0].dtype
    dev = tensors[0].device
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(dt))), dt, dev)


def linalg_norm(a, ord=None, dim=None, keepdim=False):
    """Vector or matrix norm."""
    arr = _to_numpy(a).astype(np.float64)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            axis = tuple(dim)
        else:
            axis = dim
    else:
        axis = None
    out = np.linalg.norm(arr, ord=ord, axis=axis, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_pinv(a, atol=None, rtol=None, hermitian=False):
    """Moore-Penrose pseudoinverse."""
    arr = _to_numpy(a).astype(np.float64)
    if rtol is not None:
        rcond = rtol
    elif atol is not None:
        s_max = np.linalg.svd(arr, compute_uv=False)[..., 0]
        rcond = atol / s_max
    else:
        rcond = 1e-15
    out = np.linalg.pinv(arr, rcond=rcond)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_slogdet(a):
    """Sign and log absolute value of determinant."""
    from collections import namedtuple
    arr = _to_numpy(a).astype(np.float64)
    sign, logabsdet = np.linalg.slogdet(arr)
    SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
    dt = to_numpy_dtype(a.dtype)
    return SlogdetResult(
        _from_numpy(np.ascontiguousarray(np.atleast_1d(sign).astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(np.atleast_1d(logabsdet).astype(dt)), a.dtype, a.device),
    )


def linalg_solve(a, b, left=True):
    """Solve a square system of linear equations."""
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    if not left:
        # X @ A = B => A^T @ X^T = B^T
        out = np.linalg.solve(a_np.T, b_np.T).T
    else:
        out = np.linalg.solve(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_solve_triangular(a, b, upper, left=True, unitriangular=False):
    """Solve a triangular system."""
    from scipy import linalg as scipy_linalg
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    if not left:
        a_np = a_np.T
        b_np = b_np.T
        upper = not upper
    out = scipy_linalg.solve_triangular(a_np, b_np, lower=not upper, unit_diagonal=unitriangular)
    if not left:
        out = out.T
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_svd(a, full_matrices=True):
    """Singular value decomposition."""
    arr = _to_numpy(a).astype(np.float64)
    U, S, Vh = np.linalg.svd(arr, full_matrices=full_matrices)
    dt = to_numpy_dtype(a.dtype)
    return (
        _from_numpy(np.ascontiguousarray(U.astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(S.astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(Vh.astype(dt)), a.dtype, a.device),
    )


def linalg_svdvals(a):
    """Singular values of a matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.svd(arr, compute_uv=False)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_tensorinv(a, ind=2):
    """Tensor inverse (generalization of matrix inverse)."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.tensorinv(arr, ind=ind)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_tensorsolve(a, b, dims=None):
    """Solve a tensor equation."""
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    axes = None if dims is None else tuple(dims)
    out = np.linalg.tensorsolve(a_np, b_np, axes=axes)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def linalg_vander(x, N=None):
    """Vandermonde matrix."""
    arr = _to_numpy(x)
    n = N if N is not None else len(arr)
    out = np.vander(arr, N=n, increasing=True)
    return _from_numpy(np.ascontiguousarray(out), x.dtype, x.device)


def linalg_vector_norm(a, ord=2, dim=None, keepdim=False):
    """Vector norm."""
    arr = _to_numpy(a).astype(np.float64)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            axis = tuple(dim)
        else:
            axis = dim
    else:
        axis = None
    if ord == float('inf'):
        out = np.max(np.abs(arr), axis=axis, keepdims=keepdim)
    elif ord == float('-inf'):
        out = np.min(np.abs(arr), axis=axis, keepdims=keepdim)
    elif ord == 0:
        out = np.sum(arr != 0, axis=axis, keepdims=keepdim).astype(np.float64)
    else:
        out = np.sum(np.abs(arr) ** ord, axis=axis, keepdims=keepdim) ** (1.0 / ord)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


# ---------------------------------------------------------------------------
# torch.fft ops
# ---------------------------------------------------------------------------

def fft_fft(a, n=None, dim=-1, norm=None):
    """1D discrete Fourier Transform."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    out = np.fft.fft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_ifft(a, n=None, dim=-1, norm=None):
    """1D inverse discrete Fourier Transform."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    out = np.fft.ifft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_fft2(a, s=None, dim=(-2, -1), norm=None):
    """2D discrete Fourier Transform."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.fft2(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_ifft2(a, s=None, dim=(-2, -1), norm=None):
    """2D inverse discrete Fourier Transform."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.ifft2(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_fftn(a, s=None, dim=None, norm=None):
    """N-D discrete Fourier Transform."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.fftn(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_ifftn(a, s=None, dim=None, norm=None):
    """N-D inverse discrete Fourier Transform."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.ifftn(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_rfft(a, n=None, dim=-1, norm=None):
    """1D FFT of real-valued input."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    out = np.fft.rfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_irfft(a, n=None, dim=-1, norm=None):
    """Inverse of rfft."""
    arr = _to_numpy(a)
    out = np.fft.irfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def fft_rfft2(a, s=None, dim=(-2, -1), norm=None):
    """2D FFT of real-valued input."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.rfft2(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_irfft2(a, s=None, dim=(-2, -1), norm=None):
    """Inverse of rfft2."""
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.irfft2(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def fft_rfftn(a, s=None, dim=None, norm=None):
    """N-D FFT of real-valued input."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.rfftn(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_irfftn(a, s=None, dim=None, norm=None):
    """Inverse of rfftn."""
    arr = _to_numpy(a)
    axes = tuple(dim) if dim is not None else None
    out = np.fft.irfftn(arr, s=s, axes=axes, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def fft_hfft(a, n=None, dim=-1, norm=None):
    """1D FFT of Hermitian symmetric signal (output is real)."""
    arr = _to_numpy(a)
    out = np.fft.hfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def fft_ihfft(a, n=None, dim=-1, norm=None):
    """Inverse of hfft."""
    from ..._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a)
    out = np.fft.ihfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)


def fft_fftshift(a, dim=None):
    """Shift zero-frequency component to center."""
    arr = _to_numpy(a)
    axes = None if dim is None else (tuple(dim) if isinstance(dim, (list, tuple)) else (dim,))
    out = np.fft.fftshift(arr, axes=axes)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def fft_ifftshift(a, dim=None):
    """Inverse of fftshift."""
    arr = _to_numpy(a)
    axes = None if dim is None else (tuple(dim) if isinstance(dim, (list, tuple)) else (dim,))
    out = np.fft.ifftshift(arr, axes=axes)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


# ---------------------------------------------------------------------------
# torch.special ops
# ---------------------------------------------------------------------------

def special_digamma(a):
    """Logarithmic derivative of the gamma function (psi function)."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.digamma(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_entr(a):
    """Entropy: -x * ln(x), 0 for x=0, -inf for x<0."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.where(arr > 0, -arr * np.log(arr), np.where(arr == 0, 0.0, -np.inf))
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_erfcx(a):
    """Scaled complementary error function: exp(x^2) * erfc(x)."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.erfcx(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_erfinv(a):
    """Inverse error function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.erfinv(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_gammainc(a, b):
    """Regularized lower incomplete gamma function."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.gammainc(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_gammaincc(a, b):
    """Regularized upper incomplete gamma function."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.gammaincc(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_gammaln(a):
    """Log of the absolute value of the gamma function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.gammaln(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_i0(a):
    """Zeroth order modified Bessel function of the first kind."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.i0(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_i0e(a):
    """Exponentially scaled zeroth order modified Bessel function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.i0e(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_i1(a):
    """First order modified Bessel function of the first kind."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.i1(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_i1e(a):
    """Exponentially scaled first order modified Bessel function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.i1e(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_log_ndtr(a):
    """Log of the area under the standard Gaussian PDF from -inf to x."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.log_ndtr(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_logit(a, eps=None):
    """Logit function: log(x / (1 - x))."""
    arr = _to_numpy(a).astype(np.float64)
    if eps is not None:
        arr = np.clip(arr, eps, 1.0 - eps)
    from scipy import special as sp
    out = sp.logit(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_multigammaln(a, p):
    """Multivariate log-gamma function with dimension p."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.multigammaln(arr, p)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_ndtr(a):
    """Area under the standard Gaussian PDF from -inf to x (normal CDF)."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.ndtr(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_ndtri(a):
    """Inverse of ndtr (quantile function of standard normal)."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.ndtri(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_polygamma(n, a):
    """N-th derivative of the digamma function."""
    from scipy import special as sp
    arr = _to_numpy(a).astype(np.float64)
    out = sp.polygamma(n, arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_sinc(a):
    """Normalized sinc function: sin(pi*x) / (pi*x)."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.sinc(arr)  # np.sinc already computes sin(pi*x)/(pi*x)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_xlog1py(a, b):
    """x * log1p(y), with 0 when x=0."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.xlog1py(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_xlogy(a, b):
    """x * log(y), with 0 when x=0."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.xlogy(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


def special_zeta(a, b):
    """Hurwitz zeta function."""
    from scipy import special as sp
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    out = sp.zeta(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


# ---------------------------------------------------------------------------
# F.affine_grid / F.grid_sample
# ---------------------------------------------------------------------------

def affine_grid(theta, size, align_corners=None):
    """Generate 2D or 3D affine sampling grid from transformation matrix.

    Args:
        theta: Tensor of shape (N, 2, 3) for 4-D or (N, 3, 4) for 5-D.
        size: output spatial size, e.g. torch.Size([N, C, H, W]).
        align_corners: if True, [-1, 1] maps to pixel centres at corners;
                       if False, [-1, 1] maps to the edges of the corner pixels.
    Returns:
        grid: (N, H, W, 2) for 4-D or (N, D, H, W, 3) for 5-D.
    """
    if align_corners is None:
        align_corners = False

    theta_np = _to_numpy(theta).astype(np.float32)  # (N, 2, 3) or (N, 3, 4)
    N = theta_np.shape[0]

    if len(size) == 4:
        # 2-D spatial
        _, _, H, W = size
        # Build base grid of (x, y, 1) coordinates
        if align_corners:
            # linspace from -1 to 1 with H / W points
            ys = np.linspace(-1.0, 1.0, H, dtype=np.float32) if H > 1 else np.array([0.0], dtype=np.float32)
            xs = np.linspace(-1.0, 1.0, W, dtype=np.float32) if W > 1 else np.array([0.0], dtype=np.float32)
        else:
            # coordinates centred in each cell, ranging from -1+1/W to 1-1/W
            ys = (np.arange(H, dtype=np.float32) * 2.0 + 1.0) / H - 1.0 if H > 0 else np.zeros(0, dtype=np.float32)
            xs = (np.arange(W, dtype=np.float32) * 2.0 + 1.0) / W - 1.0 if W > 0 else np.zeros(0, dtype=np.float32)

        # meshgrid: grid_x (H, W), grid_y (H, W)
        grid_x, grid_y = np.meshgrid(xs, ys)  # both (H, W)
        ones = np.ones_like(grid_x)

        # base_grid: (H*W, 3) — columns are x, y, 1
        base_grid = np.stack([grid_x.ravel(), grid_y.ravel(), ones.ravel()], axis=1)  # (H*W, 3)

        # Apply theta: out = base_grid @ theta^T  =>  (N, H*W, 2)
        # theta: (N, 2, 3), base_grid: (H*W, 3)
        # result[n] = base_grid @ theta[n].T  ->  (H*W, 2)
        out = np.einsum('ij,nkj->nik', base_grid, theta_np)  # (N, H*W, 2)
        out = out.reshape(N, H, W, 2)

    elif len(size) == 5:
        # 3-D spatial
        _, _, D, H, W = size
        if align_corners:
            zs = np.linspace(-1.0, 1.0, D, dtype=np.float32) if D > 1 else np.array([0.0], dtype=np.float32)
            ys = np.linspace(-1.0, 1.0, H, dtype=np.float32) if H > 1 else np.array([0.0], dtype=np.float32)
            xs = np.linspace(-1.0, 1.0, W, dtype=np.float32) if W > 1 else np.array([0.0], dtype=np.float32)
        else:
            zs = (np.arange(D, dtype=np.float32) * 2.0 + 1.0) / D - 1.0 if D > 0 else np.zeros(0, dtype=np.float32)
            ys = (np.arange(H, dtype=np.float32) * 2.0 + 1.0) / H - 1.0 if H > 0 else np.zeros(0, dtype=np.float32)
            xs = (np.arange(W, dtype=np.float32) * 2.0 + 1.0) / W - 1.0 if W > 0 else np.zeros(0, dtype=np.float32)

        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='xy')
        # grid_x, grid_y, grid_z are (H, W, D) due to meshgrid 'xy' indexing
        # Transpose to (D, H, W)
        grid_x = np.transpose(grid_x, (2, 0, 1))
        grid_y = np.transpose(grid_y, (2, 0, 1))
        grid_z = np.transpose(grid_z, (2, 0, 1))
        ones = np.ones_like(grid_x)

        base_grid = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel(), ones.ravel()], axis=1)  # (D*H*W, 4)
        out = np.einsum('ij,nkj->nik', base_grid, theta_np)  # (N, D*H*W, 3)
        out = out.reshape(N, D, H, W, 3)
    else:
        raise ValueError("affine_grid only supports 4-D (2-D spatial) and 5-D (3-D spatial) inputs")

    return _from_numpy(np.ascontiguousarray(out), theta.dtype, theta.device)


def _grid_sample_denormalize(coord, length, align_corners):
    """Convert normalised coordinate [-1, 1] to pixel coordinate."""
    if align_corners:
        # -1 -> 0, 1 -> length-1
        return ((coord + 1.0) / 2.0) * (length - 1)
    else:
        # -1 -> -0.5, 1 -> length - 0.5
        return ((coord + 1.0) * length - 1.0) / 2.0


def _grid_sample_compute_source(coord, length, padding_mode):
    """Apply padding mode to source pixel coordinate.

    Returns (index_array, in_bounds_mask).
    For 'zeros': index is clamped but mask marks OOB.
    For 'border': index is clamped.
    For 'reflection': index is reflected.
    """
    if padding_mode == 'border':
        coord = np.clip(coord, 0, length - 1)
        return coord, None
    elif padding_mode == 'reflection':
        # Reflect: the range [0, length-1] is mirrored.
        # Double the period and fold back.
        if length > 1:
            # Period = 2 * (length - 1)
            period = 2.0 * (length - 1)
            coord = np.abs(np.mod(coord, period))
            coord = np.where(coord > length - 1, period - coord, coord)
        else:
            coord = np.zeros_like(coord)
        coord = np.clip(coord, 0, length - 1)
        return coord, None
    else:
        # zeros padding: mark out-of-bounds, clamp index
        mask = (coord >= 0) & (coord < length)
        coord = np.clip(coord, 0, length - 1)
        return coord, mask


def _cubic_interp_weight(t):
    """Compute cubic convolution weights for distance t (|t| values).

    Uses the Keys cubic with a = -0.75 (same as PyTorch).
    Returns array of same shape as t.
    """
    t = np.abs(t)
    w = np.where(
        t <= 1.0,
        (1.5 * t - 2.5) * t * t + 1.0,
        np.where(
            t < 2.0,
            ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0,
            0.0,
        ),
    )
    return w


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    """Sample input tensor at grid coordinates.

    Args:
        input: (N, C, H_in, W_in) tensor.
        grid:  (N, H_out, W_out, 2) tensor with values in [-1, 1].
               The last dimension is (x, y) where x indexes W and y indexes H.
        mode: 'bilinear', 'nearest', or 'bicubic'.
        padding_mode: 'zeros', 'border', or 'reflection'.
        align_corners: bool.
    Returns:
        (N, C, H_out, W_out) tensor.
    """
    if align_corners is None:
        align_corners = False

    inp = _to_numpy(input).astype(np.float64)  # (N, C, H_in, W_in)
    g = _to_numpy(grid).astype(np.float64)      # (N, H_out, W_out, 2)

    N, C, H_in, W_in = inp.shape
    _, H_out, W_out, _ = g.shape

    # De-normalise grid coordinates
    gx = _grid_sample_denormalize(g[..., 0], W_in, align_corners)  # (N, H_out, W_out)
    gy = _grid_sample_denormalize(g[..., 1], H_in, align_corners)  # (N, H_out, W_out)

    if mode == 'nearest':
        ix = np.round(gx).astype(np.int64)
        iy = np.round(gy).astype(np.int64)

        ix_src, mask_x = _grid_sample_compute_source(ix.astype(np.float64), W_in, padding_mode)
        iy_src, mask_y = _grid_sample_compute_source(iy.astype(np.float64), H_in, padding_mode)
        ix_src = ix_src.astype(np.int64)
        iy_src = iy_src.astype(np.int64)

        # Gather: out[n, c, h, w] = inp[n, c, iy_src[n,h,w], ix_src[n,h,w]]
        n_idx = np.arange(N)[:, None, None]  # (N, 1, 1)
        out = inp[n_idx, :, iy_src[:, :, :], ix_src[:, :, :]]  # (N, H_out, W_out, C)
        out = out.transpose(0, 3, 1, 2)  # (N, C, H_out, W_out)

        if padding_mode == 'zeros':
            mask = mask_x & mask_y  # (N, H_out, W_out)
            out = out * mask[:, np.newaxis, :, :]

    elif mode == 'bilinear':
        ix0 = np.floor(gx).astype(np.int64)
        iy0 = np.floor(gy).astype(np.int64)

        tx = gx - np.floor(gx)  # fractional part
        ty = gy - np.floor(gy)

        out = np.zeros((N, C, H_out, W_out), dtype=np.float64)

        for dy in range(2):
            for dx in range(2):
                cy = iy0 + dy
                cx = ix0 + dx

                cx_src, mask_x = _grid_sample_compute_source(cx.astype(np.float64), W_in, padding_mode)
                cy_src, mask_y = _grid_sample_compute_source(cy.astype(np.float64), H_in, padding_mode)
                cx_src = cx_src.astype(np.int64)
                cy_src = cy_src.astype(np.int64)

                wx = (1.0 - tx) if dx == 0 else tx
                wy = (1.0 - ty) if dy == 0 else ty
                w = wx * wy  # (N, H_out, W_out)

                n_idx = np.arange(N)[:, None, None]
                val = inp[n_idx, :, cy_src, cx_src]  # (N, H_out, W_out, C)
                val = val.transpose(0, 3, 1, 2)       # (N, C, H_out, W_out)

                if padding_mode == 'zeros':
                    mask = mask_x & mask_y
                    val = val * mask[:, np.newaxis, :, :]

                out += val * w[:, np.newaxis, :, :]

    elif mode == 'bicubic':
        ix0 = np.floor(gx).astype(np.int64)
        iy0 = np.floor(gy).astype(np.int64)

        tx = gx - np.floor(gx)
        ty = gy - np.floor(gy)

        out = np.zeros((N, C, H_out, W_out), dtype=np.float64)

        for dy in range(-1, 3):
            for dx in range(-1, 3):
                cy = iy0 + dy
                cx = ix0 + dx

                cx_src, mask_x = _grid_sample_compute_source(cx.astype(np.float64), W_in, padding_mode)
                cy_src, mask_y = _grid_sample_compute_source(cy.astype(np.float64), H_in, padding_mode)
                cx_src = cx_src.astype(np.int64)
                cy_src = cy_src.astype(np.int64)

                wx = _cubic_interp_weight(tx - dx)
                wy = _cubic_interp_weight(ty - dy)
                w = wx * wy

                n_idx = np.arange(N)[:, None, None]
                val = inp[n_idx, :, cy_src, cx_src]
                val = val.transpose(0, 3, 1, 2)

                if padding_mode == 'zeros':
                    mask = mask_x & mask_y
                    val = val * mask[:, np.newaxis, :, :]

                out += val * w[:, np.newaxis, :, :]

    else:
        raise ValueError(f"grid_sample mode must be 'bilinear', 'nearest', or 'bicubic', got '{mode}'")

    out = out.astype(to_numpy_dtype(input.dtype))
    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)


# ---------------------------------------------------------------------------
# F.unfold (im2col) / F.fold (col2im)
# ---------------------------------------------------------------------------

def im2col(a, kernel_size, dilation, padding, stride):
    """F.unfold: Extract sliding local blocks from 4D input (N, C, H, W).

    Returns tensor of shape (N, C*kH*kW, L) where L = number of valid blocks.
    kernel_size, dilation, padding, stride are all 2-tuples.
    """
    arr = _to_numpy(a)
    N, C, H, W = arr.shape
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    sH, sW = stride

    # Effective kernel size with dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    # Output spatial dimensions
    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1
    L = H_out * W_out

    # Pad input if needed
    if pH > 0 or pW > 0:
        arr = np.pad(arr, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')

    # Ensure contiguous for stride tricks
    arr = np.ascontiguousarray(arr)

    # Use stride tricks to extract patches efficiently
    # Shape: (N, C, kH, kW, H_out, W_out)
    shape = (N, C, kH, kW, H_out, W_out)
    strides = (
        arr.strides[0],          # batch
        arr.strides[1],          # channel
        arr.strides[2] * dH,     # kernel height (dilated)
        arr.strides[3] * dW,     # kernel width (dilated)
        arr.strides[2] * sH,     # output height (stride)
        arr.strides[3] * sW,     # output width (stride)
    )

    patches = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    # Reshape to (N, C*kH*kW, H_out*W_out)
    out = patches.reshape(N, C * kH * kW, L)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def col2im(a, output_size, kernel_size, dilation, padding, stride):
    """F.fold: Combine array of sliding local blocks into a 4D tensor.

    Input shape: (N, C*kH*kW, L)
    Returns tensor of shape (N, C, output_size[0], output_size[1]).
    Overlapping regions are summed (matching PyTorch behavior).
    """
    arr = _to_numpy(a)
    N, C_kk, L = arr.shape
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    sH, sW = stride
    H_out, W_out = output_size

    # Effective kernel size with dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    # Number of output positions
    H_col = (H_out + 2 * pH - ekH) // sH + 1
    W_col = (W_out + 2 * pW - ekW) // sW + 1

    if H_col * W_col != L:
        raise ValueError(
            f"Expected L={H_col * W_col} (H_col={H_col}, W_col={W_col}) but got L={L}"
        )

    C = C_kk // (kH * kW)
    if C * kH * kW != C_kk:
        raise ValueError(
            f"C*kH*kW ({C}*{kH}*{kW}={C * kH * kW}) does not match input dim 1 ({C_kk})"
        )

    # Padded output dimensions
    H_pad = H_out + 2 * pH
    W_pad = W_out + 2 * pW

    # Create output with padding
    out = np.zeros((N, C, H_pad, W_pad), dtype=arr.dtype)

    # Reshape input columns to (N, C, kH, kW, H_col, W_col)
    cols = arr.reshape(N, C, kH, kW, H_col, W_col)

    # Scatter-add patches back into output
    for i in range(kH):
        for j in range(kW):
            h_start = i * dH
            w_start = j * dW
            for h_idx in range(H_col):
                for w_idx in range(W_col):
                    out[:, :, h_start + h_idx * sH, w_start + w_idx * sW] += cols[:, :, i, j, h_idx, w_idx]

    # Remove padding
    if pH > 0 or pW > 0:
        out = out[:, :, pH:pH + H_out, pW:pW + W_out]

    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


# ---------------------------------------------------------------------------
# uniform (out-of-place) — needed by F.gumbel_softmax
# ---------------------------------------------------------------------------
def uniform(a):
    """Return tensor of same shape filled with Uniform(0,1) samples."""
    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = rng.uniform(0.0, 1.0, _to_numpy(a).shape).astype(_to_numpy(a).dtype)
    return _from_numpy(arr, a.dtype, a.device)


# ---------------------------------------------------------------------------
# Upsample ops — CPU numpy implementations
# ---------------------------------------------------------------------------
def upsample_nearest1d(a, output_size):
    """Nearest-neighbor 1D upsampling. Input: (N, C, W_in) -> (N, C, W_out)."""
    arr = _to_numpy(a)
    W_out = output_size[0]
    W_in = arr.shape[2]
    indices = (np.arange(W_out, dtype=np.float64) * W_in / W_out).astype(np.intp)
    np.clip(indices, 0, W_in - 1, out=indices)
    out = arr[:, :, indices]
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def upsample_linear1d(a, output_size, align_corners=False, scales=None):
    """Linear interpolation 1D upsampling. Input: (N, C, W_in) -> (N, C, W_out)."""
    arr = _to_numpy(a).astype(np.float64)
    W_out = output_size[0]
    W_in = arr.shape[2]

    if W_out == 1:
        out = arr[:, :, :1]
        return _from_numpy(np.ascontiguousarray(out.astype(_to_numpy(a).dtype)), a.dtype, a.device)

    if align_corners and W_in > 1 and W_out > 1:
        x = np.linspace(0, W_in - 1, W_out)
    else:
        x = (np.arange(W_out, dtype=np.float64) + 0.5) * W_in / W_out - 0.5

    x = np.clip(x, 0, W_in - 1)
    x0 = np.floor(x).astype(np.intp)
    x1 = np.minimum(x0 + 1, W_in - 1)
    wx = (x - x0).reshape(1, 1, -1)

    out = arr[:, :, x0] * (1.0 - wx) + arr[:, :, x1] * wx
    return _from_numpy(np.ascontiguousarray(out.astype(_to_numpy(a).dtype)), a.dtype, a.device)


def upsample_nearest2d(a, output_size):
    """Nearest-neighbor 2D upsampling. Input: (N, C, H_in, W_in) -> (N, C, H_out, W_out)."""
    arr = _to_numpy(a)
    H_out, W_out = output_size
    H_in, W_in = arr.shape[2], arr.shape[3]
    h_idx = (np.arange(H_out, dtype=np.float64) * H_in / H_out).astype(np.intp)
    w_idx = (np.arange(W_out, dtype=np.float64) * W_in / W_out).astype(np.intp)
    np.clip(h_idx, 0, H_in - 1, out=h_idx)
    np.clip(w_idx, 0, W_in - 1, out=w_idx)
    out = arr[:, :, h_idx[:, None], w_idx[None, :]]
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def upsample_bilinear2d(a, output_size, align_corners=False, scales_h=None, scales_w=None):
    """Bilinear 2D upsampling. Input: (N, C, H_in, W_in) -> (N, C, H_out, W_out)."""
    arr = _to_numpy(a).astype(np.float64)
    H_out, W_out = output_size
    H_in, W_in = arr.shape[2], arr.shape[3]

    if align_corners and H_in > 1 and H_out > 1:
        h = np.linspace(0, H_in - 1, H_out)
    else:
        h = (np.arange(H_out, dtype=np.float64) + 0.5) * H_in / H_out - 0.5
    if align_corners and W_in > 1 and W_out > 1:
        w = np.linspace(0, W_in - 1, W_out)
    else:
        w = (np.arange(W_out, dtype=np.float64) + 0.5) * W_in / W_out - 0.5

    h = np.clip(h, 0, H_in - 1)
    w = np.clip(w, 0, W_in - 1)

    h0 = np.floor(h).astype(np.intp)
    h1 = np.minimum(h0 + 1, H_in - 1)
    w0 = np.floor(w).astype(np.intp)
    w1 = np.minimum(w0 + 1, W_in - 1)

    wh = (h - h0).reshape(1, 1, -1, 1)
    ww = (w - w0).reshape(1, 1, 1, -1)

    out = (arr[:, :, h0[:, None], w0[None, :]] * (1 - wh) * (1 - ww) +
           arr[:, :, h0[:, None], w1[None, :]] * (1 - wh) * ww +
           arr[:, :, h1[:, None], w0[None, :]] * wh * (1 - ww) +
           arr[:, :, h1[:, None], w1[None, :]] * wh * ww)
    return _from_numpy(np.ascontiguousarray(out.astype(_to_numpy(a).dtype)), a.dtype, a.device)


def upsample_bicubic2d(a, output_size, align_corners=False, scales_h=None, scales_w=None):
    """Bicubic 2D upsampling. Input: (N, C, H_in, W_in) -> (N, C, H_out, W_out)."""
    arr = _to_numpy(a).astype(np.float64)
    H_out, W_out = output_size
    H_in, W_in = arr.shape[2], arr.shape[3]

    def _cubic_weight(t):
        """Keys cubic kernel (a=-0.75)."""
        at = np.abs(t)
        return np.where(at <= 1, (1.5 * at - 2.5) * at * at + 1,
               np.where(at < 2, ((-0.5 * at + 2.5) * at - 4) * at + 2, 0.0))

    if align_corners and H_in > 1 and H_out > 1:
        h = np.linspace(0, H_in - 1, H_out)
    else:
        h = (np.arange(H_out, dtype=np.float64) + 0.5) * H_in / H_out - 0.5
    if align_corners and W_in > 1 and W_out > 1:
        w = np.linspace(0, W_in - 1, W_out)
    else:
        w = (np.arange(W_out, dtype=np.float64) + 0.5) * W_in / W_out - 0.5

    N, C = arr.shape[:2]
    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    for j in range(H_out):
        for i in range(W_out):
            hy, wx = h[j], w[i]
            hy0 = int(np.floor(hy)) - 1
            wx0 = int(np.floor(wx)) - 1
            for dh in range(4):
                for dw in range(4):
                    hh = min(max(hy0 + dh, 0), H_in - 1)
                    ww = min(max(wx0 + dw, 0), W_in - 1)
                    weight = _cubic_weight(hy - (hy0 + dh)) * _cubic_weight(wx - (wx0 + dw))
                    out[:, :, j, i] += arr[:, :, hh, ww] * weight
    return _from_numpy(np.ascontiguousarray(out.astype(_to_numpy(a).dtype)), a.dtype, a.device)


def max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode=False, return_indices=False):
    """Max pooling 1D via numpy sliding window."""
    a = _to_numpy(input)
    kW = kernel_size[0]
    sW = stride[0]
    pW = padding[0]
    dW = dilation[0]

    if input.ndim == 3:
        N, C, W = a.shape
    else:
        raise ValueError(f"Expected 3D input, got {input.ndim}D")

    # Pad input
    if pW > 0:
        a = np.pad(a, ((0, 0), (0, 0), (pW, pW)), mode='constant', constant_values=-np.inf)
    W_padded = a.shape[2]

    if ceil_mode:
        oW = int(np.ceil((W_padded - dW * (kW - 1) - 1) / sW + 1))
    else:
        oW = int(np.floor((W_padded - dW * (kW - 1) - 1) / sW + 1))

    out = np.empty((N, C, oW), dtype=a.dtype)
    for ow in range(oW):
        w_start = ow * sW
        vals = [a[:, :, w_start + k * dW] for k in range(kW) if w_start + k * dW < W_padded]
        out[:, :, ow] = np.maximum.reduce(vals)

    return _from_numpy(out, input.dtype, input.device)


def avg_pool1d(input, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True):
    """Avg pooling 1D via numpy sliding window."""
    a = _to_numpy(input)
    kW = kernel_size[0]
    sW = stride[0]
    pW = padding[0]

    N, C, W = a.shape

    if pW > 0:
        a = np.pad(a, ((0, 0), (0, 0), (pW, pW)), mode='constant', constant_values=0)
    W_padded = a.shape[2]

    if ceil_mode:
        oW = int(np.ceil((W_padded - kW) / sW + 1))
    else:
        oW = int(np.floor((W_padded - kW) / sW + 1))

    out = np.empty((N, C, oW), dtype=a.dtype)
    for ow in range(oW):
        w_start = ow * sW
        w_end = min(w_start + kW, W_padded)
        window = a[:, :, w_start:w_end]
        if count_include_pad:
            out[:, :, ow] = window.sum(axis=2) / kW
        else:
            real_start = max(w_start - pW, 0)
            real_end = min(w_end - pW, W)
            count = max(real_end - real_start, 1)
            out[:, :, ow] = window.sum(axis=2) / count

    return _from_numpy(out, input.dtype, input.device)


def adaptive_avg_pool1d(input, output_size):
    """Adaptive avg pool 1D: compute kernel and stride from output_size."""
    a = _to_numpy(input)
    N, C, W = a.shape
    oW = output_size[0]

    out = np.empty((N, C, oW), dtype=a.dtype)
    for ow in range(oW):
        w_start = int(np.floor(ow * W / oW))
        w_end = int(np.ceil((ow + 1) * W / oW))
        out[:, :, ow] = a[:, :, w_start:w_end].mean(axis=2)

    return _from_numpy(out, input.dtype, input.device)


def _conv_transpose3d_scatter(out, a, w, n, c_in, c_out_start, C_out_per_g,
                               D_in, H_in, W_in, D_out, H_out, W_out,
                               kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW):
    """Scatter one input channel's contribution for conv_transpose3d."""
    for d in range(D_in):
        for h in range(H_in):
            for wi in range(W_in):
                val = a[n, c_in, d, h, wi]
                if val == 0:
                    continue
                for c_out_local in range(C_out_per_g):
                    c_out = c_out_start + c_out_local
                    for kd in range(kD):
                        od = d * sD - pD + kd * dD
                        if od < 0 or od >= D_out:
                            continue
                        for kh in range(kH):
                            oh = h * sH - pH + kh * dH
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(kW):
                                ow = wi * sW - pW + kw * dW
                                if ow < 0 or ow >= W_out:
                                    continue
                                out[n, c_out, od, oh, ow] += val * w[c_in, c_out_local, kd, kh, kw]


def conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation):
    """Transposed convolution 3D via numpy."""
    a = _to_numpy(input)
    w = _to_numpy(weight)
    sD, sH, sW = stride
    pD, pH, pW = padding
    opD, opH, opW = output_padding
    dD, dH, dW = dilation

    N, C_in, D_in, H_in, W_in = a.shape
    C_in_w, C_out_per_g, kD, kH, kW = w.shape
    C_out = C_out_per_g * groups

    D_out = (D_in - 1) * sD - 2 * pD + dD * (kD - 1) + opD + 1
    H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
    W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1

    out = np.zeros((N, C_out, D_out, H_out, W_out), dtype=a.dtype)
    c_in_per_g = C_in // groups

    for g in range(groups):
        for n in range(N):
            for c_in_local in range(c_in_per_g):
                c_in = g * c_in_per_g + c_in_local
                _conv_transpose3d_scatter(out, a, w, n, c_in, g * C_out_per_g, C_out_per_g,
                                          D_in, H_in, W_in, D_out, H_out, W_out,
                                          kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW)

    if bias is not None:
        b = _to_numpy(bias)
        out += b.reshape(1, C_out, 1, 1, 1)

    return _from_numpy(out, input.dtype, input.device)


def max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode=False, return_indices=False):
    """Max pooling 3D via numpy sliding window."""
    a = _to_numpy(input)
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    N, C, D, H, W = a.shape

    if pD > 0 or pH > 0 or pW > 0:
        a = np.pad(a, ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)),
                   mode='constant', constant_values=-np.inf)

    D_pad, H_pad, W_pad = a.shape[2], a.shape[3], a.shape[4]

    if ceil_mode:
        oD = int(np.ceil((D_pad - dD * (kD - 1) - 1) / sD + 1))
        oH = int(np.ceil((H_pad - dH * (kH - 1) - 1) / sH + 1))
        oW = int(np.ceil((W_pad - dW * (kW - 1) - 1) / sW + 1))
    else:
        oD = int(np.floor((D_pad - dD * (kD - 1) - 1) / sD + 1))
        oH = int(np.floor((H_pad - dH * (kH - 1) - 1) / sH + 1))
        oW = int(np.floor((W_pad - dW * (kW - 1) - 1) / sW + 1))

    out = np.empty((N, C, oD, oH, oW), dtype=a.dtype)
    for od in range(oD):
        for oh in range(oH):
            for ow in range(oW):
                d_start = od * sD
                h_start = oh * sH
                w_start = ow * sW
                window_vals = []
                for kd in range(kD):
                    di = d_start + kd * dD
                    if di >= D_pad:
                        continue
                    for kh in range(kH):
                        hi = h_start + kh * dH
                        if hi >= H_pad:
                            continue
                        for kw in range(kW):
                            wi = w_start + kw * dW
                            if wi >= W_pad:
                                continue
                            window_vals.append(a[:, :, di, hi, wi])
                if window_vals:
                    out[:, :, od, oh, ow] = np.maximum.reduce(window_vals)

    return _from_numpy(out, input.dtype, input.device)


def avg_pool3d(input, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True):
    """Avg pooling 3D via numpy sliding window."""
    a = _to_numpy(input)
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding

    N, C, D, H, W = a.shape

    if pD > 0 or pH > 0 or pW > 0:
        a = np.pad(a, ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)),
                   mode='constant', constant_values=0)

    D_pad, H_pad, W_pad = a.shape[2], a.shape[3], a.shape[4]

    if ceil_mode:
        oD = int(np.ceil((D_pad - kD) / sD + 1))
        oH = int(np.ceil((H_pad - kH) / sH + 1))
        oW = int(np.ceil((W_pad - kW) / sW + 1))
    else:
        oD = int(np.floor((D_pad - kD) / sD + 1))
        oH = int(np.floor((H_pad - kH) / sH + 1))
        oW = int(np.floor((W_pad - kW) / sW + 1))

    out = np.empty((N, C, oD, oH, oW), dtype=a.dtype)
    pool_size = kD * kH * kW
    for od in range(oD):
        for oh in range(oH):
            for ow in range(oW):
                d_start = od * sD
                h_start = oh * sH
                w_start = ow * sW
                d_end = min(d_start + kD, D_pad)
                h_end = min(h_start + kH, H_pad)
                w_end = min(w_start + kW, W_pad)
                window = a[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                if count_include_pad:
                    out[:, :, od, oh, ow] = window.sum(axis=(2, 3, 4)) / pool_size
                else:
                    count = window.shape[2] * window.shape[3] * window.shape[4]
                    out[:, :, od, oh, ow] = window.sum(axis=(2, 3, 4)) / max(count, 1)

    return _from_numpy(out, input.dtype, input.device)


def adaptive_avg_pool3d(input, output_size):
    """Adaptive avg pool 3D: compute regions from output_size."""
    a = _to_numpy(input)
    N, C, D, H, W = a.shape
    oD, oH, oW = output_size

    out = np.empty((N, C, oD, oH, oW), dtype=a.dtype)
    for od in range(oD):
        d_start = int(np.floor(od * D / oD))
        d_end = int(np.ceil((od + 1) * D / oD))
        for oh in range(oH):
            h_start = int(np.floor(oh * H / oH))
            h_end = int(np.ceil((oh + 1) * H / oH))
            for ow in range(oW):
                w_start = int(np.floor(ow * W / oW))
                w_end = int(np.ceil((ow + 1) * W / oW))
                out[:, :, od, oh, ow] = a[:, :, d_start:d_end, h_start:h_end, w_start:w_end].mean(axis=(2, 3, 4))

    return _from_numpy(out, input.dtype, input.device)


def addmm(input, mat1, mat2, beta=1, alpha=1):
    """addmm: beta * input + alpha * (mat1 @ mat2)."""
    inp = _to_numpy(input)
    m1 = np.ascontiguousarray(_to_numpy(mat1))
    m2 = np.ascontiguousarray(_to_numpy(mat2))
    if _can_use_blas(m1) and _can_use_blas(m2):
        M, K = m1.shape
        N = m2.shape[1]
        # out = alpha * m1 @ m2
        out = np.empty((M, N), dtype=m1.dtype)
        if m1.dtype == np.float32:
            # Prepare C = beta * input first, then sgemm adds alpha*A*B
            out[:] = (np.broadcast_to(inp, (M, N)) * beta).astype(np.float32)
            _accel.cblas_sgemm(111, 111, M, N, K, float(alpha),
                               m1.ctypes.data, K, m2.ctypes.data, N,
                               1.0, out.ctypes.data, N)
        else:
            out[:] = (np.broadcast_to(inp, (M, N)) * beta).astype(np.float64)
            _accel.cblas_dgemm(111, 111, M, N, K, float(alpha),
                               m1.ctypes.data, K, m2.ctypes.data, N,
                               1.0, out.ctypes.data, N)
        return _from_numpy(out, input.dtype, input.device)
    out = beta * inp + alpha * np.dot(m1, m2)
    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)


def conv3d(input, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """Conv3d forward using numpy. Input: (N,C,D,H,W), Weight: (O,C/g,kD,kH,kW)."""
    inp = _to_numpy(input)
    w = _to_numpy(weight)
    N, C_in, D, H, W = inp.shape
    C_out, C_in_g, kD, kH, kW = w.shape
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    ekD = (kD - 1) * dD + 1
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1

    D_out = (D + 2 * pD - ekD) // sD + 1
    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1

    if pD > 0 or pH > 0 or pW > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)), mode='constant')

    out = np.zeros((N, C_out, D_out, H_out, W_out), dtype=inp.dtype)

    for g in range(groups):
        c_out_per_g = C_out // groups
        c_out_s = g * c_out_per_g
        c_in_s = g * C_in_g
        for co_local in range(c_out_per_g):
            co = c_out_s + co_local
            for ci_local in range(C_in_g):
                ci = c_in_s + ci_local
                kernel = w[co, ci_local]
                for od in range(D_out):
                    for oh in range(H_out):
                        for ow in range(W_out):
                            for kd in range(kD):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        id_ = od * sD + kd * dD
                                        ih = oh * sH + kh * dH
                                        iw = ow * sW + kw * dW
                                        out[:, co, od, oh, ow] += inp[:, ci, id_, ih, iw] * kernel[kd, kh, kw]

    if bias is not None:
        b = _to_numpy(bias)
        out += b.reshape(1, C_out, 1, 1, 1)

    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)


def adaptive_max_pool2d(input, output_size, return_indices=False):
    """Adaptive max pooling 2D via numpy."""
    a = _to_numpy(input)
    N, C, H, W = a.shape
    if isinstance(output_size, int):
        oH = oW = output_size
    else:
        oH, oW = output_size

    out = np.empty((N, C, oH, oW), dtype=a.dtype)
    if return_indices:
        out_indices = np.empty((N, C, oH, oW), dtype=np.int64)
        for oh in range(oH):
            h_start = oh * H // oH
            h_end = (oh + 1) * H // oH
            for ow in range(oW):
                w_start = ow * W // oW
                w_end = (ow + 1) * W // oW
                region = a[:, :, h_start:h_end, w_start:w_end]
                region_flat = region.reshape(N, C, -1)
                out[:, :, oh, ow] = region_flat.max(axis=2)
                local_idx = region_flat.argmax(axis=2)
                rW = w_end - w_start
                local_h = local_idx // rW + h_start
                local_w = local_idx % rW + w_start
                out_indices[:, :, oh, ow] = local_h * W + local_w
        result = _from_numpy(out, input.dtype, input.device)
        return result, _from_numpy(out_indices, int64_dtype, input.device)

    for oh in range(oH):
        h_start = oh * H // oH
        h_end = (oh + 1) * H // oH
        for ow in range(oW):
            w_start = ow * W // oW
            w_end = (ow + 1) * W // oW
            region = a[:, :, h_start:h_end, w_start:w_end]
            region_flat = region.reshape(N, C, -1)
            out[:, :, oh, ow] = region_flat.max(axis=2)

    return _from_numpy(out, input.dtype, input.device)


def adaptive_max_pool1d(input, output_size, return_indices=False):
    """Adaptive max pooling 1D via numpy."""
    a = _to_numpy(input)
    N, C, L = a.shape
    oL = output_size if isinstance(output_size, int) else output_size[0]

    out = np.empty((N, C, oL), dtype=a.dtype)
    if return_indices:
        out_indices = np.empty((N, C, oL), dtype=np.int64)
        for ol in range(oL):
            l_start = ol * L // oL
            l_end = (ol + 1) * L // oL
            region = a[:, :, l_start:l_end]
            out[:, :, ol] = region.max(axis=2)
            out_indices[:, :, ol] = region.argmax(axis=2) + l_start
        result = _from_numpy(out, input.dtype, input.device)
        return result, _from_numpy(out_indices, int64_dtype, input.device)

    for ol in range(oL):
        l_start = ol * L // oL
        l_end = (ol + 1) * L // oL
        region = a[:, :, l_start:l_end]
        out[:, :, ol] = region.max(axis=2)

    return _from_numpy(out, input.dtype, input.device)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    """CTC Loss forward via numpy alpha (forward variable) algorithm.

    Args:
        log_probs: (T, N, C) log probabilities
        targets: (N, S) or (sum(target_lengths),) target sequences
        input_lengths: (N,) lengths of inputs
        target_lengths: (N,) lengths of targets
        blank: blank label index
        reduction: 'none', 'mean', or 'sum'
        zero_infinity: if True, zero out infinite losses
    """
    lp = _to_numpy(log_probs).astype(np.float64)
    T, N, C = lp.shape

    if isinstance(targets, Tensor):
        tgt = _to_numpy(targets)
    else:
        tgt = np.array(targets)

    if isinstance(input_lengths, Tensor):
        inp_lens = _to_numpy(input_lengths).astype(np.int64)
    else:
        inp_lens = np.array(input_lengths, dtype=np.int64)

    if isinstance(target_lengths, Tensor):
        tgt_lens = _to_numpy(target_lengths).astype(np.int64)
    else:
        tgt_lens = np.array(target_lengths, dtype=np.int64)

    NEG_INF = -1e30
    losses = np.zeros(N, dtype=np.float64)

    # Determine if targets is 1D (concatenated) or 2D
    is_1d = (tgt.ndim == 1)
    offset = 0

    for b in range(N):
        T_b = int(inp_lens[b])
        S_b = int(tgt_lens[b])

        if is_1d:
            labels_b = tgt[offset:offset + S_b]
            offset += S_b
        else:
            labels_b = tgt[b, :S_b]

        # Build extended labels with blanks: [blank, l0, blank, l1, blank, ...]
        L = 2 * S_b + 1
        ext = np.full(L, blank, dtype=np.int64)
        for s in range(S_b):
            ext[2 * s + 1] = labels_b[s]

        # Alpha (forward variables): alpha[t, s] = log-prob of emitting ext[:s+1] up to time t
        alpha = np.full((T_b, L), NEG_INF, dtype=np.float64)
        alpha[0, 0] = lp[0, b, ext[0]]
        if L > 1:
            alpha[0, 1] = lp[0, b, ext[1]]

        for t in range(1, T_b):
            for s in range(L):
                a = alpha[t - 1, s]
                if s > 0:
                    a = np.logaddexp(a, alpha[t - 1, s - 1])
                if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                    a = np.logaddexp(a, alpha[t - 1, s - 2])
                alpha[t, s] = a + lp[t, b, ext[s]]

        # Loss = -log(alpha[T-1, L-1] + alpha[T-1, L-2])
        log_likelihood = alpha[T_b - 1, L - 1]
        if L > 1:
            log_likelihood = np.logaddexp(log_likelihood, alpha[T_b - 1, L - 2])
        loss = -log_likelihood

        if zero_infinity and np.isinf(loss):
            loss = 0.0
        losses[b] = loss

    if reduction == 'none':
        result = losses
    elif reduction == 'sum':
        result = np.array(losses.sum(), dtype=np.float64)
    else:  # 'mean'
        tgt_lens_f = tgt_lens.astype(np.float64)
        tgt_lens_f = np.maximum(tgt_lens_f, 1.0)
        result = np.array((losses / tgt_lens_f).mean(), dtype=np.float64)

    out_dtype = log_probs.dtype
    return _from_numpy(result.astype(to_numpy_dtype(out_dtype)), out_dtype, log_probs.device)

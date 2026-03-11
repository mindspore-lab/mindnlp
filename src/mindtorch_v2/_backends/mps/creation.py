import numpy as np

from ..._dtype import to_numpy_dtype
from ..._storage import mps_typed_storage_from_numpy
from ..._tensor import Tensor


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def tensor_create(data, dtype=None, device=None, requires_grad=False, memory_format=None):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride, requires_grad=requires_grad)


def zeros_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    arr = np.zeros(shape, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def ones_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    arr = np.ones(shape, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def empty_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    arr = np.empty(shape, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def full_create(shape, fill_value, dtype=None, device=None):
    shape = tuple(shape)
    arr = np.full(shape, fill_value, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)


def arange_create(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def linspace_create(start, end, steps, dtype=None, device=None):
    arr = np.linspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def logspace_create(start, end, steps, dtype=None, device=None):
    arr = np.logspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def eye_create(n, m=None, dtype=None, device=None, out=None):
    if m is None:
        m = n
    arr = np.eye(n, m, dtype=to_numpy_dtype(dtype))
    if out is not None:
        out_arr = out._numpy_view()
        out_arr[:] = arr.astype(out_arr.dtype)
        return out
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def range_create(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end + step, step, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def randn_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.randn(*shape).astype(to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def rand_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.random_sample(shape).astype(to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def randint_create(low, high=None, size=None, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    from ..._dtype import int64 as int64_dtype
    if high is None:
        low, high = 0, low
    if size is None:
        raise ValueError("size is required for randint")
    if isinstance(size, int):
        size = (size,)
    size = tuple(size)
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.randint(int(low), int(high), size=size).astype(np.int64)
    out_dtype = dtype if dtype is not None else int64_dtype
    storage = mps_typed_storage_from_numpy(arr.ravel(), out_dtype, device=device)
    stride = _contiguous_stride(size)
    return Tensor(storage, size, stride, requires_grad=requires_grad)


def randperm_create(n, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    from ..._dtype import int64 as int64_dtype
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.permutation(int(n)).astype(np.int64)
    out_dtype = dtype if dtype is not None else int64_dtype
    storage = mps_typed_storage_from_numpy(arr.ravel(), out_dtype, device=device)
    stride = _contiguous_stride(arr.shape)
    return Tensor(storage, arr.shape, stride, requires_grad=requires_grad)

import numpy as np

from ..._dtype import float32 as float32_dtype
from ..._dtype import float64 as float64_dtype
from ..._dtype import to_numpy_dtype
from ..._storage import npu_typed_storage_from_ptr
from . import runtime as npu_runtime
from . import aclnn
from . import state as npu_state


def _wrap_tensor(storage, shape, stride, requires_grad):
    from ..._tensor import Tensor

    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def _require_inplace_one_zero():
    if not aclnn.ones_zero_symbols_ok():
        raise RuntimeError("aclnnInplaceOne/Zero not available")


def tensor_create(data, dtype=None, device=None, requires_grad=False, memory_format=None):
    arr = np.array(data, dtype=npu_runtime._dtype_to_numpy(dtype))
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, arr.size, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return _wrap_tensor(storage, arr.shape, stride, requires_grad)


def zeros_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    _require_inplace_one_zero()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)
    aclnn.inplace_zero(ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def ones_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    _require_inplace_one_zero()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)
    aclnn.inplace_one(ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def empty_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    stride = npu_runtime._contiguous_stride(shape)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def randn_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create a tensor filled with random numbers from a normal distribution.

    Generates on CPU using the shared RNG state and copies to NPU.
    """
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)

    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = rng.randn(*shape).astype(npu_runtime._dtype_to_numpy(dtype))
    size = int(np.prod(shape))
    ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)

    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def rand_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create a tensor filled with random numbers from a uniform distribution [0, 1).

    Generates on CPU using the shared RNG state and copies to NPU.
    """
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)

    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = rng.random_sample(shape).astype(npu_runtime._dtype_to_numpy(dtype))
    size = int(np.prod(shape))
    ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)

    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def _resolve_dtype(dtype):
    return float32_dtype if dtype is None else dtype


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _device_index(device):
    return (device.index if hasattr(device, "index") else None) or 0


def _is_310b_profile():
    return npu_runtime.soc_profile() == "310b"


def _linspace_fallback_npu(start, end, steps, dtype, device):
    from . import ops as npu_ops

    steps = int(steps)
    shape = (steps,)
    if steps == 0:
        return empty_create(shape, dtype=dtype, device=device, requires_grad=False)
    if steps == 1:
        return full_create(shape, start, dtype=dtype, device=device)

    # Build on NPU without calling aclnn.arange (keeps linspace single-op test semantics).
    one = ones_create(shape, dtype=dtype, device=device)
    idx = npu_ops.cumsum(one, dim=0)
    idx = npu_ops.sub(idx, 1)
    step = (float(end) - float(start)) / float(steps - 1)
    idx = npu_ops.mul(idx, step)
    return npu_ops.add(idx, float(start))


def arange_create(start, end, step=1, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if step == 0:
        raise ValueError("step must be nonzero")
    if not aclnn.arange_symbols_ok():
        raise RuntimeError("aclnnArange not available")

    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    shape = tuple(arr.shape)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    size = int(arr.size)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.arange(start, end, step, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def full_create(shape, fill_value, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    base = zeros_create(shape, dtype=dtype, device=device)
    from . import ops as npu_ops

    return npu_ops.add(base, fill_value)


def linspace_create(start, end, steps, dtype=None, device=None):
    if steps < 0:
        raise ValueError("number of steps must be non-negative")
    dtype = _resolve_dtype(dtype)

    if _is_310b_profile():
        return _linspace_fallback_npu(start, end, steps, dtype=dtype, device=device)

    if not aclnn.linspace_symbols_ok():
        raise RuntimeError("aclnnLinspace not available")

    shape = (int(steps),)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    size = int(steps)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.linspace(start, end, steps, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def logspace_create(start, end, steps, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    linear = linspace_create(start, end, steps, dtype=dtype, device=device)
    from . import ops as npu_ops

    base = full_create(linear.shape, 10.0, dtype=dtype, device=device)
    return npu_ops.pow(base, linear)


def eye_create(n, m=None, dtype=None, device=None, out=None):
    if m is None:
        m = n
    if n < 0:
        raise ValueError(f"n must be greater or equal to 0, got {n}")
    if m < 0:
        raise ValueError(f"m must be greater or equal to 0, got {m}")
    dtype = _resolve_dtype(dtype)
    if not aclnn.eye_symbols_ok():
        raise RuntimeError("aclnnEye not available")

    shape = (int(n), int(m))
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))

    if out is not None:
        out_storage = out.storage()
        aclnn.eye(n, m, out_storage.data_ptr(), out.shape, out.stride, out.dtype, runtime, stream=stream.stream)
        return out

    size = int(n) * int(m)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.eye(n, m, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def range_create(start, end, step=1, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if not aclnn.range_symbols_ok():
        raise RuntimeError("aclnnRange not available")

    # torch.range is inclusive, so shape is inferred from inclusive end.
    arr = np.arange(start, end + step, step, dtype=to_numpy_dtype(dtype))
    shape = tuple(arr.shape)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))

    # CANN limitation: aclnnRange can fail for float32/int32/int64 after aclnnArange in the same process.
    # Keep single-op path by running aclnnRange in float64 then casting back when needed.
    range_dtype = float64_dtype if dtype.name in ("float32", "int32", "int64") else dtype
    size = int(arr.size)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(range_dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.range_(start, end, step, ptr, shape, stride, range_dtype, runtime, stream=stream.stream)

    if range_dtype != dtype:
        if not aclnn.cast_symbols_ok():
            raise RuntimeError("aclnnCast not available")
        out_itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
        out_ptr = npu_runtime._alloc_device(max(size, 1) * out_itemsize, runtime=runtime)
        aclnn.cast(ptr, out_ptr, shape, stride, range_dtype, dtype, runtime, stream=stream.stream)
        runtime.defer_free(ptr)
        ptr = out_ptr

    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)

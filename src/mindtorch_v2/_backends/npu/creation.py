import numpy as np

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

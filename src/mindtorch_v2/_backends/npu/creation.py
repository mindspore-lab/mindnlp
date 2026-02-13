import numpy as np

from ..._storage import npu_typed_storage_from_ptr
from . import runtime as npu_runtime
from . import aclnn


def _wrap_tensor(storage, shape, stride):
    from ..._tensor import Tensor

    return Tensor(storage, shape, stride)


def _require_inplace_one_zero():
    if not aclnn.ones_zero_symbols_ok():
        raise RuntimeError("aclnnInplaceOne/Zero not available")


def tensor_create(data, dtype=None, device=None):
    arr = np.array(data, dtype=npu_runtime._dtype_to_numpy(dtype))
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, arr.size, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return _wrap_tensor(storage, arr.shape, stride)


def zeros_create(shape, dtype=None, device=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    _require_inplace_one_zero()
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)
    aclnn.inplace_zero(ptr, shape, stride, dtype, runtime)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride)


def ones_create(shape, dtype=None, device=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    _require_inplace_one_zero()
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)
    aclnn.inplace_one(ptr, shape, stride, dtype, runtime)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride)


def empty_create(shape, dtype=None, device=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    stride = npu_runtime._contiguous_stride(shape)
    return _wrap_tensor(storage, shape, stride)

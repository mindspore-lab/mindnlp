import numpy as np

from ..._storage import npu_typed_storage_from_ptr
from . import aclnn
from . import runtime as npu_runtime


def _unwrap_storage(tensor):
    if tensor.storage().device.type != "npu":
        raise ValueError("Expected NPU storage for NPU op")
    return tensor.storage()


def _wrap_tensor(storage, shape, stride):
    from ..._tensor import Tensor

    return Tensor(storage, shape, stride)


def add(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU add expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU add requires matching dtypes")

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)

    out_size = int(np.prod(a.shape)) * np.dtype(npu_runtime._dtype_to_numpy(a.dtype)).itemsize
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.add(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime.stream,
    )
    ret = npu_runtime.acl.rt.synchronize_stream(runtime.stream)
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, int(np.prod(a.shape)), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, a.stride)


def mul(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU mul expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU mul requires matching dtypes")

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    out_size = int(np.prod(a.shape)) * np.dtype(npu_runtime._dtype_to_numpy(a.dtype)).itemsize
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.mul(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime.stream,
    )
    ret = npu_runtime.acl.rt.synchronize_stream(runtime.stream)
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, int(np.prod(a.shape)), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, a.stride)


def relu(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU relu expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = int(np.prod(a.shape)) * np.dtype(npu_runtime._dtype_to_numpy(a.dtype)).itemsize
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.relu(
        a_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime.stream,
    )
    ret = npu_runtime.acl.rt.synchronize_stream(runtime.stream)
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, int(np.prod(a.shape)), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, a.stride)


def sum_(a, dim=None, keepdim=False):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU sum expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_shape = list(a.shape)
    if dim is None:
        dims = list(range(len(out_shape)))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    for d in sorted(dims):
        out_shape[d] = 1
    if not keepdim:
        out_shape = [s for i, s in enumerate(out_shape) if i not in dims]
    out_shape = tuple(out_shape)

    out_size = int(np.prod(out_shape) if out_shape else 1) * np.dtype(npu_runtime._dtype_to_numpy(a.dtype)).itemsize
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    dims_payload = {
        "dims": dims if dim is not None else None,
        "out_shape": out_shape,
        "out_stride": npu_runtime._contiguous_stride(out_shape),
    }
    aclnn.reduce_sum(
        a_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims_payload,
        keepdim,
        runtime.stream,
    )
    ret = npu_runtime.acl.rt.synchronize_stream(runtime.stream)
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, int(np.prod(out_shape) if out_shape else 1), a.dtype, device=a.device)
    return _wrap_tensor(storage, out_shape, npu_runtime._contiguous_stride(out_shape))

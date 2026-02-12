import atexit
import os
import numpy as np

from .acl_loader import ensure_acl
from .._storage import npu_typed_storage_from_ptr

from .._dispatch.registry import registry
from . import ascend_ctypes

acl = None

# Constants mirrored from acl_engine/tools/constant.py
ACL_ERROR_CODE = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_FORMAT_ND = 2

ACL_DTYPE = {
    "float": 0,
    "float16": 1,
    "int64": 9,
}

_NP_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
}

_RUNTIME_CLEANUP_REGISTERED = False


def _register_runtime_cleanup(runtime):
    global _RUNTIME_CLEANUP_REGISTERED
    if _RUNTIME_CLEANUP_REGISTERED:
        return
    atexit.register(runtime.finalize)
    _RUNTIME_CLEANUP_REGISTERED = True


class _Runtime:
    def __init__(self):
        self.initialized = False
        self.device_id = 0
        self.context = None
        self.stream = None

    def init(self, device_id=0):
        if self.initialized:
            return
        global acl
        if acl is None:
            acl = ensure_acl()
        ret = acl.init()
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.init failed: {ret}")
        ret = acl.rt.set_device(device_id)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.set_device failed: {ret}")
        self.context, ret = acl.rt.create_context(device_id)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.create_context failed: {ret}")
        self.stream, ret = acl.rt.create_stream()
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.create_stream failed: {ret}")
        self.device_id = device_id
        self.initialized = True
        _register_runtime_cleanup(self)

    def finalize(self):
        if not self.initialized:
            return
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
        if self.context is not None:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        self.stream = None
        self.context = None
        self.initialized = False


_runtime = _Runtime()
_MODEL_DIR = None

_CANDIDATE_MODEL_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/opp",
    "/home/lvyufeng/lvyufeng/acl_engine",
)


def is_available():
    try:
        _runtime.init(0)
        return ascend_ctypes.is_available()
    except Exception:
        return False


def _probe_model_dirs():
    global _MODEL_DIR
    _runtime.init(0)
    selected = None
    for path in _CANDIDATE_MODEL_DIRS:
        if not os.path.isdir(path):
            continue
        if selected is None:
            selected = path
        try:
            ret = acl.op.set_model_dir(path)
        except Exception:
            ret = None
        if ret == ACL_ERROR_CODE:
            _MODEL_DIR = path
            return True
    if selected is not None:
        _MODEL_DIR = selected
        return True
    _MODEL_DIR = None
    return False


def _model_dir():
    if _MODEL_DIR is None:
        _probe_model_dirs()
    if _MODEL_DIR is None:
        raise RuntimeError("NPU op model dir not initialized")
    return _MODEL_DIR


def _ensure_model_dir():
    _model_dir()


def _normalize_dtype(dtype):
    name = getattr(dtype, "name", None)
    if name is not None:
        return name
    return str(dtype)


def _dtype_to_acl(dtype):
    dtype = _normalize_dtype(dtype)
    if dtype == "float32":
        return ACL_DTYPE["float"]
    if dtype == "float16":
        return ACL_DTYPE["float16"]
    if dtype == "int64":
        return ACL_DTYPE["int64"]
    raise ValueError(f"Unsupported dtype for NPU: {dtype}")


def _dtype_to_numpy(dtype):
    dtype = _normalize_dtype(dtype)
    if dtype not in _NP_DTYPE:
        raise ValueError(f"Unsupported dtype for NPU: {dtype}")
    return _NP_DTYPE[dtype]


def _alloc_device(size):
    global acl
    if acl is None:
        acl = ensure_acl()
    dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc failed: {ret}")
    return dev_ptr


def _memcpy_h2d(dst, size, src_ptr):
    global acl
    if acl is None:
        acl = ensure_acl()
    ret = acl.rt.memcpy(dst, size, src_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy H2D failed: {ret}")


def _memcpy_d2h(dst_ptr, size, src):
    global acl
    if acl is None:
        acl = ensure_acl()
    ret = acl.rt.memcpy(dst_ptr, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2H failed: {ret}")


def _create_desc(dtype, shape):
    global acl
    if acl is None:
        acl = ensure_acl()
    return acl.create_tensor_desc(_dtype_to_acl(dtype), list(shape), ACL_FORMAT_ND)


def _create_buffer(dev_ptr, size):
    global acl
    if acl is None:
        acl = ensure_acl()
    return acl.create_data_buffer(dev_ptr, size)


def _host_ptr_from_numpy(arr):
    global acl
    if acl is None:
        acl = ensure_acl()
    return acl.util.numpy_to_ptr(arr)


def _numpy_from_ptr(ptr, shape, dtype):
    global acl
    if acl is None:
        acl = ensure_acl()
    np_dtype = np.dtype(_dtype_to_numpy(dtype))
    count = int(np.prod(shape))
    size = count * np_dtype.itemsize
    byte_data = acl.util.ptr_to_bytes(ptr, size)
    arr = np.frombuffer(bytearray(byte_data), dtype=np_dtype, count=count)
    return arr.reshape(shape)


def _copy_cpu_to_npu(arr):
    _runtime.init(0)
    global acl
    if acl is None:
        acl = ensure_acl()
    size = arr.nbytes
    dev_ptr = _alloc_device(size)
    host_ptr = _host_ptr_from_numpy(arr)
    _memcpy_h2d(dev_ptr, size, host_ptr)
    return dev_ptr, int(size)


def _copy_npu_to_cpu(ptr, size, shape, dtype):
    _runtime.init(0)
    global acl
    if acl is None:
        acl = ensure_acl()
    host_ptr, ret = acl.rt.malloc_host(size)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc_host failed: {ret}")
    _memcpy_d2h(host_ptr, size, ptr)
    arr = _numpy_from_ptr(host_ptr, shape, dtype).copy()
    acl.rt.free_host(host_ptr)
    return arr


def _execute_v2(op_type, input_descs, input_bufs, output_descs, output_bufs, attr):
    global acl
    if acl is None:
        acl = ensure_acl()
    ret = acl.op.update_params(op_type, input_descs, output_descs, attr)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.op.update_params failed: {ret}")
    ret = acl.op.execute_v2(op_type, input_descs, input_bufs, output_descs, output_bufs, attr, _runtime.stream)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.op.execute_v2 failed: {ret}")
    ret = acl.rt.synchronize_stream(_runtime.stream)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")


def _unwrap_storage(tensor):
    if tensor.storage().device.type != "npu":
        raise ValueError("Expected NPU storage for NPU op")
    return tensor.storage()


def _wrap_tensor(storage, shape, stride):
    from .._tensor import Tensor
    return Tensor(storage, shape, stride)


def add(a, b):
    _runtime.init(0)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU add expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU add requires matching dtypes")

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)

    out_size = int(np.prod(a.shape)) * np.dtype(_dtype_to_numpy(a.dtype)).itemsize
    out_ptr = _alloc_device(out_size)
    ascend_ctypes.add(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        _runtime.stream,
    )
    ret = acl.rt.synchronize_stream(_runtime.stream)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, int(np.prod(a.shape)), a.dtype)
    return _wrap_tensor(storage, a.shape, a.stride)


def mul(a, b):
    _runtime.init(0)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU mul expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU mul requires matching dtypes")

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    out_size = int(np.prod(a.shape)) * np.dtype(_dtype_to_numpy(a.dtype)).itemsize
    out_ptr = _alloc_device(out_size)
    ascend_ctypes.mul(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        _runtime.stream,
    )
    ret = acl.rt.synchronize_stream(_runtime.stream)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, int(np.prod(a.shape)), a.dtype)
    return _wrap_tensor(storage, a.shape, a.stride)


def relu(a):
    _runtime.init(0)
    if a.device.type != "npu":
        raise ValueError("NPU relu expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = int(np.prod(a.shape)) * np.dtype(_dtype_to_numpy(a.dtype)).itemsize
    out_ptr = _alloc_device(out_size)
    ascend_ctypes.relu(
        a_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        _runtime.stream,
    )
    ret = acl.rt.synchronize_stream(_runtime.stream)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, int(np.prod(a.shape)), a.dtype)
    return _wrap_tensor(storage, a.shape, a.stride)


def sum_(a, dim=None, keepdim=False):
    _runtime.init(0)
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

    out_size = int(np.prod(out_shape) if out_shape else 1) * np.dtype(_dtype_to_numpy(a.dtype)).itemsize
    out_ptr = _alloc_device(out_size)
    dims_payload = {
        "dims": dims if dim is not None else None,
        "out_shape": out_shape,
        "out_stride": _contiguous_stride(out_shape),
    }
    ascend_ctypes.reduce_sum(
        a_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims_payload,
        keepdim,
        _runtime.stream,
    )
    ret = acl.rt.synchronize_stream(_runtime.stream)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, int(np.prod(out_shape) if out_shape else 1), a.dtype)
    return _wrap_tensor(storage, out_shape, _contiguous_stride(out_shape))


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def tensor_create(data, dtype=None, device=None):
    arr = np.array(data, dtype=_dtype_to_numpy(dtype))
    ptr, _ = _copy_cpu_to_npu(arr)
    storage = npu_typed_storage_from_ptr(ptr, arr.size, dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return _wrap_tensor(storage, arr.shape, stride)


def zeros_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(_dtype_to_numpy(dtype)).itemsize
    ptr = _alloc_device(out_size)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype)
    stride = _contiguous_stride(shape)
    return _wrap_tensor(storage, shape, stride)


def ones_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(_dtype_to_numpy(dtype)).itemsize
    ptr = _alloc_device(out_size)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype)
    stride = _contiguous_stride(shape)
    return _wrap_tensor(storage, shape, stride)


def empty_create(shape, dtype=None, device=None):
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(_dtype_to_numpy(dtype)).itemsize
    ptr = _alloc_device(out_size)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype)
    stride = _contiguous_stride(shape)
    return _wrap_tensor(storage, shape, stride)


registry.register("add", "npu", add)
registry.register("mul", "npu", mul)
registry.register("relu", "npu", relu)
registry.register("sum", "npu", sum_)
from .common import view as view_backend


registry.register("reshape", "npu", view_backend.reshape)
registry.register("view", "npu", view_backend.view)
registry.register("transpose", "npu", view_backend.transpose)
from .common import convert as convert_backend


registry.register("to", "npu", convert_backend.to_device)

registry.register("tensor", "npu", tensor_create)
registry.register("zeros", "npu", zeros_create)
registry.register("ones", "npu", ones_create)
registry.register("empty", "npu", empty_create)

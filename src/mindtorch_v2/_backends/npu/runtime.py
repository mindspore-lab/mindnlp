import atexit
import os
import numpy as np

from .acl_loader import ensure_acl

acl = None

ACL_ERROR_CODE = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_REPEAT_INITIALIZE = 100002

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
        self._deferred_frees = []

    def init(self, device_id=0):
        if self.initialized:
            self.activate()
            return
        global acl
        if acl is None:
            acl = ensure_acl()
        ret = acl.init()
        if ret not in (ACL_ERROR_CODE, ACL_ERROR_REPEAT_INITIALIZE):
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

    def activate(self):
        if not self.initialized:
            self.init(self.device_id)
            return
        ret = acl.rt.set_device(self.device_id)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.set_device failed: {ret}")
        if hasattr(acl.rt, "set_context"):
            ret = acl.rt.set_context(self.context)
            if ret != ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.set_context failed: {ret}")


    def defer_free(self, ptr):
        if ptr is None:
            return
        self._deferred_frees.append(ptr)

    def synchronize(self):
        if not self.initialized:
            return
        global acl
        if acl is None:
            acl = ensure_acl()
        self.activate()
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")
        frees = self._deferred_frees
        self._deferred_frees = []
        for ptr in frees:
            ret = acl.rt.free(ptr)
            if ret != ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.free failed: {ret}")

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
_RUNTIMES = {}


_MODEL_DIR = None

_CANDIDATE_MODEL_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/opp",
    "/home/lvyufeng/lvyufeng/acl_engine",
)




def get_runtime(device_id=0):
    runtime = _RUNTIMES.get(device_id)
    if runtime is None:
        runtime = _Runtime()
        runtime.init(device_id)
        _RUNTIMES[device_id] = runtime
    else:
        runtime.activate()
    return runtime


def is_available():
    try:
        get_runtime(0)
        from . import aclnn

        return aclnn.is_available()
    except Exception:
        return False


def _probe_model_dirs():
    global _MODEL_DIR
    get_runtime(0)
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




def device_count():
    global acl
    if acl is None:
        acl = ensure_acl()
    count, ret = acl.rt.get_device_count()
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.get_device_count failed: {ret}")
    return int(count)


def _normalize_dtype(dtype):
    name = getattr(dtype, "name", None)
    if name is not None:
        return name
    return str(dtype)


def _dtype_to_numpy(dtype):
    dtype = _normalize_dtype(dtype)
    if dtype not in _NP_DTYPE:
        raise ValueError(f"Unsupported dtype for NPU: {dtype}")
    return _NP_DTYPE[dtype]


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _alloc_device(size, runtime=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    if runtime is not None:
        runtime.activate()
    dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc failed: {ret}")
    return dev_ptr


def _memcpy_h2d(dst, size, src_ptr, runtime=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    if runtime is not None:
        runtime.activate()
    ret = acl.rt.memcpy(dst, size, src_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy H2D failed: {ret}")


def _memcpy_d2h(dst_ptr, size, src, runtime=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    if runtime is not None:
        runtime.activate()
    ret = acl.rt.memcpy(dst_ptr, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2H failed: {ret}")


def _host_ptr_from_numpy(arr):
    global acl
    if acl is None:
        acl = ensure_acl()
    if hasattr(acl.util, "bytes_to_ptr"):
        buf = arr.tobytes()
        return acl.util.bytes_to_ptr(buf), buf
    return acl.util.numpy_to_ptr(arr), arr


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


def _copy_cpu_to_npu(arr, runtime=None):
    if runtime is None:
        runtime = get_runtime(0)
    global acl
    if acl is None:
        acl = ensure_acl()
    size = arr.nbytes
    dev_ptr = _alloc_device(size, runtime=runtime)
    host_ptr, _host_buf = _host_ptr_from_numpy(arr)
    _memcpy_h2d(dev_ptr, size, host_ptr, runtime=runtime)
    return dev_ptr, int(size)


def _copy_npu_to_cpu(ptr, size, shape, dtype, runtime=None):
    if runtime is None:
        runtime = get_runtime(0)
    global acl
    if acl is None:
        acl = ensure_acl()
    runtime.activate()
    host_ptr, ret = acl.rt.malloc_host(size)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc_host failed: {ret}")
    _memcpy_d2h(host_ptr, size, ptr)
    arr = _numpy_from_ptr(host_ptr, shape, dtype).copy()
    acl.rt.free_host(host_ptr)
    return arr

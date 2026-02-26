import atexit
import ctypes
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
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bfloat16": np.uint16,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "bool": np.bool_,
    "complex64": np.complex64,
    "complex128": np.complex128,
}

_RUNTIME_CLEANUP_REGISTERED = False

def _normalize_priority(priority):

    if isinstance(priority, tuple):

        priority = priority[0] if priority else 0

    if priority is None:

        priority = 0

    return int(priority)


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
        self._deferred_host_frees = []

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


    def create_stream(self, priority=0):
        if not self.initialized:
            self.init(self.device_id)
        self.activate()
        priority = _normalize_priority(priority)
        stream = None
        ret = None
        if hasattr(acl.rt, 'create_stream_with_config'):
            try:
                stream, ret = acl.rt.create_stream_with_config(priority, 0)
            except TypeError:
                stream = None
                ret = None
        if ret != ACL_ERROR_CODE:
            stream, ret = acl.rt.create_stream()
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.create_stream failed: {ret}')
        return stream

    def destroy_stream(self, stream):
        if stream is None:
            return
        ret = acl.rt.destroy_stream(stream)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.destroy_stream failed: {ret}')

    def synchronize_stream(self, stream):
        if stream is None:
            return
        ret = acl.rt.synchronize_stream(stream)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.synchronize_stream failed: {ret}')

    def create_event(self, enable_timing, blocking, interprocess):
        if not self.initialized:
            self.init(self.device_id)
        self.activate()
        flag = 0
        if enable_timing:
            flag |= 1
        if hasattr(acl.rt, 'create_event_ex_with_flag'):
            event, ret = acl.rt.create_event_ex_with_flag(flag)
        elif hasattr(acl.rt, 'create_event_with_flag'):
            event, ret = acl.rt.create_event_with_flag(flag)
        else:
            event, ret = acl.rt.create_event()
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.create_event failed: {ret}')
        return event

    def destroy_event(self, event):
        if event is None:
            return
        ret = acl.rt.destroy_event(event)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.destroy_event failed: {ret}')

    def record_event(self, event, stream):
        ret = acl.rt.record_event(event, stream)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.record_event failed: {ret}')

    def synchronize_event(self, event):
        ret = acl.rt.synchronize_event(event)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.synchronize_event failed: {ret}')

    def query_event(self, event):
        ret = acl.rt.query_event(event)
        if ret != ACL_ERROR_CODE:
            return False
        return True

    def event_elapsed_time(self, start, end):
        ms, ret = acl.rt.event_elapsed_time(start, end)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.event_elapsed_time failed: {ret}')
        return float(ms)

    def stream_wait_event(self, stream, event):
        ret = acl.rt.stream_wait_event(stream, event)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.stream_wait_event failed: {ret}')

    def synchronize_device(self):
        ret = acl.rt.synchronize_device()
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f'acl.rt.synchronize_device failed: {ret}')

    def defer_host_free(self, ptr):
        if ptr is None:
            return
        self._deferred_host_frees.append(ptr)

    def defer_free(self, ptr):
        if ptr is None:
            return
        self._deferred_frees.append(ptr)

    def synchronize(self):
        if not self.initialized:
            return
        self.activate()
        from . import allocator as npu_allocator

        npu_allocator.get_allocator(self.device_id).synchronize()
        frees = self._deferred_frees
        self._deferred_frees = []
        for ptr in frees:
            npu_allocator.get_allocator(self.device_id).free(ptr)
        host_frees = self._deferred_host_frees
        self._deferred_host_frees = []
        for ptr in host_frees:
            ret = acl.rt.free_host(ptr)
            if ret != ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.free_host failed: {ret}")

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


def is_available(verbose=False):
    try:
        get_runtime(0)
        from . import aclnn

        return aclnn.is_available()
    except Exception as exc:
        if verbose:
            import warnings

            warnings.warn(f"NPU unavailable: {exc}")
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






def mem_get_info(device_id=0, attr=0):
    global acl
    if acl is None:
        acl = ensure_acl()
    runtime = get_runtime(device_id)
    runtime.activate()
    if not hasattr(acl.rt, "get_mem_info"):
        raise RuntimeError("acl.rt.get_mem_info not available")
    free, total, ret = acl.rt.get_mem_info(attr)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.get_mem_info failed: {ret}")
    return int(free), int(total)


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
    from . import state as npu_state
    from . import allocator as npu_allocator

    device_id = 0
    if runtime is not None:
        device_id = int(runtime.device_id)
    stream = npu_state.current_stream(device_id).stream
    alloc = npu_allocator.get_allocator(device_id)
    return alloc.malloc(int(size), stream=stream)


def alloc_host(size):
    global acl
    if acl is None:
        acl = ensure_acl()
    host_ptr, ret = acl.rt.malloc_host(int(size))
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc_host failed: {ret}")
    return host_ptr


def free_host(ptr):
    global acl
    if acl is None:
        acl = ensure_acl()
    ret = acl.rt.free_host(ptr)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.free_host failed: {ret}")


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


def _copy_cpu_to_npu(arr, runtime=None, non_blocking=False, stream=None):
    if runtime is None:
        runtime = get_runtime(0)
    global acl
    if acl is None:
        acl = ensure_acl()
    size = arr.nbytes
    dev_ptr = _alloc_device(size, runtime=runtime)
    runtime.activate()
    host_ptr, ret = acl.rt.malloc_host(size)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc_host failed: {ret}")
    try:
        ctypes.memmove(int(host_ptr), int(arr.ctypes.data), int(size))
        if non_blocking and hasattr(acl.rt, "memcpy_async"):
            use_stream = runtime.stream if stream is None else stream
            ret = acl.rt.memcpy_async(dev_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, use_stream)
            if ret != ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy_async H2D failed: {ret}")
            runtime.defer_host_free(host_ptr)
        else:
            _memcpy_h2d(dev_ptr, size, host_ptr, runtime=runtime)
            runtime.defer_host_free(host_ptr)
    except Exception:
        acl.rt.free_host(host_ptr)
        raise
    return dev_ptr, int(size)


def _copy_npu_to_cpu(ptr, size, shape, dtype, runtime=None, non_blocking=False, stream=None, event=None):
    if runtime is None:
        runtime = get_runtime(0)
    global acl
    if acl is None:
        acl = ensure_acl()
    runtime.activate()
    host_ptr, ret = acl.rt.malloc_host(size)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc_host failed: {ret}")
    if non_blocking and hasattr(acl.rt, "memcpy_async"):
        use_stream = runtime.stream if stream is None else stream
        stream_handle = use_stream.stream if hasattr(use_stream, "stream") else use_stream
        ret = acl.rt.memcpy_async(host_ptr, size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST, stream_handle)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.memcpy_async D2H failed: {ret}")
        if event is not None and hasattr(runtime, "stream_wait_event"):
            runtime.stream_wait_event(stream_handle, event.event)
        if hasattr(runtime, "synchronize_stream"):
            runtime.synchronize_stream(stream_handle)
        arr = _numpy_from_ptr(host_ptr, shape, dtype).copy()
        acl.rt.free_host(host_ptr)
        return arr
    _memcpy_d2h(host_ptr, size, ptr)
    arr = _numpy_from_ptr(host_ptr, shape, dtype).copy()
    acl.rt.free_host(host_ptr)
    return arr

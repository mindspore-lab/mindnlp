import ctypes
import ctypes.util
import os
from typing import Optional


CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_HOST = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

_CUDART = None
_CUDART_LOAD_ERROR = None


class CudaRuntimeError(RuntimeError):
    pass


_CANDIDATE_LIBRARIES = (
    "libcudart.so",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "libcudart.so.11",
)


def _library_candidates():
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        for name in _CANDIDATE_LIBRARIES:
            candidates.append(os.path.join(cuda_home, "lib64", name))
            candidates.append(os.path.join(cuda_home, "targets", "x86_64-linux", "lib", name))

    candidates.extend(_CANDIDATE_LIBRARIES)

    deduped = []
    seen = set()
    for item in candidates:
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def _bind(lib):
    lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cudaGetDeviceCount.restype = ctypes.c_int

    lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cudaGetDevice.restype = ctypes.c_int

    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int

    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceSynchronize.restype = ctypes.c_int

    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int

    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int

    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int

    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.cudaMemcpyAsync.restype = ctypes.c_int

    lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.cudaMemset.restype = ctypes.c_int

    lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaStreamCreate.restype = ctypes.c_int

    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int

    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int

    lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaEventCreate.restype = ctypes.c_int

    lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaEventDestroy.restype = ctypes.c_int

    lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventRecord.restype = ctypes.c_int

    lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaEventSynchronize.restype = ctypes.c_int

    if hasattr(lib, "cudaGetErrorString"):
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p


def _load_cudart():
    global _CUDART, _CUDART_LOAD_ERROR

    if _CUDART is not None:
        return _CUDART
    if _CUDART_LOAD_ERROR is not None:
        raise _CUDART_LOAD_ERROR

    last_error = None
    for candidate in _library_candidates():
        try:
            lib = ctypes.CDLL(candidate)
            _bind(lib)
            _CUDART = lib
            return lib
        except OSError as exc:
            last_error = exc

    if last_error is None:
        last_error = OSError("CUDA runtime library not found")
    _CUDART_LOAD_ERROR = last_error
    raise last_error


def _try_load_cudart():
    try:
        return _load_cudart()
    except OSError:
        return None


def _error_string(code: int) -> str:
    lib = _try_load_cudart()
    if lib is not None and hasattr(lib, "cudaGetErrorString"):
        raw = lib.cudaGetErrorString(int(code))
        if raw:
            return raw.decode("utf-8", errors="replace")
    return f"CUDA error {code}"


def _check(code: int, opname: str):
    if int(code) != CUDA_SUCCESS:
        raise CudaRuntimeError(f"{opname} failed: {_error_string(code)}")


def is_available() -> bool:
    try:
        return device_count() > 0
    except Exception:
        return False


def device_count() -> int:
    lib = _try_load_cudart()
    if lib is None:
        return 0
    count = ctypes.c_int()
    code = lib.cudaGetDeviceCount(ctypes.byref(count))
    if int(code) != CUDA_SUCCESS:
        return 0
    return int(count.value)


def current_device() -> int:
    lib = _load_cudart()
    current = ctypes.c_int()
    _check(lib.cudaGetDevice(ctypes.byref(current)), "cudaGetDevice")
    return int(current.value)


def set_device(device_index: int):
    lib = _load_cudart()
    _check(lib.cudaSetDevice(int(device_index)), "cudaSetDevice")


def synchronize(device: Optional[int] = None):
    lib = _load_cudart()
    if device is not None:
        set_device(device)
    _check(lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def malloc(nbytes: int) -> int:
    lib = _load_cudart()
    ptr = ctypes.c_void_p()
    _check(lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes))), "cudaMalloc")
    return int(ptr.value or 0)


def free(ptr: int):
    lib = _load_cudart()
    _check(lib.cudaFree(ctypes.c_void_p(int(ptr))), "cudaFree")


def memcpy(dst: int, src: int, nbytes: int, kind: int):
    lib = _load_cudart()
    _check(
        lib.cudaMemcpy(
            ctypes.c_void_p(int(dst)),
            ctypes.c_void_p(int(src)),
            ctypes.c_size_t(int(nbytes)),
            int(kind),
        ),
        "cudaMemcpy",
    )


def memcpy_async(dst: int, src: int, nbytes: int, kind: int, stream=None):
    lib = _load_cudart()
    stream_ptr = ctypes.c_void_p(0 if stream is None else int(stream))
    _check(
        lib.cudaMemcpyAsync(
            ctypes.c_void_p(int(dst)),
            ctypes.c_void_p(int(src)),
            ctypes.c_size_t(int(nbytes)),
            int(kind),
            stream_ptr,
        ),
        "cudaMemcpyAsync",
    )


def memset(dst: int, value: int, nbytes: int):
    lib = _load_cudart()
    _check(lib.cudaMemset(ctypes.c_void_p(int(dst)), int(value), ctypes.c_size_t(int(nbytes))), "cudaMemset")


def create_stream() -> int:
    lib = _load_cudart()
    stream = ctypes.c_void_p()
    _check(lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    return int(stream.value or 0)


def destroy_stream(stream: int):
    lib = _load_cudart()
    _check(lib.cudaStreamDestroy(ctypes.c_void_p(int(stream))), "cudaStreamDestroy")


def synchronize_stream(stream: int):
    lib = _load_cudart()
    _check(lib.cudaStreamSynchronize(ctypes.c_void_p(int(stream))), "cudaStreamSynchronize")


def create_event() -> int:
    lib = _load_cudart()
    event = ctypes.c_void_p()
    _check(lib.cudaEventCreate(ctypes.byref(event)), "cudaEventCreate")
    return int(event.value or 0)


def destroy_event(event: int):
    lib = _load_cudart()
    _check(lib.cudaEventDestroy(ctypes.c_void_p(int(event))), "cudaEventDestroy")


def record_event(event: int, stream=None):
    lib = _load_cudart()
    stream_ptr = ctypes.c_void_p(0 if stream is None else int(stream))
    _check(lib.cudaEventRecord(ctypes.c_void_p(int(event)), stream_ptr), "cudaEventRecord")


def synchronize_event(event: int):
    lib = _load_cudart()
    _check(lib.cudaEventSynchronize(ctypes.c_void_p(int(event))), "cudaEventSynchronize")


__all__ = [
    "CUDA_SUCCESS",
    "CUDA_MEMCPY_HOST_TO_HOST",
    "CUDA_MEMCPY_HOST_TO_DEVICE",
    "CUDA_MEMCPY_DEVICE_TO_HOST",
    "CUDA_MEMCPY_DEVICE_TO_DEVICE",
    "CudaRuntimeError",
    "is_available",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "malloc",
    "free",
    "memcpy",
    "memcpy_async",
    "memset",
    "create_stream",
    "destroy_stream",
    "synchronize_stream",
    "create_event",
    "destroy_event",
    "record_event",
    "synchronize_event",
]

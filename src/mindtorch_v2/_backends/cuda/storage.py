import weakref

import numpy as np

from ..._device import device as Device
from ..._dtype import to_numpy_dtype
from . import runtime as cuda_runtime


class CudaUntypedStorage:
    def __init__(self, nbytes, device=None, device_ptr=0, host_shadow=None):
        if isinstance(device, str):
            device = Device(device)
        self.device = device or Device("cuda")
        self._nbytes = int(nbytes)
        self._device_ptr = int(device_ptr or 0)
        if host_shadow is None:
            host_shadow = np.empty((self._nbytes,), dtype=np.uint8)
        self._host_shadow = np.ascontiguousarray(host_shadow, dtype=np.uint8).reshape(-1)
        self._finalizer = None
        if self._device_ptr:
            self._finalizer = weakref.finalize(self, cuda_runtime.free, self._device_ptr)

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        return self._device_ptr

    def buffer(self):
        raise RuntimeError("Cannot get buffer of CUDA storage on CPU")

    def resize_(self, new_nbytes):
        new_nbytes = int(new_nbytes)
        if new_nbytes == self._nbytes:
            return self

        new_shadow = np.empty((new_nbytes,), dtype=np.uint8)
        copy_bytes = min(self._nbytes, new_nbytes)
        if copy_bytes:
            new_shadow[:copy_bytes] = self._host_shadow[:copy_bytes]

        new_ptr = 0
        if new_nbytes and cuda_runtime._try_load_cudart() is not None and cuda_runtime.device_count() > 0:
            new_ptr = cuda_runtime.malloc(new_nbytes)
            if self._device_ptr and copy_bytes:
                cuda_runtime.memcpy(
                    new_ptr,
                    self._device_ptr,
                    copy_bytes,
                    cuda_runtime.CUDA_MEMCPY_DEVICE_TO_DEVICE,
                )

        old_ptr = self._device_ptr
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if old_ptr:
            cuda_runtime.free(old_ptr)

        self._device_ptr = int(new_ptr)
        self._nbytes = new_nbytes
        self._host_shadow = new_shadow
        if self._device_ptr:
            self._finalizer = weakref.finalize(self, cuda_runtime.free, self._device_ptr)
        return self

    def is_shared(self):
        return False

    def is_pinned(self):
        return False


def _allocate_device_bytes(nbytes):
    if int(nbytes) <= 0:
        return 0
    if cuda_runtime._try_load_cudart() is None:
        return 0
    if cuda_runtime.device_count() <= 0:
        return 0
    try:
        return cuda_runtime.malloc(int(nbytes))
    except Exception:
        return 0


def untyped_from_numpy(arr, device=None):
    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1).copy()
    ptr = _allocate_device_bytes(raw.size)
    if ptr and raw.size:
        cuda_runtime.memcpy(
            ptr,
            int(raw.ctypes.data),
            raw.size,
            cuda_runtime.CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    return CudaUntypedStorage(raw.size, device=device, device_ptr=ptr, host_shadow=raw)


def empty_untyped(shape, dtype, device=None):
    shape = tuple(shape)
    itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
    nbytes = int(np.prod(shape)) * itemsize
    raw = np.empty((nbytes,), dtype=np.uint8)
    ptr = _allocate_device_bytes(nbytes)
    return CudaUntypedStorage(nbytes, device=device, device_ptr=ptr, host_shadow=raw)


def to_numpy(untyped, dtype, shape=None):
    np_dtype = to_numpy_dtype(dtype)
    itemsize = np.dtype(np_dtype).itemsize
    if shape is None:
        count = 0 if itemsize == 0 else int(untyped.nbytes() // itemsize)
        shape = (count,)
    else:
        shape = tuple(shape)
        count = int(np.prod(shape))
    if count == 0:
        return np.empty(shape, dtype=np_dtype)

    if untyped.data_ptr():
        arr = np.empty(shape, dtype=np_dtype)
        cuda_runtime.memcpy(
            int(arr.ctypes.data),
            untyped.data_ptr(),
            arr.nbytes,
            cuda_runtime.CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        untyped._host_shadow = arr.view(np.uint8).reshape(-1).copy()
        return arr

    raw = np.ascontiguousarray(untyped._host_shadow, dtype=np.uint8)
    return raw.view(np_dtype).copy().reshape(shape)


def copy_untyped(dst, src):
    copy_bytes = min(dst.nbytes(), src.nbytes())
    if copy_bytes <= 0:
        return
    if dst.data_ptr() and src.data_ptr():
        cuda_runtime.memcpy(
            dst.data_ptr(),
            src.data_ptr(),
            copy_bytes,
            cuda_runtime.CUDA_MEMCPY_DEVICE_TO_DEVICE,
        )
    if dst._host_shadow.size != dst.nbytes():
        dst._host_shadow = np.empty((dst.nbytes(),), dtype=np.uint8)
    dst._host_shadow[:copy_bytes] = src._host_shadow[:copy_bytes]


__all__ = [
    "CudaUntypedStorage",
    "untyped_from_numpy",
    "empty_untyped",
    "to_numpy",
    "copy_untyped",
]

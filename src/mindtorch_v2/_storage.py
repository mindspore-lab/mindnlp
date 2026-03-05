import atexit
import ctypes
import os
import threading
import weakref
import numpy as np

from ._device import _default_device, device as Device
from ._dtype import float32, to_numpy_dtype
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


_SHARED_FILE_REGISTRY = set()
_SHARED_FILE_REGISTRY_LOCK = threading.Lock()


def _register_shared_file(path):
    if not path:
        return
    with _SHARED_FILE_REGISTRY_LOCK:
        _SHARED_FILE_REGISTRY.add(path)


def _unregister_shared_file(path):
    if not path:
        return
    with _SHARED_FILE_REGISTRY_LOCK:
        _SHARED_FILE_REGISTRY.discard(path)


def _cleanup_shared_file(path):
    if not path:
        return False
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return True
    except Exception:
        return False


def _close_fd_and_cleanup(fd, path=None):
    if fd is not None:
        try:
            os.close(int(fd))
        except Exception:
            pass
    if path and _cleanup_shared_file(path):
        _unregister_shared_file(path)


def cleanup_shared_files():
    with _SHARED_FILE_REGISTRY_LOCK:
        paths = list(_SHARED_FILE_REGISTRY)
    for path in paths:
        if _cleanup_shared_file(path):
            _unregister_shared_file(path)


def shared_files_count():
    with _SHARED_FILE_REGISTRY_LOCK:
        return len(_SHARED_FILE_REGISTRY)


atexit.register(cleanup_shared_files)


class UntypedStorage:
    def __init__(self, device):
        if isinstance(device, str):
            device = Device(device)
        self.device = device

    def nbytes(self):
        raise NotImplementedError

    def data_ptr(self):
        raise NotImplementedError

    def resize_(self, new_nbytes):
        raise NotImplementedError

    def share_memory_(self):
        raise NotImplementedError

    def buffer(self):
        raise NotImplementedError("Subclass must implement buffer()")

    def is_shared(self):
        return False

    def is_pinned(self):
        return False

    @classmethod
    def from_file(cls, filename, shared=False):
        return _CPUUntypedStorage.from_file(filename, shared=shared)

    def filename(self):
        return None



class _PinnedCPUUntypedStorage(UntypedStorage):
    def __init__(self, array, ptr, filename=None, shared=False, device=None):
        super().__init__(device or Device("cpu"))
        self._array = array
        self._shared = shared
        self._filename = filename
        self._ptr = int(ptr)
        from ._backends.npu import runtime as npu_runtime

        self._finalizer = weakref.finalize(self, npu_runtime.free_host, self._ptr)

    def nbytes(self):
        return int(self._array.nbytes)

    def data_ptr(self):
        return int(self._array.ctypes.data)

    def buffer(self):
        return self._array

    def resize_(self, new_nbytes):
        raise NotImplementedError("pinned storage cannot resize")

    def share_memory_(self):
        self._shared = True
        return self

    def is_shared(self):
        return self._shared

    def is_pinned(self):
        return True

    @classmethod
    def from_file(cls, filename, shared=False):
        data = np.memmap(filename, mode="r+", dtype=np.uint8)
        return cls(data, int(data.ctypes.data), filename=filename, shared=shared)

    def filename(self):
        return self._filename


class _CPUUntypedStorage(UntypedStorage):
    def __init__(
        self,
        array,
        filename=None,
        shared=False,
        device=None,
        mmap_obj=None,
        fd=None,
        tmp_file=None,
        sharing_mechanism=None,
        cleanup_finalizer=None,
    ):
        super().__init__(device or Device("cpu"))
        self._array = array
        self._shared = shared
        self._filename = filename
        self._mmap = mmap_obj
        self._fd = fd
        self._tmp_file = tmp_file
        self._sharing_mechanism = sharing_mechanism
        self._cleanup_finalizer = cleanup_finalizer

    def nbytes(self):
        return int(self._array.nbytes)

    def data_ptr(self):
        return int(self._array.ctypes.data)

    def buffer(self):
        return self._array

    def resize_(self, new_nbytes):
        if self._filename is not None or self._shared:
            raise RuntimeError("Trying to resize storage that is not resizable")
        new_array = np.empty(int(new_nbytes), dtype=np.uint8)
        old_bytes = self._array.view(np.uint8)
        copy_bytes = min(old_bytes.size, new_array.size)
        new_array[:copy_bytes] = old_bytes[:copy_bytes]
        self._array = new_array
        return self

    def share_memory_(self, strategy="file_descriptor"):
        if self._shared:
            return self

        nbytes = int(self._array.nbytes)
        if nbytes == 0:
            self._shared = True
            self._sharing_mechanism = strategy
            return self

        if strategy == "file_descriptor":
            import mmap
            import os
            import tempfile

            fd, filename = tempfile.mkstemp(prefix="mindtorch_v2_fd_", suffix=".bin")
            try:
                os.ftruncate(fd, nbytes)
                mm = mmap.mmap(fd, nbytes)
            except Exception:
                os.close(fd)
                raise

            dst = np.frombuffer(mm, dtype=np.uint8, count=nbytes)
            dst[:] = self._array.view(np.uint8).reshape(-1)

            self._array = dst
            self._mmap = mm
            self._fd = fd
            self._tmp_file = filename
            self._filename = filename
            self._shared = True
            self._sharing_mechanism = "file_descriptor"

            if self._cleanup_finalizer is not None:
                self._cleanup_finalizer.detach()
            self._cleanup_finalizer = weakref.finalize(
                self,
                _close_fd_and_cleanup,
                int(fd),
                filename,
            )
            return self

        if strategy == "file_system":
            import tempfile

            fd, filename = tempfile.mkstemp(prefix="mindtorch_v2_shm_", suffix=".bin")
            try:
                with open(fd, "wb", closefd=False) as f:
                    f.truncate(nbytes)
            finally:
                import os

                os.close(fd)

            mmap_arr = np.memmap(filename, mode="r+", dtype=np.uint8, shape=(nbytes,))
            mmap_arr[:] = self._array.view(np.uint8).reshape(-1)

            self._array = mmap_arr
            self._filename = filename
            self._shared = True
            self._sharing_mechanism = "file_system"
            _register_shared_file(filename)

            if self._cleanup_finalizer is not None:
                self._cleanup_finalizer.detach()
            # file_system lifetime is process-level; explicit cleanup occurs via cleanup_shared_files()
            self._cleanup_finalizer = None
            return self

        raise ValueError(f"unsupported sharing strategy: {strategy}")

    def is_shared(self):
        return self._shared

    def shared_memory_meta(self):
        if self._sharing_mechanism == "file_descriptor":
            return {
                "mechanism": "file_descriptor",
                "fd": int(self._fd),
                "filename": self._filename,
                "nbytes": int(self._array.nbytes),
            }
        if self._sharing_mechanism == "file_system":
            return {
                "mechanism": "file_system",
                "filename": self._filename,
                "nbytes": int(self._array.nbytes),
            }
        return None

    def typed_view(self, dtype, size):
        return np.frombuffer(self._array, dtype=to_numpy_dtype(dtype), count=int(size))

    @classmethod
    def from_shared_memory(cls, filename, nbytes):
        data = np.memmap(filename, mode="r+", dtype=np.uint8, shape=(int(nbytes),))
        _register_shared_file(filename)
        return cls(data, filename=filename, shared=True, sharing_mechanism="file_system")

    @classmethod
    def from_shared_fd(cls, fd, nbytes, filename=None):
        import mmap

        fd = int(fd)
        mm = mmap.mmap(fd, int(nbytes))
        arr = np.frombuffer(mm, dtype=np.uint8, count=int(nbytes))
        storage = cls(
            arr,
            filename=filename,
            shared=True,
            mmap_obj=mm,
            fd=fd,
            tmp_file=filename,
            sharing_mechanism="file_descriptor",
        )
        storage._cleanup_finalizer = weakref.finalize(
            storage,
            _close_fd_and_cleanup,
            fd,
            filename,
        )
        return storage

    @classmethod
    def from_file(cls, filename, shared=False):
        data = np.memmap(filename, mode="r+", dtype=np.uint8)
        return cls(data, filename=filename, shared=shared)

    def filename(self):
        return self._filename


class _NPUUntypedStorage(UntypedStorage):
    def __init__(self, device_ptr, nbytes, device=None):
        super().__init__(device or Device("npu"))
        self._device_ptr = int(device_ptr)
        self._nbytes = int(nbytes)
        from ._backends.npu import allocator as npu_allocator

        alloc = npu_allocator.get_allocator(self.device.index or 0)
        self._finalizer = weakref.finalize(self, alloc.free, self._device_ptr, None)

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        return self._device_ptr

    def is_pinned(self):
        return False

    def buffer(self):
        raise RuntimeError("Cannot get buffer of NPU storage on CPU")

    def resize_(self, new_nbytes):
        new_nbytes = int(new_nbytes)
        if new_nbytes == self._nbytes:
            return self
        from ._backends.npu import allocator as npu_allocator
        from ._backends.npu import runtime as npu_runtime
        from ._backends.npu import state as npu_state

        device_id = self.device.index or 0
        runtime = npu_runtime.get_runtime(device_id)
        stream = npu_state.current_stream(device_id).stream
        alloc = npu_allocator.get_allocator(device_id)
        runtime.activate()
        new_ptr = alloc.malloc(new_nbytes, stream=stream)
        if self._device_ptr:
            copy_bytes = min(self._nbytes, new_nbytes)
            if copy_bytes:
                ret = npu_runtime.acl.rt.memcpy(
                    new_ptr,
                    copy_bytes,
                    self._device_ptr,
                    copy_bytes,
                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                )
                if ret != 0:
                    raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
        alloc.free(self._device_ptr, stream=stream)
        self._device_ptr = int(new_ptr)
        self._nbytes = new_nbytes
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, alloc.free, self._device_ptr, None)
        return self


class _MetaUntypedStorage(UntypedStorage):
    def __init__(self, nbytes, device=None):
        super().__init__(device or Device("meta"))
        self._nbytes = int(nbytes)

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        raise RuntimeError("meta tensor has no data")

    def is_pinned(self):
        return False

    def resize_(self, new_nbytes):
        self._nbytes = int(new_nbytes)
        return self


class TypedStorage:
    def __init__(self, untyped, dtype=None, size=0, data=None):
        self._untyped = untyped
        self.dtype = dtype or float32
        self._size = int(size)
        self._data = data

    @property
    def device(self):
        return self._untyped.device

    def size(self):
        return self._size

    def nbytes(self):
        itemsize = np.dtype(to_numpy_dtype(self.dtype)).itemsize
        return int(self._size * itemsize)

    def data_ptr(self):
        return self._untyped.data_ptr()

    def untyped_storage(self):
        return self._untyped

    def is_shared(self):
        return self._untyped.is_shared()

    def is_pinned(self):
        return self._untyped.is_pinned()

    @property
    def data(self):
        if self.device.type != "cpu":
            raise RuntimeError("storage has no CPU data")
        return self._data

    def clone(self):
        if self.device.type == "cpu":
            return typed_storage_from_numpy(np.copy(self._data), self.dtype, device=self.device)
        if self.device.type == "npu":
            from ._backends.npu import runtime as npu_runtime

            size = self.nbytes()
            runtime = npu_runtime.get_runtime(self.device.index or 0)
            dst_ptr = npu_runtime._alloc_device(size, runtime=runtime)
            runtime.activate()
            ret = npu_runtime.acl.rt.memcpy(
                dst_ptr,
                size,
                self.data_ptr(),
                size,
                ACL_MEMCPY_DEVICE_TO_DEVICE,
            )
            if ret != 0:
                raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
            untyped = _NPUUntypedStorage(dst_ptr, size, device=self.device)
            return TypedStorage(untyped, self.dtype, self._size)
        if self.device.type == "meta":
            return meta_typed_storage_from_size(self._size, self.dtype, device=self.device)
        raise NotImplementedError(f"Unsupported device: {self.device}")

    def copy_(self, other):
        if self.device.type != other.device.type:
            raise NotImplementedError("cross-device copy_ not supported")
        if self.device.type == "cpu":
            np.copyto(self._data, other._data)
            return self
        if self.device.type == "npu":
            from ._backends.npu import runtime as npu_runtime

            size = min(self.nbytes(), other.nbytes())
            runtime = npu_runtime.get_runtime(self.device.index or 0)
            runtime.activate()
            ret = npu_runtime.acl.rt.memcpy(
                self.data_ptr(),
                size,
                other.data_ptr(),
                size,
                ACL_MEMCPY_DEVICE_TO_DEVICE,
            )
            if ret != 0:
                raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
            return self
        raise NotImplementedError(f"Unsupported device: {self.device}")

    def resize_(self, new_size):
        itemsize = np.dtype(to_numpy_dtype(self.dtype)).itemsize
        self._untyped.resize_(int(new_size) * itemsize)
        self._size = int(new_size)
        if self.device.type == "cpu":
            buf = self._untyped.buffer()
            self._data = np.frombuffer(buf, dtype=to_numpy_dtype(self.dtype), count=self._size)
        return self


class Storage(TypedStorage):
    pass


def typed_storage_from_numpy(arr, dtype, device=None):
    arr = np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    untyped = _CPUUntypedStorage(arr.view(np.uint8), device=device)
    return TypedStorage(untyped, dtype, arr.size, data=arr)


def empty_cpu_typed_storage(shape, dtype, device=None):
    arr = np.empty(shape, dtype=to_numpy_dtype(dtype))
    return typed_storage_from_numpy(arr, dtype, device=device)


def meta_typed_storage_from_shape(shape, dtype, device=None):
    size = int(np.prod(shape))
    return meta_typed_storage_from_size(size, dtype, device=device)


def meta_typed_storage_from_size(size, dtype, device=None):
    itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
    untyped = _MetaUntypedStorage(int(size) * itemsize, device=device)
    return TypedStorage(untyped, dtype, int(size))


def npu_typed_storage_from_ptr(device_ptr, size, dtype, device=None):
    itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
    untyped = _NPUUntypedStorage(device_ptr, int(size) * itemsize, device=device)
    return TypedStorage(untyped, dtype, int(size))


class PendingStorage:
    def __init__(self, shape, dtype, device):
        if isinstance(device, str):
            device = Device(device)
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        size = 1
        for d in self._shape:
            size *= d
        self._size = int(size)

    def size(self):
        return self._size

    def nbytes(self):
        itemsize = np.dtype(to_numpy_dtype(self.dtype)).itemsize
        return int(self._size * itemsize)

    def data_ptr(self):
        raise RuntimeError("pending tensor has no data")

    @property
    def data(self):
        raise RuntimeError(
            "PendingStorage has no data. Call flush() on the pipeline context "
            "to materialize the storage, or move the tensor to a device."
        )

    def untyped_storage(self):
        return self

    def is_shared(self):
        return False

    def is_pinned(self):
        return False



def pinned_cpu_typed_storage_from_numpy(arr, dtype, device=None):
    from ._backends.npu import runtime as npu_runtime

    arr = np.ascontiguousarray(arr, dtype=to_numpy_dtype(dtype))
    size = int(arr.nbytes)
    host_ptr = npu_runtime.alloc_host(size)
    buf = np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(int(host_ptr)))
    buf[:] = arr.view(np.uint8).reshape(-1)
    raw = np.frombuffer(buf, dtype=np.uint8)
    untyped = _PinnedCPUUntypedStorage(raw, host_ptr, device=device)
    data = np.frombuffer(raw, dtype=to_numpy_dtype(dtype), count=arr.size)
    return TypedStorage(untyped, dtype, arr.size, data=data)

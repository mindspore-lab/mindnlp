import numpy as np

from ._device import _default_device, device as Device
from ._dtype import float32, to_numpy_dtype
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


class UntypedStorage:
    def __init__(self, device):
        self.device = device

    def nbytes(self):
        raise NotImplementedError

    def data_ptr(self):
        raise NotImplementedError

    def resize_(self, new_nbytes):
        raise NotImplementedError

    def share_memory_(self):
        raise NotImplementedError

    def is_shared(self):
        return False

    @classmethod
    def from_file(cls, filename, shared=False):
        return _CPUUntypedStorage.from_file(filename, shared=shared)

    def filename(self):
        return None


class _CPUUntypedStorage(UntypedStorage):
    def __init__(self, array, filename=None, shared=False, device=None):
        super().__init__(device or Device("cpu"))
        self._array = array
        self._shared = shared
        self._filename = filename

    def nbytes(self):
        return int(self._array.nbytes)

    def data_ptr(self):
        return int(self._array.ctypes.data)

    def buffer(self):
        return self._array

    def resize_(self, new_nbytes):
        if self._filename is not None:
            raise NotImplementedError("file-backed storage cannot resize")
        if self._shared:
            raise NotImplementedError("shared storage cannot resize")
        new_array = np.empty(int(new_nbytes), dtype=np.uint8)
        old_bytes = self._array.view(np.uint8)
        copy_bytes = min(old_bytes.size, new_array.size)
        new_array[:copy_bytes] = old_bytes[:copy_bytes]
        self._array = new_array
        return self

    def share_memory_(self):
        self._shared = True
        return self

    def is_shared(self):
        return self._shared

    @classmethod
    def from_file(cls, filename, shared=False):
        data = np.memmap(filename, mode="r+", dtype=np.uint8)
        return cls(data, filename=filename, shared=shared)

    def filename(self):
        return self._filename


class _NPUUntypedStorage(UntypedStorage):
    def __init__(self, device_ptr, nbytes):
        super().__init__(Device("npu"))
        self._device_ptr = int(device_ptr)
        self._nbytes = int(nbytes)

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        return self._device_ptr


class _MetaUntypedStorage(UntypedStorage):
    def __init__(self, nbytes, device=None):
        super().__init__(device or Device("meta"))
        self._nbytes = int(nbytes)

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        raise RuntimeError("meta tensor has no data")


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
            npu_runtime._runtime.init(0)
            dst_ptr = npu_runtime._alloc_device(size)
            ret = npu_runtime.acl.rt.memcpy(
                dst_ptr,
                size,
                self.data_ptr(),
                size,
                ACL_MEMCPY_DEVICE_TO_DEVICE,
            )
            if ret != 0:
                raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
            untyped = _NPUUntypedStorage(dst_ptr, size)
            return TypedStorage(untyped, self.dtype, self._size)
        if self.device.type == "meta":
            return meta_typed_storage_from_size(self._size, self.dtype)
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
            npu_runtime._runtime.init(0)
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
        if self.device.type != "cpu":
            raise NotImplementedError("resize_ only supported on CPU storage")
        itemsize = np.dtype(to_numpy_dtype(self.dtype)).itemsize
        self._untyped.resize_(int(new_size) * itemsize)
        self._size = int(new_size)
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


def npu_typed_storage_from_ptr(device_ptr, size, dtype):
    itemsize = np.dtype(to_numpy_dtype(dtype)).itemsize
    untyped = _NPUUntypedStorage(device_ptr, int(size) * itemsize)
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

    def untyped_storage(self):
        return self

    def is_shared(self):
        return False

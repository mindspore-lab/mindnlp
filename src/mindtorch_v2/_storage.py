"""PyTorch-compatible Storage classes backed by MindSpore tensors.

Storage represents a contiguous 1D memory buffer. Multiple Tensor views
can share the same Storage. This is the foundation of PyTorch's view semantics.
"""

import numpy as np
import mindspore
from . import _dtype as dtype_mod


class UntypedStorage:
    """Raw byte-level storage. Equivalent to torch.UntypedStorage."""

    def __init__(self, size_bytes=0, *, device=None):
        from ._device import device as device_cls
        self._device = device_cls(device or "cpu")
        self._nbytes = size_bytes
        if size_bytes > 0:
            self._data = mindspore.numpy.zeros(size_bytes, dtype=mindspore.uint8)
        else:
            self._data = mindspore.numpy.zeros(1, dtype=mindspore.uint8)

    def nbytes(self):
        return self._nbytes

    @property
    def device(self):
        return self._device

    def data_ptr(self):
        return self._data.data_ptr()


class TypedStorage:
    """Element-typed storage. Equivalent to torch.TypedStorage."""

    def __init__(self, size_or_data=0, *, dtype=None, device=None):
        from ._device import device as device_cls

        if dtype is None:
            dtype = dtype_mod.float32
        self._dtype = dtype
        self._device = device_cls(device or "cpu")

        if isinstance(size_or_data, int):
            self._size = size_or_data
            ms_dtype = dtype.to_mindspore()
            if size_or_data > 0:
                self._ms_tensor = mindspore.ops.zeros(size_or_data, dtype=ms_dtype)
            else:
                self._ms_tensor = mindspore.Tensor([], dtype=ms_dtype)
        elif isinstance(size_or_data, mindspore.Tensor):
            if size_or_data.ndim != 1:
                size_or_data = size_or_data.reshape(-1)
            self._ms_tensor = size_or_data
            self._size = size_or_data.shape[0]
            self._dtype = dtype_mod.from_mindspore_dtype(size_or_data.dtype)
        else:
            raise TypeError(f"Unsupported type: {type(size_or_data)}")

    @classmethod
    def _from_numpy(cls, arr):
        """Create TypedStorage from numpy array."""
        arr = np.ascontiguousarray(arr).ravel()
        ms_tensor = mindspore.Tensor(arr)
        dtype = dtype_mod.from_mindspore_dtype(ms_tensor.dtype)
        storage = cls.__new__(cls)
        storage._ms_tensor = ms_tensor
        storage._size = arr.shape[0]
        storage._dtype = dtype
        from ._device import device as device_cls
        storage._device = device_cls("cpu")
        return storage

    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def nbytes(self):
        return self._size * self._dtype.itemsize

    def data_ptr(self):
        return self._ms_tensor.data_ptr()

    def __getitem__(self, idx):
        return self._ms_tensor[idx].asnumpy().item()

    def __setitem__(self, idx, value):
        np_data = self._ms_tensor.asnumpy()
        np_data[idx] = value
        self._ms_tensor = mindspore.Tensor(np_data)

    def __len__(self):
        return self._size

    def __repr__(self):
        return f"TypedStorage(size={self._size}, dtype={self._dtype})"

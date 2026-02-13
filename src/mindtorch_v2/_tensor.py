import numpy as np

from ._storage import (
    Storage,
    empty_cpu_typed_storage,
    meta_typed_storage_from_shape,
    npu_typed_storage_from_ptr,
    typed_storage_from_numpy,
)
from ._device import _default_device, device as Device
from ._dtype import float32, to_numpy_dtype
from ._functional import add, mul, matmul, relu, sum, reshape as reshape_dispatch
from ._functional import transpose as transpose_dispatch, view as view_dispatch, to as to_dispatch
from ._autograd.engine import backward as _backward
from ._printing import format_tensor


class Tensor:
    def __init__(self, storage, shape, stride, offset=0, requires_grad=False):
        self._storage = storage
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.offset = int(offset)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._pending = False

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def device(self):
        return self._storage.device

    def storage(self):
        return self._storage

    def untyped_storage(self):
        return self._storage.untyped_storage()

    def _numpy_view(self):
        if self.device.type == "meta":
            raise RuntimeError("meta tensor has no data")
        base = self._storage.data.ravel()
        itemsize = base.itemsize
        strides = tuple(s * itemsize for s in self.stride)
        return np.lib.stride_tricks.as_strided(
            base[self.offset:], shape=self.shape, strides=strides
        )

    def reshape(self, new_shape):
        return reshape_dispatch(self, new_shape)

    def view(self, new_shape):
        return view_dispatch(self, new_shape)

    def transpose(self, dim0, dim1):
        return transpose_dispatch(self, dim0, dim1)

    def _ones_like(self):
        if self.device.type == "meta":
            storage = meta_typed_storage_from_shape(self.shape, self.dtype, device=self.device)
            return Tensor(storage, self.shape, self.stride)
        arr = np.ones(self.shape, dtype=to_numpy_dtype(self.dtype))
        storage = typed_storage_from_numpy(arr, self.dtype, device=self.device if self.device.type == "cpu" else None)
        stride = tuple(np.array(arr.strides) // arr.itemsize)
        tensor = Tensor(storage, arr.shape, stride)
        if self.device.type != "cpu":
            return tensor.to(self.device)
        return tensor

    def numpy(self):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        if self.device.type == "meta":
            raise RuntimeError("meta tensor has no data")
        if self.device.type != "cpu":
            raise RuntimeError("numpy() is only available for CPU tensors")
        return self._numpy_view()

    def backward(self, gradient=None):
        _backward(self, gradient)

    def to(self, dev):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        return to_dispatch(self, dev)

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def matmul(self, other):
        return matmul(self, other)

    def relu(self):
        return relu(self)

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def __repr__(self):
        return format_tensor(self)

    def __str__(self):
        return format_tensor(self)

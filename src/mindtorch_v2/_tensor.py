import numpy as np

from ._storage import Storage
from ._device import _default_device
from ._dtype import float32, to_numpy_dtype


class Tensor:
    def __init__(self, storage, shape, stride, offset=0, requires_grad=False):
        self.storage = storage
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.offset = int(offset)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    @property
    def dtype(self):
        return self.storage.dtype

    @property
    def device(self):
        return self.storage.device

    def reshape(self, new_shape):
        size = 1
        for d in self.shape:
            size *= d
        new_size = 1
        for d in new_shape:
            new_size *= d
        if size != new_size:
            raise ValueError("reshape size mismatch")
        new_stride = []
        acc = 1
        for d in reversed(new_shape):
            new_stride.append(acc)
            acc *= d
        new_stride = tuple(reversed(new_stride))
        return Tensor(self.storage, new_shape, new_stride, self.offset, self.requires_grad)

    def transpose(self, dim0, dim1):
        shape = list(self.shape)
        stride = list(self.stride)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
        return Tensor(self.storage, shape, stride, self.offset, self.requires_grad)


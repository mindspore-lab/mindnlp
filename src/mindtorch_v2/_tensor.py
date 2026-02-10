import numpy as np

from ._storage import Storage
from ._device import _default_device, device as Device
from ._dtype import float32, to_numpy_dtype
from ._functional import add, mul, matmul, relu, sum
from ._autograd.engine import backward as _backward


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

    def _numpy_view(self):
        base = self.storage.data.ravel()
        itemsize = base.itemsize
        strides = tuple(s * itemsize for s in self.stride)
        return np.lib.stride_tricks.as_strided(
            base[self.offset:], shape=self.shape, strides=strides
        )

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

    def _ones_like(self):
        arr = np.ones(self.shape, dtype=to_numpy_dtype(self.dtype))
        storage = Storage(arr, device=self.device, dtype=self.dtype)
        stride = tuple(np.array(arr.strides) // arr.itemsize)
        return Tensor(storage, arr.shape, stride)

    def backward(self, gradient=None):
        _backward(self, gradient)

    def to(self, dev):
        if isinstance(dev, str):
            dev = Device(dev)
        new_storage = self.storage.to(dev)
        return Tensor(new_storage, self.shape, self.stride, self.offset, self.requires_grad)

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

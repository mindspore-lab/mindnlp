import math
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.tensor import _TensorMeta
from mindspore._c_expression.typing import Type
try:
    from mindspore.common._stub_tensor import StubTensor
except:
    class StubTensor: pass

try:
    from mindspore._c_expression import TensorPy as Tensor_
except:
    from mindspore._c_expression import Tensor as Tensor_

from . import ops, _dtype
from ._dtype import dtype2np
from ._bind import get_default_device, device_
from .configs import use_pyboost, ON_A1
from .storage import UntypedStorage
from ._utils import _rebuild_tensor_v2

DTYPE_ELEMENT_SIZE_MAP = {
    mindspore.float64: 8,
    mindspore.int64: 8,
    mindspore.int32: 4,
    mindspore.float32: 4,
    mindspore.int16: 2,
    mindspore.bfloat16: 2,
    mindspore.float16: 2,
}

DEVICE_MAP = {
    'GPU': 'cuda',
    'Ascend': 'npu',
    'CPU': 'cpu'
}

class TypedTensorMeta(_TensorMeta):
    def __isinstancecheck__(self, instance):
        if not isinstance(instance, Tensor):
            return False
        return instance.dtype == self.dtype

class IntTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.int
    def __init__(self, data, device=None):
        super().__init__(data, dtype=_dtype.int)

class LongTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.long
    def __init__(self, data, device=None):
        super().__init__(data, dtype=_dtype.long)

class FloatTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float32
    def __init__(self, data, device=None):
        super().__init__(data, dtype=_dtype.float32)


class HalfTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float16
    def __init__(self, data, device=None):
        super().__init__(data, dtype=_dtype.float16)

class BFloat16Tensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float16
    def __init__(self, data, device=None):
        super().__init__(data, dtype=_dtype.bfloat16)


class BoolTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.bool
    def __init__(self, data, device=None):
        super().__init__(data, dtype=_dtype.bool)

def tensor(data, *, dtype=None, device=None, requires_grad=False):
    if isinstance(data, Tensor):
        UserWarning("To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than core.tensor(sourceTensor).")
        return Tensor(data)

    if device is None:
        device = get_default_device()

    if dtype is not None:
        tensor = Tensor(data, dtype=dtype)
    else:
        tensor = Tensor(data)

    tensor = tensor.to(device)
    tensor.requires_grad_(requires_grad)
    return tensor

def is_tensor(x):
    return isinstance(x, Tensor)

def enable_mindspore_patch():
    def __reduce_ex__(self, protocol):
        if isinstance(self, StubTensor):
            data = Tensor_(self.stub_sync())
        else:
            data = Tensor_(self)
        storage_offset = 0
        size = data._shape
        stride = data.stride()
        requires_grad = False
        args = (data, storage_offset, size, stride, requires_grad, None, None)
        return (
            _rebuild_from_type_v2, (_rebuild_tensor_v2, type(self), args, None))

    Tensor.__reduce_ex__ = __reduce_ex__
    StubTensor.__reduce_ex__ = __reduce_ex__

    def to_(self, *args, **kwargs):
        dtype_to = None
        if len(args) == 1:
            if isinstance(args[0], Type):
                dtype_to = args[0]
            elif isinstance(args[0], Tensor):
                dtype_to = args[0].dtype
        elif len(args) == 2:
            _, dtype_to = args
        else:
            dtype_to = kwargs.get("dtype", None)
        if dtype_to is not None:
            return mindspore.ops.cast(self, dtype_to)
        return self

    Tensor.to = to_
    StubTensor.to = to_

    def size(self, dim=None):
        if dim is None:
            return self.shape
        assert isinstance(dim, int), f'`dim` must be int but got {type(dim)}'
        return self.shape[dim]

    Tensor.size = size
    StubTensor.size = size

    @property
    def is_meta(self):
        return False

    Tensor.is_meta = is_meta
    StubTensor.is_meta = is_meta

    def data_ptr(self):
        return self._data_ptr()
    
    Tensor.data_ptr = data_ptr
    StubTensor.data_ptr = data_ptr

    Tensor.device = device_(DEVICE_MAP[mindspore.get_context('device_target')])
    StubTensor.device = device_(DEVICE_MAP[mindspore.get_context('device_target')])

    def _expand(self, *size):
        if len(size) == 1:
            size = size[0]
        return self.broadcast_to(size)

    Tensor.expand = _expand
    StubTensor.expand = _expand

    def clone(self, *args, **kwargs):
        return self.copy()
    
    Tensor.clone = clone
    StubTensor.clone = clone

    def _repeat(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = sizes[0]
        return ops.tile(self, tuple(sizes))

    Tensor.repeat = _repeat
    StubTensor.repeat = _repeat

    def __or__(self, other):
        if isinstance(other, (int, bool, float, Tensor)):
            return ops.bitwise_or(self, other)
        raise TypeError("Unsupported operand type(s) for |: 'Tensor' and '{}'".format(type(other)))

    Tensor.__or__ = __or__
    StubTensor.__or__ = __or__

    def __and__(self, other):
        if isinstance(other, (int, bool, float, Tensor)):
            return ops.bitwise_and(self, other)
        raise TypeError("Unsupported operand type(s) for &: 'Tensor' and '{}'".format(type(other)))

    Tensor.__and__ = __and__
    StubTensor.__and__ = __and__

    def detach(self):
        return ops.stop_gradient(self)

    Tensor.detach = detach
    StubTensor.detach = detach

    origin_getitem = Tensor.__getitem__
    def __getitem__(self, slices):
        # if 0 in self.shape:
        #     return self
        if isinstance(slices, tuple):
            new_slices = ()
            for s in slices:
                if isinstance(s, range):
                    s = list(s)
                new_slices += (s,)
            slices = new_slices
        return origin_getitem(self, slices)

    Tensor.__getitem__ = __getitem__
    StubTensor.__getitem__ = __getitem__

    def numel(self):
        return math.prod(self.shape)

    Tensor.numel = numel
    StubTensor.numel = numel
    Tensor.nelement = numel
    StubTensor.nelement = numel

    @property
    def nbytes(self):
        return self.numel() * self.element_size()

    Tensor.nbytes = nbytes
    StubTensor.nbytes = nbytes

    Tensor.normal_ = ops.inplace_normal
    StubTensor.normal_ = ops.inplace_normal


    Tensor.softmax = ops.softmax
    StubTensor.softmax = ops.softmax

    Tensor.squeeze = ops.squeeze
    StubTensor.squeeze = ops.squeeze

    Tensor.unsqueeze = ops.unsqueeze
    StubTensor.unsqueeze = ops.unsqueeze

    def log_softmax(self, dim=-1, dtype=None):
        if use_pyboost():
            return mindspore.mint.nn.functional.log_softmax(self, dim=dim, dtype=dtype)
        out = mindspore.ops.log_softmax(self, dim)
        if dtype is not None:
            out = out.to(dtype)
        return out

    Tensor.log_softmax = log_softmax
    StubTensor.log_softmax = log_softmax

    def untyped_storage(self):
        return UntypedStorage(self)
    
    Tensor.untyped_storage = untyped_storage
    StubTensor.untyped_storage = untyped_storage

    def element_size(self,):
        return DTYPE_ELEMENT_SIZE_MAP[self.dtype]

    Tensor.element_size = element_size
    StubTensor.element_size = element_size

    @property
    def layout(self):
        return None

    Tensor.layout = layout
    StubTensor.layout = layout

    def __add__(self, other):
        # if 0 in self.shape:
        #     return self
        return ops.add(self, other)
    
    Tensor.__add__ = __add__
    StubTensor.__add__ = __add__

    Tensor.repeat_interleave = ops.repeat_interleave
    StubTensor.repeat_interleave = ops.repeat_interleave

    def dim(self):
        return self.ndim

    Tensor.dim = dim
    StubTensor.dim = dim

    def unfold(self, dimension, size, step):
        return ops.unfold(self, dimension, size, step)

    Tensor.unfold = unfold
    StubTensor.unfold = unfold

    def new(self, *shape):
        return ops.empty(*shape, dtype=self.dtype)

    Tensor.new = new
    StubTensor.new = new

    def view(self, *args):
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        return self.reshape(*args)
    
    Tensor.view = view
    StubTensor.view = view

    def cpu(self):
        return self

    Tensor.cpu = cpu
    StubTensor.cpu = cpu

    Tensor.take = ops.take
    StubTensor.take = ops.take

    Tensor.sort = ops.sort
    StubTensor.sort = ops.sort

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    Tensor.requires_grad_ = requires_grad_
    StubTensor.requires_grad_ = requires_grad_

    @property
    def data(self):
        return Tensor(self)

    @data.setter
    def data(self, new_value):
        if isinstance(self, StubTensor) and isinstance(new_value, StubTensor):
            self.stub = new_value.stub
        else:
            self.assign_value(new_value)

    Tensor.data = data
    StubTensor.data = data

    Tensor.narrow = ops.narrow
    StubTensor.narrow = ops.narrow


def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    return ret
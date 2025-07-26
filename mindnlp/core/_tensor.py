import math
import numpy as np
import warnings
import mindspore
from mindspore import Tensor
from mindspore.common.tensor import _TensorMeta
from mindspore._c_expression.typing import Type
try:
    from mindspore.common._stub_tensor import StubTensor, _stub_method
except:
    class StubTensor: pass

try:
    from mindspore._c_expression import TensorPy as Tensor_
except:
    from mindspore._c_expression import Tensor as Tensor_

from . import ops, _dtype
from ._dtype import dtype2np
from ._bind import get_default_device, device_, get_default_dtype
from .configs import use_pyboost, ON_A1
from .storage import UntypedStorage
from ._utils import _rebuild_tensor_v2
from ._C.size import Size

DTYPE_ELEMENT_SIZE_MAP = {
    mindspore.float64: 8,
    mindspore.int64: 8,
    mindspore.int32: 4,
    mindspore.float32: 4,
    mindspore.int16: 2,
    mindspore.bfloat16: 2,
    mindspore.float16: 2,
    mindspore.bool_: 1
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=_dtype.int, **kwargs)

class LongTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.long
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=_dtype.long, **kwargs)

class FloatTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float32
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=_dtype.float32, **kwargs)

class HalfTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float16
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=_dtype.float16, **kwargs)

class BFloat16Tensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float16
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=_dtype.bfloat16, **kwargs)

class BoolTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.bool
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=_dtype.bool, **kwargs)


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    if isinstance(data, Tensor):
        UserWarning("To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than core.tensor(sourceTensor).")
        return Tensor(data)

    if isinstance(data, list):
        new_data = []
        for d in data:
            if isinstance(d, Tensor):
                d = d.item()
            new_data.append(d)
        data = new_data

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

    def tensor_meta_str(self):
        return "<class 'torch.Tensor'>"

    _TensorMeta.__str__ = tensor_meta_str

    old_init = Tensor.__init__
    def __init__(self, *args, **kwargs):
        if len(args) > 1 and all([isinstance(arg, int) for arg in args]):
            tensor = Tensor_(shape=args, dtype=get_default_dtype())
            old_init(self, tensor, internal=True)
        else:
            old_init(self, *args, **kwargs)

    Tensor.__init__ = __init__

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
        dtype_to = kwargs.get("dtype", None)
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
    def shape(self):
        if isinstance(self, StubTensor):
            if self.stub is not None:
                stub_shape = self.stub.get_shape()
            else:
                stub_shape = self.tensor.shape
            return Size(stub_shape)
        return Size(self._shape)

    Tensor.shape = shape
    StubTensor.shape = shape

    @property
    def is_meta(self):
        return False

    Tensor.is_meta = is_meta
    StubTensor.is_meta = is_meta

    def data_ptr(self):
        ptr = self._data_ptr()
        if ptr != 0:
            return ptr
        self + 1
        return self._data_ptr()
    
    Tensor.data_ptr = data_ptr
    StubTensor.data_ptr = data_ptr

    Tensor.device = device_(DEVICE_MAP[mindspore.get_context('device_target')])
    StubTensor.device = device_(DEVICE_MAP[mindspore.get_context('device_target')])

    def _expand(self, *size):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        new_size = ()
        for s in size:
            if isinstance(s, Tensor):
                s = s.item()
            new_size += (s,)
        return self.broadcast_to(new_size)

    Tensor.expand = _expand
    StubTensor.expand = _expand

    Tensor.broadcast_to = ops.broadcast_to
    StubTensor.broadcast_to = ops.broadcast_to

    def clone(self, *args, **kwargs):
        return self.copy()
    
    Tensor.clone = clone
    StubTensor.clone = clone

    def _repeat(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = sizes[0]
        new_sizes = ()
        for s in sizes:
            if not isinstance(s, int):
                s = s.item()
            new_sizes += (s,)

        return ops.tile(self, new_sizes)

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
        slices = self._convert_numpy_slices(slices)
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

    def _convert_numpy_slices(self, key):
        """递归转换 key 中的 NumPy 整数为内置 int"""
        # 处理元组：遍历所有元素并递归转换
        if isinstance(key, tuple):
            return tuple(self._convert_numpy_slices(k) for k in key)
        
        # 处理 slice 对象：转换 start/stop/step
        elif isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step
            
            # 转换 NumPy 整数为 Python int
            if isinstance(start, np.integer):
                start = int(start)
            if isinstance(stop, np.integer):
                stop = int(stop)
            if isinstance(step, np.integer):
                step = int(step)
            
            return slice(start, stop, step)
        
        # 转换单个 NumPy 索引值
        elif isinstance(key, np.integer):
            return int(key)
        
        # 其他类型（如 int、None）直接返回
        else:
            return key

    Tensor._convert_numpy_slices = _convert_numpy_slices
    StubTensor._convert_numpy_slices = _convert_numpy_slices

    origin_setitem = Tensor.__setitem__
    def __setitem__(self, slices, value):
        slices = self._convert_numpy_slices(slices)
        if isinstance(value, float):
            if value == float('inf'):
                value = ops.finfo(self.dtype).max
            elif value == -float('inf'):
                value = ops.finfo(self.dtype).min
        if isinstance(slices, tuple):
            new_slices = ()
            for s in slices:
                if isinstance(s, range):
                    s = list(s)
                new_slices += (s,)
            slices = new_slices
        if not isinstance(value, Tensor):
            value = tensor(value, dtype=self.dtype)
        else:
            value = value.to(self.dtype)

        if 1 in value.shape and self[slices].ndim != value.ndim:
            value = value.squeeze()
        
        return origin_setitem(self, slices, value)

    Tensor.__setitem__ = __setitem__
    StubTensor.__setitem__ = __setitem__

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

    def __sub__(self, other):
        # if 0 in self.shape:
        #     return self
        if isinstance(other, (np.ndarray, np.integer)):
            other = tensor(other)
        return ops.sub(self, other)
    
    Tensor.__sub__ = __sub__
    StubTensor.__sub__ = __sub__


    def __mul__(self, other):
        # if 0 in self.shape:
        #     return self
        if isinstance(other, (np.ndarray, np.integer)):
            other = tensor(other)
        return ops.mul(self, other)
    
    Tensor.__mul__ = __mul__
    StubTensor.__mul__ = __mul__


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
        if not isinstance(shape[0], int):
            return tensor(shape[0], dtype=self.dtype)
        return ops.empty(*shape, dtype=self.dtype)

    Tensor.new = new
    StubTensor.new = new

    def view(self, *args):
        return self.reshape(*args)
    
    Tensor.view = view
    StubTensor.view = view

    def cpu(self, *args, **kwargs):
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

    def bitwise_or_(self, other):
        out = ops.bitwise_or(self, other)
        self.copy_(out)
        return self
    
    Tensor.bitwise_or_ = bitwise_or_
    StubTensor.bitwise_or_ = bitwise_or_

    # fix TypeError: unhashable type: 'StubTensor'
    StubTensor.__hash__ = Tensor.__hash__

    Tensor.masked_fill = ops.masked_fill
    StubTensor.masked_fill = ops.masked_fill

    Tensor.reshape = ops.reshape
    StubTensor.reshape = ops.reshape

    def __rmul__(self, other):
        if isinstance(other, (str, list)):
            return self.item() * other
        return self.__mul__(other)

    Tensor.__rmul__ = __rmul__
    StubTensor.__rmul__ = __rmul__

    Tensor.norm = ops.norm
    StubTensor.norm = ops.norm

    def clamp_min(self, value):
        return ops.clamp(self, value)

    Tensor.clamp_min = clamp_min
    StubTensor.clamp_min = clamp_min

    Tensor.index_copy_ = ops.inplace_index_copy
    StubTensor.index_copy_ = ops.inplace_index_copy

    Tensor.max = ops.max
    StubTensor.max = ops.max

    Tensor.min = ops.min
    StubTensor.min = ops.min

    Tensor.squeeze_ = ops.inplace_squeeze
    StubTensor.squeeze_ = ops.inplace_squeeze

    Tensor.unsqueeze_ = ops.inplace_unsqueeze
    StubTensor.unsqueeze_ = ops.inplace_unsqueeze

    def pin_memory(self, *args, **kwargs):
        return self
    
    Tensor.pin_memory = pin_memory
    StubTensor.pin_memory = pin_memory

    def __deepcopy__(self, memodict):
        new_obj = Tensor(self)
        return new_obj

    Tensor.__deepcopy__ = __deepcopy__
    StubTensor.__deepcopy__ = __deepcopy__

    def asnumpy(self):
        return Tensor_.asnumpy(self)

    Tensor.asnumpy = asnumpy
    StubTensor.asnumpy = _stub_method(asnumpy)

    def backward(self, *args, **kwargs):
        pass

    Tensor.backward = backward
    StubTensor.backward = backward

    def __repr__(self):
        Tensor_.data_sync(self, True)
        return Tensor_.__repr__(self)

    Tensor.__repr__ = __repr__
    StubTensor.__repr__ = _stub_method(__repr__)

    def detach_(self):
        return ops.stop_gradient(self)

    Tensor.detach_ = detach_
    StubTensor.detach_ = detach_

    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
        return ops.full(size, fill_value, dtype=dtype if dtype is not None else self.dtype)

    Tensor.new_full = new_full
    StubTensor.new_full = new_full

    def new_zeros(self, *size, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
        if isinstance(size[0], (tuple, list)):
            size = size[0]

        new_size = ()
        for s in size:
            if isinstance(s, Tensor):
                s = s.item()
            new_size += (s,)
        return ops.zeros(*new_size, dtype=dtype if dtype is not None else self.dtype)

    Tensor.new_zeros = new_zeros
    StubTensor.new_zeros = new_zeros

    def new_ones(self, *size, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False, **kwargs):
        size = kwargs.get('size', size)
        if isinstance(size[0], (tuple, list)):
            size = size[0]

        new_size = ()
        for s in size:
            if isinstance(s, Tensor):
                s = s.item()
            new_size += (s,)
        return ops.ones(*new_size, dtype=dtype if dtype is not None else self.dtype)

    Tensor.new_ones = new_ones
    StubTensor.new_ones = new_ones

    Tensor.sum = ops.sum
    StubTensor.sum = ops.sum

    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
        return tensor(data, dtype=dtype if dtype is not None else self.dtype)

    Tensor.new_tensor = new_tensor
    StubTensor.new_tensor = new_tensor

    Tensor.fill_diagonal_ = ops.inplace_fill_diagonal
    StubTensor.fill_diagonal_ = ops.inplace_fill_diagonal

    Tensor.triu_ = ops.inplace_triu
    StubTensor.triu_ = ops.inplace_triu

    @property
    def real(self):
        return ops.real(self)
    
    Tensor.real = real
    StubTensor.real = real

    @property
    def imag(self):
        return ops.imag(self)

    Tensor.imag = imag
    StubTensor.imag = imag

    def bfloat16(self):
        return self.to(_dtype.bfloat16)

    Tensor.bfloat16 = bfloat16
    StubTensor.bfloat16 = bfloat16

    def sort(self, dim=-1, descending=False):
        return ops.sort(self, dim=dim, descending=descending)

    Tensor.sort = sort
    StubTensor.sort = sort

    Tensor.cumsum = ops.cumsum
    StubTensor.cumsum = ops.cumsum

    Tensor.scatter_ = ops.inplace_scatter
    StubTensor.scatter_ = ops.inplace_scatter

    def __contains__(self, item):
        return ops.eq(self, item).any()

    Tensor.__contains__ = __contains__
    StubTensor.__contains__ = __contains__

    Tensor.tile = ops.tile
    StubTensor.tile = ops.tile

    Tensor.mean = ops.mean
    StubTensor.mean = ops.mean

    Tensor.amax = ops.amax
    StubTensor.amax = ops.amax

    Tensor.as_strided = ops.as_strided
    StubTensor.as_strided = ops.as_strided

    Tensor.split = ops.split
    StubTensor.split = ops.split

    Tensor.flip = ops.flip
    StubTensor.flip = ops.flip

    Tensor.unflatten = ops.unflatten
    StubTensor.unflatten = ops.unflatten

    Tensor.round_ = ops.inplace_round
    StubTensor.round_ = ops.inplace_round

    Tensor.split_with_sizes = ops.split_with_sizes
    StubTensor.split_with_sizes = ops.split_with_sizes

    Tensor.scatter_reduce_ = ops.inplace_scatter_reduce
    StubTensor.scatter_reduce_ = ops.inplace_scatter_reduce

    Tensor.exponential_ = ops.inplace_exponential
    StubTensor.exponential_ = ops.inplace_exponential

    Tensor.log_ = ops.inplace_log
    StubTensor.log_ = ops.inplace_log

    Tensor.mul_ = ops.inplace_mul
    StubTensor.mul_ = ops.inplace_mul

    Tensor.neg_ = ops.inplace_neg
    StubTensor.neg_ = ops.inplace_neg

    Tensor.exp_ = ops.inplace_exp
    StubTensor.exp_ = ops.inplace_exp

    Tensor.sub_ = ops.inplace_sub
    StubTensor.sub_ = ops.inplace_sub

    Tensor.roll = ops.roll
    StubTensor.roll = ops.roll

    Tensor.bernoulli_ = ops.inplace_bernoulli
    StubTensor.bernoulli_ = ops.inplace_bernoulli

    Tensor.scatter_reduce = ops.scatter_reduce
    StubTensor.scatter_reduce = ops.scatter_reduce

    Tensor.tril_ = ops.inplace_tril
    StubTensor.tril_ = ops.inplace_tril

    Tensor.var = ops.var
    StubTensor.var = ops.var

    Tensor.logsumexp = ops.logsumexp
    StubTensor.logsumexp = ops.logsumexp

    def __iter__(self):
        if self.ndim == 0:
            yield self
        else:
            for i in range(len(self)):
                yield self[i]

    Tensor.__iter__ = __iter__
    StubTensor.__iter__ = __iter__

    def __float__(self):
        out = self.item()
        return round(float(out), 5)

    Tensor.__float__ = __float__
    StubTensor.__float__ = __float__

def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    return ret
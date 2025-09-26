import gc
import math
import ctypes
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.tensor import _TensorMeta
from mindspore._c_expression.typing import Type
from mindspore._c_expression import ParamInfo # pylint: disable=no-name-in-module
from mindspore._c_expression import typing
try:
    from mindspore.common._stub_tensor import StubTensor, _stub_method
except:
    class StubTensor: pass

try:
    from mindspore._c_expression import TensorPy as Tensor_
except:
    from mindspore._c_expression import Tensor as Tensor_

import mindtorch
from . import ops, _dtype
from ._bind import get_device_in_context, device_, get_default_dtype
from ._utils import _rebuild_tensor_v2
from ._C.size import Size
from .configs import DEVICE_TARGET, cpu_use_numpy

device_map = {
    'cpu': 'CPU',
    'npu': 'Ascend',
    'cuda': 'GPU'
}

if DEVICE_TARGET == 'Ascend':
    import acl
else:
    acl = None

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

class TypedTensorMeta(_TensorMeta):
    def __isinstancecheck__(self, instance):
        if not isinstance(instance, Tensor):
            return False
        return instance.dtype == self.dtype

class IntTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.int
    def __init__(self, *args, **kwargs):
        self._device = kwargs.pop('device', device_('cpu'))
        super().__init__(*args, dtype=_dtype.int, **kwargs)

class LongTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.long
    def __init__(self, *args, **kwargs):
        self._device = kwargs.pop('device', device_('cpu'))
        super().__init__(*args, dtype=_dtype.long, **kwargs)

class FloatTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float32
    def __init__(self, *args, **kwargs):
        self._device = kwargs.pop('device', device_('cpu'))
        super().__init__(*args, dtype=_dtype.float32, **kwargs)

class HalfTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float16
    def __init__(self, *args, **kwargs):
        self._device = kwargs.pop('device', device_('cpu'))
        super().__init__(*args, dtype=_dtype.float16, **kwargs)

class BFloat16Tensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.float16
    def __init__(self, *args, **kwargs):
        self._device = kwargs.pop('device', device_('cpu'))
        super().__init__(*args, dtype=_dtype.bfloat16, **kwargs)

class BoolTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = _dtype.bool
    def __init__(self, *args, **kwargs):
        self._device = kwargs.pop('device', device_('cpu'))
        super().__init__(*args, dtype=_dtype.bool, **kwargs)


def tensor_meta_str(self):
    return "<class 'torch.Tensor'>"

_TensorMeta.__str__ = tensor_meta_str

old_init = Tensor.__init__
def __init__(self, *args, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    device = kwargs.pop('device', mindtorch.get_default_device())
    if len(args) > 1 and all([isinstance(arg, int) for arg in args]):
        # tensor = Tensor_(shape=args, dtype=get_default_dtype())
        # Tensor_(self, tensor)
        old_init(self, shape=args, dtype=get_default_dtype())
    else:
        old_init(self, *args, **kwargs)
    if requires_grad:
        self.requires_grad_(requires_grad)
    self._device = device

Tensor.__init__ = __init__
origin_setitem = Tensor.__setitem__
origin_is_contiguous = Tensor.is_contiguous
Tensor._requires_grad = False

def tensor(data, *, dtype=None, device=None, requires_grad=False):
    if isinstance(data, Tensor):
        UserWarning("To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than mindtorch.tensor(sourceTensor).")
        out = Tensor(data)
        if device is not None:
            out._device = device
        else:
            out._device = data.device

        return out

    # if isinstance(data, list):
    #     new_data = []
    #     for d in data:
    #         if isinstance(d, Tensor):
    #             d = d.item()
    #         new_data.append(d)
    #     data = new_data

    if device is None:
        device = get_device_in_context()

    if isinstance(device, (str, int)):
        device = device_(device)

    if isinstance(data, float) and data == float('-inf'):
        data = mindtorch.finfo(get_default_dtype()).min
    elif isinstance(data, list) and float('-inf') in data:
        data = [mindtorch.finfo(get_default_dtype()).min if d == float('-inf') else d for d in data]

    if dtype is not None:
        tensor = Tensor(data, dtype=dtype)
    else:
        tensor = Tensor(data)

    tensor._device = device
    if DEVICE_TARGET == 'Ascend' and device.type == 'cuda':
        device.type = 'npu'
    if device.type not in ['meta', 'cpu']:
        tensor = tensor.to(device)
    if requires_grad:
        tensor.requires_grad_(requires_grad)
    return tensor

def scalar_tensor(*args, **kwargs):
    return tensor(*args, **kwargs)

def is_tensor(x):
    return isinstance(x, Tensor)

class TensorPlaceHolder:

    def cpu(self):
        return self.to(device_('cpu'))

    def npu(self, device=None, non_blocking=False):
        if device is None:
            device = device_('npu', 0)
        return self.to(device, non_blocking=non_blocking)

    def cuda(self, device=None, non_blocking=False):
        if DEVICE_TARGET == 'Ascend':
            return self.npu(device, non_blocking)
        if device is None:
            device = device_('cuda', 0)
        return self.to(device, non_blocking=non_blocking)

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    def __array_wrap__(self, array):
        if array.dtype == bool:
            # Workaround, torch has no built-in bool tensor
            array = array.astype("uint8")
        return ops.from_numpy(array)

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

    def __hash__(self):
        return hash(id(self))

    def __len__(self):
        if self.shape == ():
            return 1
        return self.shape[0]

    def __repr__(self) -> str:
        self.data_sync(True)
        return Tensor_.__repr__(self)[:-1] + f', device={self.device})'

    def __format__(self, format_spec):
        return np.ndarray.__format__(self.asnumpy(), format_spec)

    def __iter__(self):
        if self.ndim == 0:
            yield self
        else:
            for i in range(len(self)):
                yield self[i]

    def __getitem__(self, slices):
        slices = self._convert_numpy_slices(slices)
        # if 0 in self.shape:
        #     return self
        if isinstance(slices, tuple):
            new_slices = ()
            for s in slices:
                if isinstance(s, range):
                    s = list(s)
                if isinstance(s, np.ndarray):
                    s = tensor(s, device=self.device)
                new_slices += (s,)
            slices = new_slices

        if self.device.type == 'meta':
            out = ops.getitem_np(self, slices)
        else:
            out = ops.tensor_getitem(self, slices)

        out._device = self.device
        return out

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
            value = tensor(value, dtype=self.dtype, device=self.device)
        else:
            value = value.to(self.dtype)

        if 1 in value.shape and self[slices].ndim != value.ndim:
            value = value.squeeze()

        if self.device.type == 'meta':
            return self

        if value.device != self.device:
            value._device = self.device

        if self.device.type == 'npu':
            if value.device != self.device:
                value._device = self.device
            out = ops.tensor_setitem(self, slices, value)
        else:
            out = ops.setitem(self, slices, value)
        return self

    def __add__(self, other):
        # if 0 in self.shape:
        #     return self
        return ops.add(self, other)

    def __iadd__(self, other):
        return self.copy_(ops.add(self, other))

    def __radd__(self, other):
        return Tensor.__add__(other, self)

    def __div__(self, other):
        # if 0 in self.shape:
        #     return self
        if isinstance(other, (np.ndarray, np.integer)):
            other = tensor(other)
        return ops.div(self, other)

    def __rshift__(self, other):
        return ops.bitwise_right_shift(self, other)

    def __rtruediv__ (self, other):
        return ops.div(other, self)

    def __ne__(self, other):
        if isinstance(other, list):
            return True
        return ops.ne(self, other)

    def __neg__(self):
        return ops.neg(self)

    def __mul__(self, other):
        # if 0 in self.shape:
        #     return self
        if isinstance(other, (np.ndarray, np.integer)):
            other = tensor(other)
        return ops.mul(self, other)

    def __rmul__(self, other):
        if isinstance(other, (str, list)):
            return self.item() * other
        return self.__mul__(other)

    def __abs__(self):
        return ops.abs(self)

    def __imul__(self, other):
        return self.copy_(ops.mul(self, other))

    def __itruediv__(self, other):
        return self.copy_(ops.div(self, other))

    def __pow__(self, other):
        return ops.pow(self, other)

    def __rpow__(self, other):
        return ops.pow(other, self)

    def __sub__(self, other):
        # if 0 in self.shape:
        #     return self
        if isinstance(other, (np.ndarray, np.integer)):
            other = tensor(other)
        return ops.sub(self, other)

    def __isub__(self, other):
        return self.copy_(ops.sub(self, other))

    def __rsub__(self, other):
        return ops.sub(other, self)

    def __eq__(self, other):
        if other is None:
            return False
        return ops.eq(self, other)

    def __gt__(self, other):
        return ops.gt(self, other)

    def __ge__(self, other):
        return ops.ge(self, other)

    def __lt__(self, other):
        return ops.lt(self, other)

    def __le__(self, other):
        return ops.le(self, other)

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        return bool(self.item())

    def __index__(self):
        # if self.ndim > 0:
        #     return self.tolist()
        return int(self.item())

    def __and__(self, other):
        return ops.bitwise_and(self, other)

    def __xor__(self, other):
        return ops.bitwise_xor(self, other)

    def __or__(self, other):
        return ops.bitwise_or(self, other)

    def __invert__(self):
        return ops.logical_not(self)

    def __round__(self):
        return ops.round(self)

    def new(self, *shape):
        if not isinstance(shape[0], int):
            return tensor(shape[0], dtype=self.dtype, device=self.device)
        return ops.empty(*shape, dtype=self.dtype, device=self.device)

    # Tensor.new_tensor
    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        return tensor(data, dtype=dtype, device=device)

    # Tensor.new_full
    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return ops.full(size, fill_value, dtype=dtype, device=device)

    # Tensor.new_empty
    def new_empty(self, size, *, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return ops.empty(*size, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

    # Tensor.new_ones
    def new_ones(self, *size, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False, **kwargs):
        size = kwargs.get('size', size)
        if isinstance(size[0], (tuple, list)):
            size = size[0]

        new_size = ()
        for s in size:
            if isinstance(s, Tensor):
                s = s.item()
            new_size += (s,)
        if new_size == new_size:
            new_size = (new_size,)
        return ops.ones(*new_size, dtype=dtype if dtype is not None else self.dtype, device=self.device)


    # Tensor.new_zeros
    def new_zeros(self, *size, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
        if isinstance(size[0], (tuple, list)):
            size = size[0]

        new_size = ()
        for s in size:
            if isinstance(s, Tensor):
                s = s.item()
            new_size += (s,)
        return ops.zeros(
            *new_size,
            dtype=dtype if dtype is not None else self.dtype,
            device=device if device is not None else self.device
        )

    # Tensor.ndim
    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    def ndimension(self):
        return self.ndim

    # Tensor.real
    @property
    def real(self):
        return ops.real(self)

    # Tensor.imag
    @property
    def imag(self):
        return ops.imag(self)

    # Tensor.nbytes
    @property
    def nbytes(self):
        return self.numel() * self.element_size()

    # Tensor.itemsize
    @property
    def itemsize(self):
        return self._data._itemsize

    # Tensor.abs
    def abs(self):
        return ops.abs(self)

    # Tensor.abs_
    def abs_(self):
        return self.copy_(ops.abs(input))

    # Tensor.absolute
    absolute = abs

    # Tensor.absolute_
    absolute_ = abs_

    # Tensor.acos
    def acos(self):
        return ops.acos(self)

    # Tensor.acos_
    def acos_(self):
        return self.copy_(ops.acos(input))

    # Tensor.arccos
    arccos = acos

    # Tensor.arccos_
    arccos_ = acos_

    # Tensor.add
    def add(self, other, *, alpha=1):
        return ops.add(self, other, alpha=alpha)

    # Tensor.add_
    def add_(self, other, *, alpha=1):
        return ops.inplace_add(self, other, alpha=alpha)

    # Tensor.addbmm
    def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return ops.addbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    # Tensor.addbmm_
    def addbmm_(self, batch1, batch2, *, beta=1, alpha=1):
        return self.copy_(ops.addbmm(self, batch1, batch2, beta=beta, alpha=alpha))

    # Tensor.addcdiv
    def addcdiv(self, tensor1, tensor2, *, value=1):
        return ops.addcdiv(self, tensor1, tensor2, value=value)

    # Tensor.addcdiv_
    def addcdiv_(self, tensor1, tensor2, *, value=1):
        return self.copy_(ops.addcdiv(self, tensor1, tensor2, value=value))

    # Tensor.addcmul
    def addcmul(self, tensor1, tensor2, *, value=1):
        return ops.addcmul(self, tensor1, tensor2, value=value)

    # Tensor.addcmul_
    def addcmul_(self, tensor1, tensor2, *, value=1):
        return self.copy_(ops.addcmul(self, tensor1, tensor2, value=value))

    # Tensor.addmm
    def addmm(self, mat1, mat2, *, beta=1, alpha=1):
        return ops.addmm(self, mat1, mat2, beta=beta, alpha=alpha)

    # Tensor.addmm_
    def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
        return self.copy_(ops.addmm(self, mat1, mat2, beta=beta, alpha=alpha))

    # Tensor.sspaddmm


    # Tensor.addmv
    def addmv(self, mat, vec, *, beta=1, alpha=1):
        return ops.addmv(self, mat, vec, beta=beta, alpha=alpha)

    # Tensor.addmv_
    def addmv_(self, mat, vec, *, beta=1, alpha=1):
        return self.copy_(ops.addmv(self, mat, vec, beta=beta, alpha=alpha))

    # Tensor.addr

    # Tensor.addr_


    # Tensor.adjoint

    # Tensor.allclose
    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return ops.allclose(self, other, rtol, atol, equal_nan)

    # Tensor.amax
    def amax(self, dim=None, keepdim=False):
        return ops.amax(self, dim, keepdim)

    # Tensor.amin
    def amin(self, dim=None, keepdim=False):
        return ops.amin(self, dim, keepdim)

    # Tensor.aminmax
    def aminmax(self, dim=None, keepdim=False):
        return ops.aminmax(self, dim=dim, keepdim=keepdim)

    # Tensor.angle


    # Tensor.apply_
    def apply_(self, callable):
        return self.copy_(callable(self))

    # Tensor.argmax
    def argmax(self, dim=None, keepdim=False):
        out = ops.argmax(self, dim, keepdim)
        return out

    # Tensor.argmin
    def argmin(self, dim=None, keepdim=False):
        out = ops.argmin(self, dim, keepdim)
        return out

    # Tensor.argsort
    def argsort(self, dim=-1, descending=False):
        return ops.argsort(self, dim=-1, descending=False)

    # Tensor.argwhere
    def argwhere(self):
        return ops.argwhere(self)

    # Tensor.asin
    def asin(self):
        return ops.asin(self)

    # Tensor.asin_
    def asin_(self):
        return self.copy_(ops.asin(self))

    # Tensor.arcsin
    arcsin = asin

    # Tensor.arcsin_
    arcsin_ = asin_

    # Tensor.as_strided
    def as_strided(self, size, stride, storage_offset=None):
        return ops.as_strided(self, size, stride, storage_offset)

    # Tensor.atan
    def atan(self):
        return ops.atan(self)

    # Tensor.atan_
    def atan_(self):
        return self.copy_(ops.atan(self))

    # Tensor.arctan
    arctan = atan

    # Tensor.arctan_
    arctan_ = atan_

    # Tensor.atan2
    def atan2(self, other):
        return ops.atan2(self, other)

    # Tensor.atan2_
    def atan2_(self, other):
        return self.copy_(ops.atan2(self, other))

    # Tensor.arctan2
    arctan2 = atan2

    # Tensor.arctan2_
    arctan2_ = atan2_

    # Tensor.all
    def all(self, dim=None, keepdim=False):
        return ops.all(self, dim, keepdim)

    # Tensor.any
    def any(self, dim=None, keepdim=False):
        return ops.any(self, dim, keepdim)

    # Tensor.baddbmm
    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return ops.baddbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    # Tensor.baddbmm_
    def baddbmm_(self, batch1, batch2, *, beta=1, alpha=1):
        return self.copy_(ops.baddbmm(self, batch1, batch2, beta=beta, alpha=alpha))

    # Tensor.bernoulli
    def bernoulli(self, *, generator=None):
        return ops.bernoulli(self, generator=generator)

    # Tensor.bernoulli_
    def bernoulli_(self, *, generator=None):
        return self.copy_(ops.bernoulli(self, generator=generator))

    # Tensor.bfloat16
    def bfloat16(self):
        return self.to(ops.bfloat16)

    # Tensor.bincount
    def bincount(self, weight=None, minlength=0):
        return ops.bincount(self, weight, minlength)

    # Tensor.bitwise_not
    def bitwise_not(self):
        return ops.bitwise_not(self)

    # Tensor.bitwise_not_
    def bitwise_not_(self):
        return self.copy_(ops.bitwise_not(self))

    # Tensor.bitwise_and
    def bitwise_and(self, other):
        return ops.bitwise_and(self, other)

    # Tensor.bitwise_and_
    def bitwise_and_(self, other):
        return self.copy_(ops.bitwise_and(self, other))

    # Tensor.bitwise_or
    def bitwise_or(self, other):
        return ops.bitwise_or(self, other)

    # Tensor.bitwise_or_
    def bitwise_or_(self, other):
        return self.copy_(ops.bitwise_or(self, other))

    # Tensor.bitwise_xor
    def bitwise_xor(self, other):
        return ops.bitwise_xor(self, other)

    # Tensor.bitwise_xor_
    def bitwise_xor_(self, other):
        return self.copy_(ops.bitwise_xor(self, other))

    # Tensor.bitwise_left_shift


    # Tensor.bitwise_left_shift_


    # Tensor.bitwise_right_shift


    # Tensor.bitwise_right_shift_


    # Tensor.bmm
    def bmm(self, batch2):
        return ops.bmm(self, batch2)

    # Tensor.bool
    def bool(self):
        return self.to(mindspore.bool_)

    # Tensor.byte
    def byte(self):
        return self.to(mindspore.uint8)

    # Tensor.broadcast_to
    def broadcast_to(self, shape):
        return ops.broadcast_to(self, shape)

    # Tensor.cauchy_


    # Tensor.ceil
    def ceil(self):
        return ops.ceil(self)

    # Tensor.ceil_
    def ceil_(self):
        return self.copy_(ops.ceil(self))

    # Tensor.char
    def char(self):
        return self.to(mindspore.int8)

    # Tensor.cholesky


    # Tensor.cholesky_inverse


    # Tensor.cholesky_solve


    # Tensor.chunk
    def chunk(self, chunks, dim=0):
        return ops.chunk(self, chunks, dim)

    # Tensor.clamp
    def clamp(self, min=None, max=None):
        return ops.clamp(self, min, max)

    def clamp_min(self, min):
        return ops.clamp(self, min, None)

    def clamp_max(self, min):
        return ops.clamp(self, None, max)

    # Tensor.clamp_
    def clamp_(self, min=None, max=None):
        return self.copy_(ops.clamp(self, min, max))

    # Tensor.clip
    def clip(self, min=None, max=None):
        return ops.clip(self, min, max)

    # Tensor.clip_
    def clip_(self, min=None, max=None):
        return self.copy_(ops.clip(self, min, max))

    # Tensor.clone
    def clone(self, memory_format=None):
        return ops.clone(self)

    # Tensor.contiguous
    def contiguous(self):
        return ops.contiguous(self)

    # Tensor.copy_
    def copy_(self, value):
        if self.dtype != value.dtype:
            value = value.to(self.dtype)
        return ops.inplace_copy(self, value)

    # Tensor.conj
    def conj(self):
        return ops.conj(self)

    # Tensor.conj_physical


    # Tensor.conj_physical_


    # Tensor.resolve_conj


    # Tensor.resolve_neg


    # Tensor.copysign


    # Tensor.copysign_


    # Tensor.cos
    def cos(self):
        return ops.cos(self)

    # Tensor.cos_
    def cos_(self):
        return self.copy_(ops.cos(self))

    # Tensor.cosh
    def cosh(self):
        return ops.cosh(self)

    # Tensor.cosh_
    def cosh_(self):
        return self.copy_(ops.cosh(self))

    # Tensor.corrcoef


    # Tensor.count_nonzero
    def count_nonzero(self, dim=None):
        return ops.count_nonzero(self, dim)

    # Tensor.cov


    # Tensor.acosh
    def acosh(self):
        return ops.acosh(self)

    # Tensor.acosh_
    def acosh_(self):
        return self.copy_(ops.acosh(self))

    # Tensor.arccosh
    arccosh = acosh

    # Tensor.arccosh_
    arccosh_ = acosh_

    # Tensor.cross


    # Tensor.logcumsumexp


    # Tensor.cummax


    # Tensor.cummin


    # Tensor.cumprod


    # Tensor.cumprod_


    # Tensor.cumsum
    def cumsum(self, dim, dtype=None):
        return ops.cumsum(self, dim, dtype)

    # Tensor.cumsum_
    def cumsum_(self, dim, dtype=None):
        return self.copy_(ops.cumsum(self, dim, dtype))

    # Tensor.chalf


    # Tensor.cfloat


    # Tensor.cdouble


    @property
    def data(self):
        out = Tensor(self)
        out._device = self.device
        out._base = self
        return out

    @data.setter
    def data(self, new_value):
        if isinstance(self, StubTensor) and isinstance(new_value, StubTensor):
            self.stub = new_value.stub
        else:
            # if self.device.type == 'cpu' and new_value.device.type == 'cpu' \
            #     and self.shape == new_value.shape and self.dtype == new_value.dtype:
            #     src_ct = ctypes.c_void_p(new_value.data_ptr())
            #     dst_ct = ctypes.c_void_p(self.data_ptr())
            #     ctypes.memmove(dst_ct, src_ct, self.nbytes)
            # else:
            if getattr(self, '_base', None) is not None:
                self._base.assign_value(new_value)
            self.assign_value(new_value)
        self._device = new_value.device

    # Tensor.data_ptr
    def data_ptr(self):
        if self.device.type in ['cpu']:
            self.dyn_shape()
        # ptr = self._data_ptr()
        return self._data_ptr()

    def dyn_shape(self):
        return ops.dyn_shape(self)

    # Tensor.deg2rad
    def deg2rad(self):
        return ops.deg2rad(self)

    # Tensor.dequantize


    # Tensor.det


    # Tensor.dense_dim


    # Tensor.diag
    def diag(self, diagonal=0):
        return ops.diag(self, diagonal)

    # Tensor.diag_embed


    # Tensor.diagflat


    # Tensor.diagonal
    def diagnoal(self, offset=0, dim1=0, dim2=1):
        return ops.diagonal(self, offset, dim1, dim2)


    # Tensor.diagonal_scatter

    # Tensor.fill_diagonal_
    def fill_diagonal_(self, value, wrap=False):
        return ops.inplace_fill_diagonal(self, value, wrap)

    # Tensor.fmax


    # Tensor.fmin


    # Tensor.diff


    # Tensor.digamma


    # Tensor.digamma_


    # Tensor.dim_order


    # Tensor.dist


    # Tensor.div
    def div(self, other, rounding_mode=None):
        return ops.div(self, other, rounding_mode=rounding_mode)

    # Tensor.div_
    def div_(self, other):
        return self.copy_(ops.div(self, other))

    # Tensor.divide
    divide = div

    # Tensor.divide_
    divide_ = div_

    # Tensor.dot
    def dot(self, other):
        return ops.dot(self, other)

    # Tensor.double
    def double(self):
        return self.to(mindspore.float64)

    # Tensor.dsplit


    # Tensor.element_size
    def element_size(self,):
        return DTYPE_ELEMENT_SIZE_MAP[self.dtype]

    # Tensor.eq
    def eq(self, other):
        return ops.eq(self, other)

    # Tensor.eq_
    def eq_(self, other):
        return self.copy_(ops.eq(self, other))

    # Tensor.equal
    def equal(self, other):
        return ops.eq(self, other)

    # Tensor.erf
    def erf(self):
        return ops.erf(self)

    # Tensor.erf_
    def erf_(self):
        return self.copy_(ops.erf(self))

    # Tensor.erfc
    def erfc(self):
        return ops.erfc(self)

    # Tensor.erfc_
    def erfc_(self):
        return self.copy_(ops.erfc(self))

    # Tensor.erfinv
    def erfinv(self):
        return ops.erfinv(self)


    # Tensor.erfinv_
    def erfinv_(self):
        return self.copy_(ops.erfinv(self))

    # Tensor.exp
    def exp(self):
        return ops.exp(self)

    # Tensor.exp_
    def exp_(self):
        return self.copy_(ops.exp(self))


    # Tensor.expm1
    def expm1(self):
        return ops.expm1(self)


    # Tensor.expm1_
    def expm1_(self):
        return self.copy_(ops.expm1(self))


    # Tensor.expand
    def expand(self, *size, **kwargs):
        size = kwargs.pop('size', size)
        if len(size) == 1:
            size = size[0]
        if isinstance(size, int):
            size = (size,)
        return self.broadcast_to(size)

    # Tensor.expand_as
    def expand_as(self, other):
        return self.expand(other.size())

    # Tensor.exponential_
    def exponential_(self, lambd=1, *, generator=None):
        return ops.inplace_exponential(self, lambd, generator)

    # Tensor.fix


    # Tensor.fix_


    # Tensor.fill_
    def fill_(self, value):
        ops.inplace_fill(self, value)
        return self

    # Tensor.flatten
    def flatten(self, start_dim=0, end_dim=-1):
        return ops.flatten(self, start_dim, end_dim)

    # Tensor.flip
    def flip(self, dims):
        return ops.flip(self, dims)

    # Tensor.fliplr


    # Tensor.flipud


    # Tensor.float
    def float(self):
        return self.to(mindspore.float32)

    # Tensor.float_power
    def float_power(self, exponent):
        return ops.float_power(self, exponent)

    # Tensor.float_power_
    def float_power_(self, exponent):
        return self.copy_(ops.float_power(self, exponent))

    # Tensor.floor
    def floor(self):
        return ops.floor(self)

    # Tensor.floor_
    def floor_(self):
        return self.copy_(ops.floor(self))

    # Tensor.floor_divide
    def floor_divide(self, other):
        return ops.floor_divide(self, other)

    # Tensor.floor_divide_
    def floor_divide_(self, other):
        return self.copy_(ops.floor_divide(self, other))


    # Tensor.fmod
    def fmod(self, other):
        return ops.fmod(self, other)

    # Tensor.fmod_
    def fmod_(self, other):
        return self.copy_(ops.fmod(self, other))

    # Tensor.frac
    def frac(self):
        return ops.frac(self)

    # Tensor.frac_
    def frac_(self):
        return self.copy_(ops.frac(self))


    # Tensor.frexp


    # Tensor.gather
    def gather(self, dim, index):
        return ops.gather(self, dim, index)

    # Tensor.gcd


    # Tensor.gcd_


    # Tensor.ge
    def ge(self, other):
        return ops.ge(self, other)

    # Tensor.ge_
    def ge_(self, other):
        return self.copy_(ops.ge(self, other))

    # Tensor.greater_equal
    greater_equal = ge

    # Tensor.greater_equal_
    greater_equal_ = ge_


    # Tensor.geometric_


    # Tensor.geqrf


    # Tensor.ger


    # Tensor.get_device
    def get_device(self):
        return self.device.index

    # Tensor.gt
    def gt(self, other):
        return ops.gt(self, other)

    # Tensor.gt_
    def gt_(self, other):
        return self.copy_(ops.gt(self, other))

    # Tensor.greater
    greater = gt

    # Tensor.greater_
    greater_ = gt_


    # Tensor.half
    def half(self):
        return self.to(mindspore.float16)

    # Tensor.hardshrink
    def hardshrink(self, lambd=0.5):
        return ops.nn.functional.hardshrink(self, lambd)

    # Tensor.heaviside


    # Tensor.histc
    def histc(self, bins=100, min=0, max=0):
        return ops.histc(self, bins, min, max)

    # Tensor.histogram


    # Tensor.hsplit


    # Tensor.hypot


    # Tensor.hypot_


    # Tensor.i0


    # Tensor.i0_


    # Tensor.igamma


    # Tensor.igamma_


    # Tensor.igammac


    # Tensor.igammac_


    # Tensor.index_add_
    def index_add_(self, dim, index, source, *, alpha=1):
        return self.copy_(ops.index_add(self, dim, index, source, alpha=alpha))

    # Tensor.index_add
    def index_add(self, dim, index, source, *, alpha=1):
        return ops.index_add(self, dim, index, source, alpha=alpha)

    # Tensor.index_copy_
    def index_copy_(self, dim, index, tensor2):
        return self.copy_(self.index_copy(dim, index, tensor2))

    # Tensor.index_copy
    def index_copy(self, dim, index, tensor2):
        original_values_at_index = self.index_select(dim, index)
        result = self.index_add(dim, index, -original_values_at_index)
        result.index_add_(dim, index, tensor2)
        return result

    # Tensor.index_fill_


    # Tensor.index_fill


    # Tensor.index_put_


    # Tensor.index_put


    # Tensor.index_reduce_


    # Tensor.index_reduce

    # Tensor.index_select
    def index_select(self, dim, index):
        return ops.index_select(self, dim, index)

    # Tensor.indices


    # Tensor.inner


    # Tensor.int
    def int(self):
        return self.to(mindspore.int32)

    # Tensor.int_repr


    # Tensor.inverse


    # Tensor.isclose
    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return ops.isclose(self, other, rtol, atol, equal_nan)

    # Tensor.isfinite
    def isfinite(self):
        return ops.isfinite(self)

    # Tensor.isinf
    def isinf(self):
        return ops.isinf(self)

    # Tensor.isposinf


    # Tensor.isneginf


    # Tensor.isnan
    def isnan(self):
        return ops.isnan(self)

    # Tensor.is_contiguous
    def is_contiguous(self, memory_format=None):
        return origin_is_contiguous(self)

    # Tensor.is_complex
    def is_complex(self):
        return False

    # Tensor.is_conj


    # Tensor.is_floating_point
    def is_floating_point(self):
        return isinstance(self.dtype, typing.Float)

    # Tensor.is_inference


    # Tensor.is_leaf
    @property
    def is_leaf(self):
        if not self.requires_grad:
            return True
        if self.requires_grad and hasattr(self, 'param_info'):
            return True
        return False

    # Tensor.is_pinned
    def is_pinned(self):
        return False

    # Tensor.is_set_to


    # Tensor.is_shared


    # Tensor.is_signed


    # Tensor.is_sparse
    @property
    def is_sparse(self):
        return False

    # Tensor.istft


    # Tensor.isreal


    # Tensor.item
    def item(self):
        return self._item()

    # Tensor.kthvalue

    @property
    def layout(self):
        return None

    # Tensor.lcm


    # Tensor.lcm_


    # Tensor.ldexp


    # Tensor.ldexp_


    # Tensor.le
    def le(self, other):
        return ops.le(self, other)

    # Tensor.le_
    def le_(self, other):
        return self.copy_(ops.le(self, other))

    # Tensor.less_equal
    less_equal = le

    # Tensor.less_equal_
    less_equal_ = le_


    # Tensor.lerp
    def lerp(self, end, weight):
        return ops.lerp(self, end, weight)

    # Tensor.lerp_
    def lerp_(self, end, weight):
        return self.copy_(ops.lerp(self, end, weight))


    # Tensor.lgamma


    # Tensor.lgamma_


    # Tensor.log
    def log(self):
        return ops.log(self)

    # Tensor.log_
    def log_(self):
        return self.copy_(ops.log(self))

    # Tensor.logdet


    # Tensor.log10
    def log10(self):
        return ops.log10(self)


    # Tensor.log10_
    def log10_(self):
        return self.copy_(ops.log10(self))

    # Tensor.log1p
    def log1p(self):
        return ops.log1p(self)


    # Tensor.log1p_
    def log1p_(self):
        return self.copy_(ops.log1p(self))


    # Tensor.log2
    def log2(self):
        return ops.log2(self)


    # Tensor.log2_
    def log2_(self):
        return self.copy_(ops.log2(self))


    # Tensor.log_normal_


    # Tensor.logaddexp


    # Tensor.logaddexp2


    # Tensor.logsumexp
    def logsumexp(self, dim, keepdim=False):
        return ops.logsumexp(self, dim, keepdim)

    # Tensor.logical_and
    def logical_and(self, other):
        return ops.logical_and(self, other)

    # Tensor.logical_and_
    def logical_and_(self, other):
        return self.copy_(ops.logical_and(self, other))


    # Tensor.logical_not
    def logical_not(self):
        return ops.logical_not(self)


    # Tensor.logical_not_
    def logical_not_(self):
        return self.copy_(ops.logical_not(self))


    # Tensor.logical_or
    def logical_or(self, other):
        return ops.logical_or(self, other)


    # Tensor.logical_or_
    def logical_or_(self, other):
        return self.copy_(ops.logical_or(self, other))


    # Tensor.logical_xor
    def logical_xor(self, other):
        return ops.logical_xor(self, other)

    # Tensor.logical_xor_
    def logical_xor_(self, other):
        return self.copy_(ops.logical_xor(self, other))

    # Tensor.logit


    # Tensor.logit_


    # Tensor.long
    def long(self):
        return self.to(mindspore.int64)

    # Tensor.lt
    def lt(self, other):
        return ops.lt(self, other)

    # Tensor.lt_
    def lt_(self, other):
        return self.copy_(ops.lt(self, other))

    # Tensor.less
    less = lt

    # Tensor.less_
    less_ = lt_

    # Tensor.lu


    # Tensor.lu_solve


    # Tensor.as_subclass


    # Tensor.map_


    # Tensor.masked_scatter_
    def masked_scatter_(self, mask, tensor):
        return self.copy_(ops.masked_scatter(self, mask, tensor))

    # Tensor.masked_scatter
    def masked_scatter(self, mask, tensor):
        return ops.masked_scatter(self, mask, tensor)

    # Tensor.masked_fill_
    def masked_fill_(self, mask, value):
        return self.copy_(ops.masked_fill(self, mask, value))

    # Tensor.masked_fill
    def masked_fill(self, mask, value):
        return ops.masked_fill(self, mask, value)

    # Tensor.masked_select
    def masked_select(self, mask):
        return ops.masked_select(self, mask)

    # Tensor.matmul
    def matmul(self, other):
        return ops.matmul(self, other)

    # Tensor.matrix_power


    # Tensor.matrix_exp


    # Tensor.max
    def max(self, dim=None, keepdim=False, **kwargs):
        dim = kwargs.pop('axis', dim)
        return ops.max(self, dim, keepdim)

    # Tensor.maximum
    def maximum(self, other):
        return ops.maximum(self, other)

    # Tensor.mean
    def mean(self, dim=None, keepdim=False, *, dtype=None, **kwargs):
        dim = kwargs.pop('axis', dim)
        return ops.mean(self, dim, keepdim, dtype=dtype)

    # Tensor.module_load


    # Tensor.nanmean


    # Tensor.median
    def median(self, dim=-1, keepdim=False):
        return ops.median(self, dim, keepdim)

    # Tensor.nanmedian


    # Tensor.min
    def min(self, dim=None, keepdim=False):
        return ops.min(self, dim, keepdim)

    # Tensor.minimum
    def minimum(self, other):
        return ops.minimum(self, other)

    # Tensor.mm
    mm = matmul

    # Tensor.smm


    # Tensor.mode
    def mode(self, dim=None, keepdim=False):
        return ops.mode(self, dim, keepdim)

    # Tensor.movedim
    def movedim(self, source, destination):
        return ops.movedim(source, destination)

    # Tensor.moveaxis
    moveaxis = movedim

    # Tensor.msort
    def msort(self):
        return ops.msort(self)

    # Tensor.mul
    def mul(self, other):
        return ops.mul(self, other)

    # Tensor.mul_
    def mul_(self, other):
        return self.copy_(ops.mul(self, other))

    # Tensor.multiply
    multiply = mul

    # Tensor.multiply_
    multiply_ = mul_


    # Tensor.multinomial
    def multinomial(self, num_samples, replacement=False, *, generator=None):
        return ops.multinomial(self, num_samples, replacement, generator=generator)

    # Tensor.mv


    # Tensor.mvlgamma


    # Tensor.mvlgamma_


    # Tensor.nansum
    def nansum(self, dim=None, keepdim=False, *, dtype=None):
        return ops.nansum(self, dim, keepdim, dtype=dtype)

    # Tensor.narrow
    def narrow(self, dim, start, length):
        return ops.narrow(self, dim, start, length)

    # Tensor.narrow_copy
    def narrow_copy(self, dimension, start, length):
        return ops.narrow(self, dimension, start, length).clone()

    # Tensor.nan_to_num
    def nan_to_num(self, nan=0.0, posinf=None, neginf=None):
        return ops.nan_to_num(self, nan, posinf, neginf)

    # Tensor.nan_to_num_
    def nan_to_num_(self, nan=0.0, posinf=None, neginf=None):
        return self.copy_(ops.nan_to_num(self, nan, posinf, neginf))

    # Tensor.ne
    def ne(self, other):
        return ops.ne(self, other)

    # Tensor.ne_
    def ne_(self, other):
        return self.copy_(ops.ne(self, other))

    # Tensor.not_equal
    not_equal = ne

    # Tensor.not_equal_
    not_equal_ = ne_


    # Tensor.neg
    def neg(self):
        return ops.neg(self)

    # Tensor.neg_
    def neg_(self):
        return self.copy_(ops.neg(self))

    # Tensor.negative
    negative = neg

    # Tensor.negative_
    negative_ = neg_


    # Tensor.numel
    def numel(self):
        return math.prod(self.shape)

    # Tensor.nelement
    nelement = numel

    # Tensor.nextafter


    # Tensor.nextafter_


    # Tensor.nonzero
    def nonzero(self, as_tuple=False):
        return ops.nonzero(self, as_tuple=as_tuple)

    # Tensor.norm
    def norm(self, p='fro', dim=None, keepdim=False, dtype=None):
        return ops.norm(self, p, dim, keepdim, dtype)

    # Tensor.normal_
    def normal_(self, mean=0, std=1, *, generator=None):
        return ops.inplace_normal(self, mean, std, generator=generator)

    # Tensor.numpy
    def numpy(self):
        assert self.device.type == 'cpu'
        return self.asnumpy()

    def mindspore(self):
        return mindspore.Tensor(self._data)

    # Tensor.orgqr


    # Tensor.ormqr


    # Tensor.outer
    def outer(self, vec2):
        return ops.outer(self, vec2)

    # Tensor.permute
    def permute(self, *dims):
        if isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return ops.permute(self, dims)

    # Tensor.pin_memory


    # Tensor.pinverse


    # Tensor.polygamma


    # Tensor.polygamma_


    # Tensor.positive
    def positive(self):
        return self

    # Tensor.pow
    def pow(self, exponent):
        return ops.pow(self, exponent)

    # Tensor.pow_
    def pow_(self, exponent):
        return self.copy_(ops.pow(self, exponent))


    # Tensor.prod
    def prod(self, dim=None, keepdim=False, dtype=None):
        return ops.prod(self, dim, keepdim, dtype=dtype)

    # Tensor.put_


    # Tensor.qr


    # Tensor.qscheme


    # Tensor.quantile


    # Tensor.nanquantile


    # Tensor.q_scale


    # Tensor.q_zero_point


    # Tensor.q_per_channel_scales


    # Tensor.q_per_channel_zero_points


    # Tensor.q_per_channel_axis


    # Tensor.rad2deg


    # Tensor.ravel
    def ravel(self):
        return ops.ravel(self)

    # Tensor.reciprocal
    def reciprocal(self):
        return ops.reciprocal(self)

    # Tensor.reciprocal_
    def reciprocal_(self):
        return self.copy_(ops.reciprocal(self))


    # Tensor.record_stream
    def record_stream(self, stream):
        pass

    # Tensor.register_hook
    def register_hook(self, hook):
        return self._data.register_hook(hook)

    # Tensor.register_post_accumulate_grad_hook


    # Tensor.remainder
    def remainder(self, other):
        return ops.remainder(self, other)

    # Tensor.remainder_
    def remainder_(self, other):
        return self.copy_(ops.remainder(self, other))

    # Tensor.renorm


    # Tensor.renorm_


    # Tensor.repeat
    def repeat(self, *repeats):
        return ops.tile(self, repeats)

    # Tensor.repeat_interleave
    def repeat_interleave(self, repeats, dim=None, output_size=None):
        return ops.repeat_interleave(self, repeats, dim, output_size=output_size)

    # Tensor.reshape
    def reshape(self, *shape, **kwargs):
        shape = kwargs.pop('shape', shape)
        return ops.reshape(self, *shape)

    # Tensor.reshape_as
    def reshape_as(self, other):
        return self.reshape(*other.shape)

    # Tensor.resize_
    def resize_(self, *shape):
        self.data = ops.reshape(self, *shape)
        return self

    # Tensor.resize_as_
    def resize_as_(self, other):
        self.data = ops.reshape(self, *other.shape)
        return self

    # Tensor.retains_grad
    @property
    def retains_grad(self):
        return not self.is_leaf and self._retain_grad

    # Tensor.roll
    def roll(self, shifts, dims=None):
        return ops.roll(self, shifts, dims)

    # Tensor.rot90


    # Tensor.round
    def round(self):
        return ops.round(self)

    # Tensor.round_
    def round_(self):
        return self.copy_(ops.round(self))


    # Tensor.rsqrt
    def rsqrt(self):
        return ops.rsqrt(self)

    # Tensor.rsqrt_
    def rsqrt_(self):
        return self.copy_(ops.rsqrt(self))


    # Tensor.scatter
    def scatter(self, dim, index, src):
        return ops.scatter(self, dim, index, src)

    # Tensor.scatter_
    def scatter_(self, dim, index, src):
        return self.copy_(ops.scatter(self, dim, index, src))

    # Tensor.scatter_add_
    def scatter_add_(self, dim, index, src):
        return self.copy_(ops.scatter_add(self, dim, index, src))

    # Tensor.scatter_add
    def scatter_add(self, dim, index, src):
        return ops.scatter_add(self, dim, index, src)


    # Tensor.scatter_reduce_
    def scatter_reduce_(self, dim, index, src, reduce, *, include_self=True):
        return self.copy_(ops.scatter_reduce(self, dim, index, src))


    # Tensor.scatter_reduce
    def scatter_reduce(self, dim, index, src, reduce, *, include_self=True):
        return ops.scatter_reduce(self, dim, index, src, reduce)


    # Tensor.select
    def select(self, dim, index):
        return ops.select(self, dim, index)

    # Tensor.select_scatter


    # Tensor.set_


    # Tensor.share_memory_


    # Tensor.short
    def short(self):
        return self.to(mindspore.int16)

    # Tensor.sigmoid
    def sigmoid(self):
        return ops.sigmoid(self)

    # Tensor.sigmoid_
    def sigmoid_(self):
        return self.copy_(ops.sigmoid(self))

    # Tensor.sign
    def sign(self):
        return ops.sign(self)

    # Tensor.sign_
    def sign_(self):
        return self.copy_(ops.sign(self))


    # Tensor.signbit


    # Tensor.sgn


    # Tensor.sgn_


    # Tensor.sin
    def sin(self):
        return ops.sin(self)

    # Tensor.sin_
    def sin_(self):
        return self.copy_(ops.sin(self))


    # Tensor.sinc
    def sinc(self):
        return ops.sinc(self)


    # Tensor.sinc_
    def sinc_(self):
        return self.copy_(ops.sinc(self))

    # Tensor.sinh
    def sinh(self):
        return ops.sinh(self)


    # Tensor.sinh_
    def sinh_(self):
        return self.copy_(ops.sinh(self))


    # Tensor.asinh
    def asinh(self):
        return ops.asinh(self)


    # Tensor.asinh_
    def asinh_(self):
        return self.copy_(ops.asinh(self))


    # Tensor.arcsinh
    arcsinh_ = asinh

    # Tensor.arcsinh_
    arcsinh_ = asinh_


    # Tensor.size
    def size(self, dim=None):
        if dim is None:
            return self.shape
        assert isinstance(dim, int), f'`dim` must be int but got {type(dim)}'
        return self.shape[dim]

    # Tensor.slogdet


    # Tensor.slice_scatter


    # Tensor.softmax
    def softmax(self, dim, dtype=None):
        return ops.softmax(self, dim, dtype=dtype)

    # Tensor.sort
    def sort(self, dim=-1, descending=False):
        return ops.sort(self, dim=dim, descending=descending)

    # Tensor.split
    def split(self, split_size, dim=0):
        return ops.split(self, split_size, dim)

    # Tensor.sparse_mask


    # Tensor.sparse_dim


    # Tensor.sqrt
    def sqrt(self):
        return ops.sqrt(self)

    # Tensor.sqrt_
    def sqrt_(self):
        return self.copy_(ops.sqrt(self))


    # Tensor.square
    def square(self):
        return ops.square(self)


    # Tensor.square_
    def square_(self):
        return self.copy_(ops.square(self))

    # Tensor.squeeze
    def squeeze(self, *dim, **kwargs):
        dim = kwargs.pop('dim', dim)
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]
        return ops.squeeze(self, dim)

    # Tensor.squeeze_
    def squeeze_(self, dim=None):
        return self.copy_(ops.squeeze(self, dim))


    # Tensor.std
    def std(self, dim=None, *, correction=1, keepdim=False):
        return ops.std(self, dim, correction=correction, keepdim=keepdim)

    # Tensor.stft


    # Tensor.storage


    # Tensor.untyped_storage
    def untyped_storage(self):
        return mindtorch.UntypedStorage(self)

    # Tensor.storage_offset


    # Tensor.storage_type


    # Tensor.stride
    # def stride(self, dim=None):
    #     if dim is None:
    #         return self.stride()
    #     return self.stride()[dim]


    # Tensor.sub
    def sub(self, other, *, alpha=1):
        return ops.sub(self, other, alpha=alpha)

    # Tensor.sub_
    def sub_(self, other, *, alpha=1):
        return self.copy_(ops.sub(self, other, alpha=alpha))


    # Tensor.subtract
    subtract = sub

    # Tensor.subtract_
    subtract_ = sub_

    # Tensor.sum
    def sum(self, dim=None, keepdim=False, dtype=None, **kwargs):
        dim = kwargs.pop('axis', dim)
        keepdim = kwargs.pop('keepdims', keepdim)
        if isinstance(dim, list):
            dim = tuple(dim)
        return ops.sum(self, dim, keepdim, dtype=dtype)

    # Tensor.sum_to_size


    # Tensor.svd


    # Tensor.swapaxes
    def swapaxes(self, dim0, dim1):
        return ops.swapaxes(self, dim0, dim1)

    # Tensor.swapdims
    swapdims = swapaxes

    @property
    def T(self):
        return self.t()

    # Tensor.t
    def t(self):
        return ops.t(self)

    # Tensor.t_
    def t_(self):
        self.data = ops.t(self)
        return self

    # Tensor.tensor_split
    def tensor_split(self, indices_or_sections, dim=0):
        return ops.tensor_split(self, indices_or_sections, dim)

    # Tensor.tile
    def tile(self, *dims):
        return ops.tile(self, dims)

    # Tensor.to
    def _move_to(self, device, non_blocking=False):
        if device.type == 'meta':
            out = Tensor(Tensor_(shape=self.shape, dtype=self.dtype))
            out._device = device
            return out
        if self.device == device:
            return self
        else:
            if DEVICE_TARGET == 'Ascend' and device.type == 'cuda':
                device.type = 'npu'
            device_str = device_map[device.type]
            # if device_str == 'Ascend':
            #     out = ops.empty_like(self, device=device)
            #     ACL_MEMCPY_HOST_TO_DEVICE = 1
            #     ret = acl.rt.memcpy(out.data_ptr(), self.nbytes, self.data_ptr(), self.nbytes, ACL_MEMCPY_HOST_TO_DEVICE)
            # else:
            # self.data_sync(True)
            if self.device.type == 'cpu':
                self.data_ptr()
            data = self.move_to(device_str, blocking=not non_blocking)

            out = Tensor(data)
            out._device = device
            return out

    def to(self, *args, **kwargs):
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)
        out = self
        device = kwargs.pop('device', None)
        dtype = kwargs.pop('dtype', None)
        if device:
            args += (device,)
        if dtype:
            args += (dtype,)

        for arg in args:
            if isinstance(arg, device_):
                out = Tensor._move_to(out, arg, non_blocking)
            elif isinstance(arg, int):
                device = device_(arg)
                out = Tensor._move_to(out, device, non_blocking)
            elif isinstance(arg, str):
                device = device_(arg)
                out = Tensor._move_to(out, device, non_blocking)
            elif isinstance(arg, mindspore.common.dtype.Type):
                if out.dtype == arg:
                    return out
                else:
                    out = ops.cast(out, arg)
            elif isinstance(arg, Tensor):
                out = Tensor._move_to(out, arg.device, non_blocking)
                if out.dtype == arg:
                    return out
                else:
                    out = ops.cast(out, arg.dtype)
        return out

    # Tensor.take
    def take(self, index):
        return ops.take(self, index)

    # Tensor.take_along_dim


    # Tensor.tan
    def tan(self):
        return ops.tan(self)

    # Tensor.tan_
    def tan_(self):
        return self.copy_(ops.tan(self))


    # Tensor.tanh
    def tanh(self):
        return ops.tanh(self)


    # Tensor.tanh_
    def tanh_(self):
        return self.copy_(ops.tanh(self))


    # Tensor.atanh

    def atanh(self):
        return ops.atanh(self)


    # Tensor.atanh_
    def atanh_(self):
        return self.copy_(ops.atanh(self))


    # Tensor.arctanh
    arctanh = atanh

    # Tensor.arctanh_
    arctanh_ = atanh_

    # Tensor.tolist
    # def tolist(self):
    #     return self.numpy().tolist()

    # Tensor.topk
    def topk(self, k, dim=-1, largest=True, sorted=True):
        return ops.topk(self, k, dim, largest, sorted)

    # Tensor.to_dense


    # Tensor.to_sparse


    # Tensor.to_sparse_csr


    # Tensor.to_sparse_csc


    # Tensor.to_sparse_bsr


    # Tensor.to_sparse_bsc


    # Tensor.trace


    # Tensor.transpose
    def transpose(self, dim0, dim1):
        return ops.transpose(self, dim0, dim1)

    # Tensor.transpose_
    def transpose_(self, dim0, dim1):
        self.data = ops.transpose(self, dim0, dim1)
        return self

    # Tensor.triangular_solve


    # Tensor.tril
    def tril(self, diagonal=0):
        return ops.tril(self, diagonal)

    # Tensor.tril_
    def tril_(self, diagonal=0):
        return self.copy_(ops.tril(self, diagonal))


    # Tensor.triu
    def triu(self, diagonal=0):
        return ops.triu(self, diagonal)


    # Tensor.triu_
    def triu_(self, diagonal=0):
        return self.copy_(ops.triu(self, diagonal))


    # Tensor.true_divide
    def true_divide(self, other):
        return ops.true_divide(self, other)

    # Tensor.true_divide_
    def true_divide_(self, other):
        return self.copy_(ops.true_divide(self, other))


    # Tensor.trunc
    def trunc(self):
        return ops.trunc(self)

    # Tensor.trunc_
    def trunc_(self):
        return self.copy_(ops.trunc(self))


    # Tensor.type
    def type(self, dtype=None, non_blocking=False):
        if dtype is None:
            dtype_str = str(dtype_class_map[self.dtype])[8:-2]
            dtype_str = dtype_str.replace('_tensor', self.device.type) \
                if self.device.type != 'cpu' else dtype_str.replace('._tensor', '')
            return dtype_str
        return self.to(dtype, non_blocking=non_blocking)

    # Tensor.type_as
    def type_as(self, tensor):
        out = self.type(tensor.dtype)
        if self.device != tensor.device:
            out = out.to(tensor.device)
        return out

    # Tensor.unbind
    def unbind(self, dim=0):
        return ops.unbind(self, dim)

    # Tensor.unflatten
    def unflatten(self, dim, sizes):
        return ops.unflatten(self, dim, sizes)

    # Tensor.unfold
    def unfold(self, dimension, size, step):
        return ops.unfold(self, dimension, size, step)

    # Tensor.uniform_
    def uniform_(self, *args, **kwargs):
        return ops.inplace_uniform(self, *args, **kwargs)

    # Tensor.random_
    def random_(self, *args, **kwargs):
        return ops.inplace_random(self, *args, **kwargs)

    # Tensor.unique
    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        return ops.unique(self, sorted, return_inverse, return_counts, dim)

    # Tensor.unique_consecutive
    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        return ops.unique_consecutive(self, return_inverse, return_counts, dim)

    # Tensor.unsqueeze
    def unsqueeze(self, dim):
        return ops.unsqueeze(self, dim)

    # Tensor.unsqueeze_
    def unsqueeze_(self, dim):
        self.data = ops.unsqueeze(self, dim)
        return self


    # Tensor.values


    # Tensor.var
    def var(self, dim=None, *, correction=1, keepdim=False, **kwargs):
        correction = int(kwargs.pop('unbiased', correction))
        return ops.var(self, dim, correction=correction, keepdim=keepdim)

    # Tensor.vdot


    # Tensor.view
    def view(self, *shape):
        return self.reshape(*shape)

    # Tensor.view_as
    def view_as(self, other):
        return self.reshape(*other.shape)

    # Tensor.vsplit


    # Tensor.where
    def where(self, condition, y):
        return ops.where(condition, self, y)

    # Tensor.xlogy
    def xlogy(self, other):
        return ops.xlogy(self, other)

    # Tensor.xlogy_
    def xlogy_(self, other):
        return self.copy_(ops.xlogy(self, other))

    # Tensor.zero_
    def zero_(self):
        return ops.inplace_zero(self) 

    # Tensor.detach
    def detach(self):
        return ops.stop_gradient(self)

    # Tensor.detach_
    def detach_(self):
        self.requires_grad_(False)
        return self

    def stub_sync(self):
        if self.stub:
            self.tensor = self.stub.get_value()
            self.stub = None
        return self.tensor

    @property
    def is_cuda(self):
        device_type = 'cuda'
        if DEVICE_TARGET == 'Ascend':
            device_type = 'npu'
        return self.device.type == device_type

    def tobytes(self):
        return self.get_bytes()
    
    def __contains__(self, item):
        return ops.eq(self, item).any()

    def __float__(self):
        out = self.item()
        return round(float(out), 5)

    def pin_memory(self, *args, **kwargs):
        return self


    @property
    def shape(self):
        if isinstance(self, StubTensor):
            if self.stub is not None:
                stub_shape = self.stub.get_shape()
            else:
                stub_shape = self.tensor.shape
            return Size(stub_shape)
        return Size(self._shape)

    @property
    def is_meta(self):
        return False

    @property
    def device(self):
        if not hasattr(self, '_device'):
            raise ValueError('Tensor must have device')
        return self._device

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
            if isinstance(start, (np.integer, Tensor)):
                start = int(start)
            if isinstance(stop, (np.integer, Tensor)):
                stop = int(stop)
            if isinstance(step, (np.integer, Tensor)):
                step = int(step)
            
            return slice(start, stop, step)
        
        # 转换单个 NumPy 索引值
        elif isinstance(key, np.integer):
            return int(key)

        # 其他类型（如 int、None）直接返回
        else:
            return key

    def __deepcopy__(self, memodict):
        new_obj = Tensor(self)
        new_obj._device = self.device
        return new_obj

    def __matmul__(self, other):
        return ops.matmul(self, other)

    def __truediv__(self, other):
        return ops.true_divide(self, other)

    def __floordiv__(self, other):
        return ops.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return ops.floor_divide(other, self)


    def __ifloordiv__(self, other):
        return self.copy_(ops.floor_divide(self, other))

    def __mod__(self, other):
        return ops.fmod(self, other)

    def backward(self):
        return self

    def log_softmax(self, dim):
        return ops.log_softmax(self, dim)

    def char(self):
        return self.to(mindtorch.int8)

    def cross(self, other, dim=None):
        return ops.cross(self, other, dim)

    @property
    def is_nested(self):
        return False

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if not isinstance(value, bool):
            raise TypeError("The 'requires_grad' attribute of parameter must be set as bool.")
        self._requires_grad = value
        if self.param_info is not None:
            self.param_info.requires_grad = value
        else:
            self.param_info = ParamInfo()
            self.param_info.requires_grad = value

        if value:
            if not hasattr(self, 'handle'):
                self.retain_grad()
        else:
            if hasattr(self, 'handle'):
                self.handle.remove()
                delattr(self, 'handle')

    def retain_grad(self):
        pass

def enable_mindspore_patch():
    fn_keys = list(TensorPlaceHolder.__dict__)
    fn_keys.remove('__doc__')
    fn_keys.remove('__dict__')
    fn_keys.remove('__weakref__')
    fn_keys.remove('__module__')

    for fn in fn_keys:
        setattr(Tensor, fn, getattr(TensorPlaceHolder, fn))
        if StubTensor is not None:
            setattr(StubTensor, fn, getattr(TensorPlaceHolder, fn))


def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    return ret
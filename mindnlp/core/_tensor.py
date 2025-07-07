import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.tensor import _TensorMeta
try:
    from mindspore.common._stub_tensor import StubTensor
except:
    class StubTensor: pass

from ._dtype import dtype2np
from ._bind import get_default_device

from ._dtype import *


class TypedTensorMeta(_TensorMeta):
    def __isinstancecheck__(self, instance):
        if not isinstance(instance, Tensor):
            return False
        return instance.dtype == self.dtype

class LongTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = long
    def __init__(self, data, device=None):
        super().__init__(data, dtype=long)

class FloatTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = float32
    def __init__(self, data, device=None):
        super().__init__(data, dtype=float32)


class HalfTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = float16
    def __init__(self, data, device=None):
        super().__init__(data, dtype=float16)

class BFloat16Tensor(Tensor, metaclass=TypedTensorMeta):
    dtype = float16
    def __init__(self, data, device=None):
        super().__init__(data, dtype=bfloat16)


class BoolTensor(Tensor, metaclass=TypedTensorMeta):
    dtype = bool
    def __init__(self, data, device=None):
        super().__init__(data, dtype=bool)

def tensor(data, *, dtype=None, device=None, requires_grad=False):
    if isinstance(data, Tensor):
        UserWarning("To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than core.tensor(sourceTensor).")
        return Tensor(data)

    if device is None:
        device = get_default_device()

    data_np = np.array(data, order='C') # must be C for mindspore Tensor
    if dtype is not None:
        data_np = data_np.astype(dtype2np[dtype])

    tensor = Tensor(data_np).to(device)
    return tensor

def is_tensor(x):
    return isinstance(x, Tensor)

def enable_mindspore_patch():
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

    Tensor.device = None
    StubTensor.device = None

    def _expand(self, *size):
        if len(size) == 1:
            size = size[0]
        return self.broadcast_to(size)

    Tensor.expand = _expand
    StubTensor.expand = _expand

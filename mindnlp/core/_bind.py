import ctypes
from typing import Any
from ._dtype import *
from .types import device as device_
from .configs import ON_A1

DEFAULT_DTYPE, DEFAULT_DEVICE = float32, device_('cpu')

AUTO_CAST_DTYE = {
    'cuda': float16,
    'cpu': bfloat16,
    'npu': float16
}

def set_autocast_dtype(device_type, dtype):
    assert device_type in AUTO_CAST_DTYE.keys(), f'{device_type} is not in {AUTO_CAST_DTYE.keys()}'
    AUTO_CAST_DTYE[device_type] = dtype

def get_autocast_dtype(device_type):
    return AUTO_CAST_DTYE[device_type]

def set_default_dtype(dtype):
    """set default dtype"""
    global DEFAULT_DTYPE
    if ON_A1 and dtype == bfloat16:
        dtype = float16
    DEFAULT_DTYPE = dtype

def get_default_dtype():
    """get default dtype"""
    return DEFAULT_DTYPE

def set_default_device(device):
    """set default dtype"""
    global DEFAULT_DEVICE
    if isinstance(device, str):
        device = device_(device)
    DEFAULT_DEVICE = device

def get_default_device():
    """get default dtype"""
    return DEFAULT_DEVICE

bits_map = {

}

min_map = {
    float32: -3.40282e+38,
    float16: -65504,
    bfloat16: -3.38953e+38
}

max_map = {
    float32: 3.40282e+38,
    float16: 65504,
    bfloat16: 3.38953e+38
}

eps_map = {
    float32: 1.19209e-07,
    float16: 0.000976562,
    bfloat16: 0.0078125
}

tiny_map = {
    float32: 1.17549e-38,
    float16: 6.10352e-05,
    bfloat16: 1.17549e-38
}

smallest_normal_map = {
    float32: 1.17549e-38,
    float16: 6.10352e-05,
    bfloat16: 1.17549e-38
}

resolution_map = {
    float32: 1e-06,
    float16: 0.001,
    bfloat16: 0.01
}
class iinfo:
    def __init__(self, dtype):
        self._dtype = dtype

    @property
    def bits(self):
        return bits_map[self._dtype]

    @property
    def min(self):
        return min_map[self._dtype]

    @property
    def max(self):
        return max_map[self._dtype]

    @property
    def dtype(self):
        return str(self._dtype)


class finfo:
    def __init__(self, dtype):
        self._dtype = dtype

    @property
    def bits(self):
        return bits_map[self._dtype]

    @property
    def min(self):
        return min_map[self._dtype]

    @property
    def max(self):
        return max_map[self._dtype]

    @property
    def eps(self):
        return eps_map[self._dtype]

    @property
    def tiny(self):
        return tiny_map[self._dtype]

    @property
    def smallest_normal(self):
        return smallest_normal_map[self._dtype]

    @property
    def resolution(self):
        return resolution_map[self._dtype]

    @property
    def dtype(self):
        return str(self._dtype)

def asarray(obj: Any, *, dtype, device=None, copy = None, requires_grad = False):
    data = obj.data.view(core.dtype2np[dtype])
    out = core.Tensor(data)
    core._utils.set_device_address(out)
    return out

def view(self, dtype):
    data_ptr = self.data_ptr()
    nbytes = self.nbytes
    data = np.ctypeslib.as_array((ctypes.c_byte * nbytes).from_address(data_ptr), shape=(nbytes,))
    data = data.view(core.dtype2np[dtype])
    assert data_ptr == data.ctypes.data
    out = core.Tensor(data)
    core._utils.set_device_address(out)
    return out
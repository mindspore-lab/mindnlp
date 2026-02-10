__version__ = "0.1.0"

from ._dtype import DType, float32, float16, int64
from ._device import device as Device, _default_device


class _StubTensor:
    def __init__(self, data, dtype=float32, device=None):
        self.data = data
        self.dtype = dtype
        self.device = device or _default_device


def tensor(data, dtype=float32, device=None):
    if device is None:
        dev = _default_device
    elif isinstance(device, str):
        dev = Device(device)
    else:
        dev = device
    return _StubTensor(data, dtype=dtype, device=dev)

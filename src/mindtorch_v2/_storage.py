import numpy as np

from ._device import _default_device, device as Device
from ._dtype import float32, to_numpy_dtype


class Storage:
    def __init__(self, data, device=None, dtype=None):
        self.device = device or _default_device
        self.dtype = dtype or float32
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data, dtype=to_numpy_dtype(self.dtype))

    def to(self, device):
        if isinstance(device, str):
            device = Device(device)
        if device.type == self.device.type:
            return self
        if device.type == "cpu":
            raise NotImplementedError("NPU->CPU copy not implemented yet")
        raise NotImplementedError("CPU->NPU copy not implemented yet")

import numpy as np

from ._device import _default_device
from ._dtype import float32, to_numpy_dtype


class Storage:
    def __init__(self, data, device=None, dtype=None):
        self.device = device or _default_device
        self.dtype = dtype or float32
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data, dtype=to_numpy_dtype(self.dtype))

__version__ = "0.1.0"

from ._dtype import DType, float32, float16, int64
from ._device import device as Device, _default_device
from ._tensor import Tensor, tensor

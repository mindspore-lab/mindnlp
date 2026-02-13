__version__ = "0.1.0"

from ._dtype import DType, float32, float16, int64
from ._device import device as Device, _default_device
from ._tensor import Tensor
from ._creation import tensor, zeros, ones
from ._storage import UntypedStorage, TypedStorage
from ._functional import add, mul, matmul, relu, sum
from ._printing import set_printoptions, get_printoptions
from ._dispatch import pipeline_context
from ._backends import cpu
from ._autograd.grad_mode import is_grad_enabled, set_grad_enabled, no_grad, enable_grad
from . import npu
from . import _C


def pipeline():
    return pipeline_context()

from typing import Any

import mindspore
from mindspore import get_rng_state, set_rng_state, manual_seed
from mindspore.hal import *

from mindnlp import core

FloatTensor = core.FloatTensor
HalfTensor = core.FloatTensor
BFloat16Tensor = core.BFloat16Tensor

def manual_seed_all(seed: int):
    manual_seed(seed)

def current_device():
    return core.device('cuda', 0)

def is_available():
    return mindspore.get_context('device_target') == 'GPU'

def set_device(device):
    pass

def _lazy_call(callable, **kwargs):
    callable()

class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (core.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = -1

    def __exit__(self, type: Any, value: Any, traceback: Any):
        return False

OutOfMemoryError = RuntimeError
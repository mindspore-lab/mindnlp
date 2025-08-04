from typing import Any

import mindspore
from mindspore import get_rng_state, set_rng_state, manual_seed
from mindspore.runtime import memory_reserved as ms_memory_reserved, \
    memory_allocated as ms_memory_allocated, StreamCtx as StreamContext, Stream, empty_cache, \
    reset_peak_memory_stats, reset_max_memory_allocated, max_memory_allocated, synchronize, \
    current_stream
from mindspore.device_context.ascend import device_count 

from mindnlp import core
from ..configs import SUPPORT_BF16

FloatTensor = core.FloatTensor
HalfTensor = core.FloatTensor
BFloat16Tensor = core.BFloat16Tensor

def set_compile_mode(*args, **kwargs):
    pass

def manual_seed_all(seed: int):
    manual_seed(seed)

def current_device():
    return core.device('npu', 0)

def is_available():
    return mindspore.get_context('device_target') == 'Ascend'

def set_device(device):
    pass

def _lazy_call(callable, **kwargs):
    callable()

def is_bf16_supported():
    return SUPPORT_BF16

def memory_allocated(device=None):
    return ms_memory_allocated()

def memory_reserved(device=None):
    return ms_memory_reserved()

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

def mem_get_info(index):
    return (1024, 1024)

def current_device():
    return core.device('npu', 0)

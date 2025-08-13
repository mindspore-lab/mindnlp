import os
from typing import Any

import mindspore
from mindspore._c_expression import _ms_memory_recycle
from mindspore import get_rng_state, set_rng_state, manual_seed
from mindspore.runtime import memory_reserved as ms_memory_reserved, \
    memory_allocated as ms_memory_allocated, StreamCtx as StreamContext, Stream, empty_cache as ms_empty_cache, \
    reset_peak_memory_stats, reset_max_memory_allocated, max_memory_allocated, synchronize, \
    current_stream

from mindspore.device_context.ascend import device_count as ms_device_count
from mindspore.communication import GlobalComm, get_group_size

from mindnlp import core
from mindnlp.core.executor import execute
from ..configs import SUPPORT_BF16

FloatTensor = core.FloatTensor
HalfTensor = core.FloatTensor
BFloat16Tensor = core.BFloat16Tensor

def set_compile_mode(*args, **kwargs):
    pass

def manual_seed_all(seed: int):
    manual_seed(seed)

def device_count():
    if GlobalComm.INITED:
        return get_group_size()
    return ms_device_count()

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
    if os.environ.get("MS_ALLOC_CONF", None) is not None:
        # increase_size = 2GB
        out = ((ms_memory_allocated() // (1024 * 1024 * 2048)) + 1) * (1024 * 1024 * 2048)
        return out
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

def _try_initial_ascend():
    x = core.tensor(1, device='npu')
    _ = x + 0

def mem_get_info(device=None):
    if not isinstance(device, int):
        device = mindspore.context.get_context("device_id")

    res = mindspore.hal.get_device_properties(device)
    if res.total_memory == 0:
        _try_initial_ascend()
        res = mindspore.hal.get_device_properties(device)

    return (res.free_memory, res.total_memory)

def current_device():
    return core.device('npu', 0)

def get_device_capability(device=None):
    return 10, 0


def npu_rotary_mul(x, cos, sin):
    return execute('rotary_position_embedding', x, cos, sin, 0)

def empty_cache():
    ms_empty_cache()
    _ms_memory_recycle()
from typing import Any, Optional

import mindspore
from mindspore import get_rng_state, set_rng_state, manual_seed
from mindspore.runtime import memory_reserved as ms_memory_reserved, \
    memory_allocated as ms_memory_allocated, StreamCtx as StreamContext, Stream, empty_cache, \
    reset_peak_memory_stats, reset_max_memory_allocated, max_memory_allocated, synchronize, \
    current_stream
from mindspore.device_context.gpu import device_count as ms_device_count
from mindspore.hal import get_device_properties
from mindspore.communication import GlobalComm, get_group_size

import mindtorch

FloatTensor = mindtorch.FloatTensor
HalfTensor = mindtorch.FloatTensor
BFloat16Tensor = mindtorch.BFloat16Tensor

def device_count():
    if not is_available():
        return 0
    if GlobalComm.INITED:
        return get_group_size()
    return 1

def manual_seed_all(seed: int):
    manual_seed(seed)

def current_device():
    return mindtorch.device('cuda', 0)

def is_available():
    return mindspore.get_context('device_target') == 'GPU'

def set_device(device):
    pass

def _lazy_call(callable, **kwargs):
    callable()

class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (mindtorch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = -1

    def __exit__(self, type: Any, value: Any, traceback: Any):
        return False

OutOfMemoryError = RuntimeError

def is_bf16_supported():
    return False

def mem_get_info(device=None):
    if not isinstance(device, int):
        device = mindspore.context.get_context("device_id")

    res = mindspore.hal.get_device_properties(device)
    return (res.total_memory, res.total_memory)

def get_device_capability(device=None):
    if not isinstance(device, int):
        device = mindspore.context.get_context("device_id")

    res = mindspore.hal.get_device_properties(device)
    return (res.major, res.minor)

def memory_reserved(device=None):
    return ms_memory_reserved()

def memory_allocated(device=None):
    return ms_memory_allocated()

def stream(stream: Optional["torch.cuda.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note::
        In eager mode stream is of type Stream class while in JIT it is
        an object of the custom class ``torch.classes.cuda.Stream``.
    """
    return StreamContext(stream)

def is_initialized():
    return True
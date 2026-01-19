from typing import Any, Optional

import mindspore

import mindtorch

FloatTensor = mindtorch.FloatTensor
HalfTensor = mindtorch.FloatTensor
BFloat16Tensor = mindtorch.BFloat16Tensor


def device_count():
    return 1


def manual_seed_all(seed: int):
    mindspore.set_seed(seed)


def current_device():
    return mindtorch.device("cpu", 0)


def is_available():
    return True


def set_device(device):
    pass


def _lazy_call(callable, **kwargs):
    callable()


class device:
    def __init__(self, device: Any):
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = -1

    def __exit__(self, type: Any, value: Any, traceback: Any):
        return False


OutOfMemoryError = RuntimeError


def is_bf16_supported():
    return True


def get_device_properties(device=None):
    class CPUDeviceProperties:
        def __init__(self):
            self.total_memory = 1024 * 1024 * 1024 * 1024
            self.major = 0
            self.minor = 0
            self.name = "CPU"
    return CPUDeviceProperties()


def memory_reserved(device=None):
    return 0


def memory_allocated(device=None):
    return 0


def synchronize(device=None):
    pass


def is_initialized():
    return True

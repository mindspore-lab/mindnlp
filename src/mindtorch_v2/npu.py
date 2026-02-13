from ._backends.npu import is_available
from ._backends.npu.runtime import device_count
from ._device import device as Device

__all__ = ["is_available", "device_count", "synchronize"]


def synchronize(device=None):
    from ._backends.npu import runtime as npu_runtime
    if device is None:
        dev = Device("npu")
    elif isinstance(device, str):
        dev = Device(device)
    else:
        dev = device
    runtime = npu_runtime.get_runtime(dev.index or 0)
    runtime.synchronize()


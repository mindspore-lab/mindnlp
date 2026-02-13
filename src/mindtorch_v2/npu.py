from ._backends.npu import is_available
from ._backends.npu import state as npu_state
from ._backends.npu.runtime import device_count
from ._device import device as Device

__all__ = [
    "is_available",
    "device_count",
    "synchronize",
    "current_device",
    "set_device",
    "device",
]


def _normalize_npu_device(device):
    if device is None:
        return Device("npu", index=npu_state.current_device())
    if isinstance(device, Device):
        dev = device
    elif isinstance(device, int):
        dev = Device("npu", index=device)
    else:
        dev = Device(device)
    if dev.type != "npu":
        raise ValueError(f"Expected NPU device, got {dev}")
    if dev.index is None:
        return Device("npu", index=npu_state.current_device())
    return dev


def synchronize(device=None):
    from ._backends.npu import runtime as npu_runtime
    dev = _normalize_npu_device(device)
    runtime = npu_runtime.get_runtime(dev.index or 0)
    runtime.synchronize()


def current_device():
    return npu_state.current_device()


def set_device(device):
    dev = _normalize_npu_device(device)
    npu_state.set_device(dev.index or 0)


class device:
    def __init__(self, dev):
        self.dev = _normalize_npu_device(dev)
        self._ctx = None

    def __enter__(self):
        self._ctx = npu_state.device_guard(self.dev.index or 0)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)

from ._backends.npu import is_available as _backend_is_available
from ._backends.npu import state as npu_state
from ._backends.npu import allocator as npu_allocator
from ._backends.npu.runtime import device_count
from ._backends.npu.streams import Event, Stream
from ._device import device as Device



def is_available(verbose=False):
    return _backend_is_available(verbose=verbose)


__all__ = [
    "is_available",
    "device_count",
    "synchronize",
    "current_device",
    "set_device",
    "device",
    "Stream",
    "Event",
    "default_stream",
    "current_stream",
    "stream",
    "empty_cache",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "memory_stats",
    "max_memory_reserved",
    "memory_reserved",
    "max_memory_allocated",
    "memory_allocated",
    "mem_get_info",
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




def _get_allocator(device=None):
    dev = _normalize_npu_device(device)
    return npu_allocator.get_allocator(dev.index or 0)


def synchronize(device=None):
    from ._backends.npu import runtime as npu_runtime

    dev = _normalize_npu_device(device)
    runtime = npu_runtime.get_runtime(dev.index or 0)
    if hasattr(runtime, "synchronize_device"):
        runtime.synchronize_device()
    else:
        runtime.synchronize()


def mem_get_info(device=None):
    dev = _normalize_npu_device(device)
    from ._backends.npu import runtime as npu_runtime

    return npu_runtime.mem_get_info(dev.index or 0)


def current_device():
    return npu_state.current_device()


def set_device(device):
    dev = _normalize_npu_device(device)
    npu_state.set_device(dev.index or 0)


def default_stream(device=None):
    dev = _normalize_npu_device(device)
    return npu_state.default_stream(dev.index or 0)


def current_stream(device=None):
    dev = _normalize_npu_device(device)
    return npu_state.current_stream(dev.index or 0)


class stream:
    def __init__(self, s):
        self.stream = s
        self._prev = None
        self._dev_ctx = None

    def __enter__(self):
        self._prev = npu_state.current_stream(self.stream.device.index or 0)
        self._dev_ctx = npu_state.device_guard(self.stream.device.index or 0)
        self._dev_ctx.__enter__()
        npu_state.set_current_stream(self.stream)
        return self.stream

    def __exit__(self, exc_type, exc, tb):
        npu_state.set_current_stream(self._prev)
        return self._dev_ctx.__exit__(exc_type, exc, tb)


class device:
    def __init__(self, dev):
        self.dev = _normalize_npu_device(dev)
        self._ctx = None

    def __enter__(self):
        self._ctx = npu_state.device_guard(self.dev.index or 0)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def memory_allocated(device=None):
    return _get_allocator(device).memory_stats().get("allocated_bytes.all.current", 0)


def max_memory_allocated(device=None):
    return _get_allocator(device).memory_stats().get("allocated_bytes.all.peak", 0)


def memory_reserved(device=None):
    return _get_allocator(device).memory_stats().get("reserved_bytes.all.current", 0)


def max_memory_reserved(device=None):
    return _get_allocator(device).memory_stats().get("reserved_bytes.all.peak", 0)


def memory_stats(device=None):
    return _get_allocator(device).memory_stats()


def reset_peak_memory_stats(device=None):
    return _get_allocator(device).reset_peak_memory_stats()


def reset_accumulated_memory_stats(device=None):
    return _get_allocator(device).reset_accumulated_memory_stats()


def empty_cache(device=None):
    return _get_allocator(device).empty_cache()

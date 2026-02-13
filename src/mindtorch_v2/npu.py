from ._backends.npu import is_available as _backend_is_available
from ._backends.npu import state as npu_state
from ._backends.npu import allocator as npu_allocator
from ._backends.npu.runtime import device_count
from ._backends.npu.streams import Event, Stream
from ._device import device as Device

_MEMORY_FRACTION = None



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
    "stream_priority_range",
    "empty_cache",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "memory_stats",
    "max_memory_reserved",
    "memory_reserved",
    "max_memory_allocated",
    "memory_allocated",
    "mem_get_info",
    "memory_summary",
    "memory_snapshot",
    "set_per_process_memory_fraction",
    "get_device_name",
    "get_device_capability",
    "can_device_access_peer",
    "enable_peer_access",
    "disable_peer_access",
    "pin_memory",
    "is_pinned",
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


def stream_priority_range():
    return (0, 0)


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


def _reset_memory_fraction_for_test():
    global _MEMORY_FRACTION
    _MEMORY_FRACTION = None


def _get_memory_stats(device=None):
    try:
        import mindspore

        return mindspore.hal.memory_stats()
    except Exception:
        return {}


def _enforce_memory_fraction(device=None):
    if _MEMORY_FRACTION is None:
        return
    stats = _get_memory_stats(device)
    total = stats.get("total_memory", 0)
    if total <= 0:
        return
    limit = total * _MEMORY_FRACTION
    used = stats.get("used_memory", 0)
    if used > limit:
        raise RuntimeError(
            f"NPU memory usage exceeded per-process limit: {used} > {limit}"
        )


def set_per_process_memory_fraction(fraction, device=None):
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1]")
    global _MEMORY_FRACTION
    _MEMORY_FRACTION = float(fraction)
    _enforce_memory_fraction(device)


def _get_device_name(device=None):
    return "Ascend"


def _get_device_capability(device=None):
    return (0, 0)


def get_device_name(device=None):
    dev = _normalize_npu_device(device)
    return _get_device_name(dev.index or 0)


def get_device_capability(device=None):
    dev = _normalize_npu_device(device)
    return _get_device_capability(dev.index or 0)


def can_device_access_peer(device, peer_device):
    return False


def enable_peer_access(peer_device, device=None):
    raise RuntimeError("Peer access is not supported on NPU")


def disable_peer_access(peer_device, device=None):
    raise RuntimeError("Peer access is not supported on NPU")


def pin_memory(tensor):
    setattr(tensor, "_pinned", True)
    return tensor


def is_pinned(tensor):
    return bool(getattr(tensor, "_pinned", False))


def memory_summary(device=None, abbreviated=False):
    stats = {}
    alloc = _get_allocator(device)
    if alloc is not None and hasattr(alloc, "memory_stats"):
        stats = alloc.memory_stats() or {}
    allocated = stats.get("allocated_bytes.all.current", 0)
    allocated_peak = stats.get("allocated_bytes.all.peak", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    reserved_peak = stats.get("reserved_bytes.all.peak", 0)
    active = stats.get("active_bytes.all.current", 0)
    active_peak = stats.get("active_bytes.all.peak", 0)
    inactive = stats.get("inactive_split_bytes.all.current", 0)
    inactive_peak = stats.get("inactive_split_bytes.all.peak", 0)
    lines = [
        "|===========================================================================|",
        "|                             NPU Memory Summary                            |",
        "|===========================================================================|",
        f"| Allocated memory      | current: {allocated:>10} | peak: {allocated_peak:>10} |",
        f"| Reserved memory       | current: {reserved:>10} | peak: {reserved_peak:>10} |",
        f"| Active memory         | current: {active:>10} | peak: {active_peak:>10} |",
        f"| Inactive split memory | current: {inactive:>10} | peak: {inactive_peak:>10} |",
        "|===========================================================================|",
    ]
    if abbreviated:
        return "\n".join(lines[:4])
    return "\n".join(lines)


def memory_snapshot():
    alloc = _get_allocator(None)
    if alloc is not None and hasattr(alloc, "snapshot"):
        return alloc.snapshot()
    return {
        "segments": [],
        "device": current_device() if _backend_is_available() else 0,
        "allocator": "npu",
    }

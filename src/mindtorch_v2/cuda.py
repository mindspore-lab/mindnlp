from ._backends.cuda import runtime as cuda_runtime
from ._device import device as Device


def _normalize_cuda_device(dev=None):
    if dev is None:
        if is_available():
            return Device("cuda", index=current_device())
        return Device("cuda", index=0)
    if isinstance(dev, Device):
        out = dev
    elif isinstance(dev, int):
        out = Device("cuda", index=dev)
    else:
        out = Device(dev)
    if out.type != "cuda":
        raise ValueError(f"Expected a cuda device, but got: {out}")
    if out.index is None:
        return Device("cuda", index=current_device() if is_available() else 0)
    return out


def is_available():
    return cuda_runtime.is_available()


def device_count():
    return cuda_runtime.device_count()


def current_device():
    return cuda_runtime.current_device()


def set_device(dev):
    cuda_dev = _normalize_cuda_device(dev)
    cuda_runtime.set_device(cuda_dev.index or 0)


def synchronize(device=None):
    if device is None:
        cuda_runtime.synchronize()
        return
    cuda_dev = _normalize_cuda_device(device)
    cuda_runtime.synchronize(cuda_dev.index or 0)


class Stream:
    def __init__(self, device=None, priority=0, stream=None):
        self.device = _normalize_cuda_device(device)
        self.priority = int(priority)
        self._owns_stream = stream is None
        self.stream = int(stream) if stream is not None else cuda_runtime.create_stream()

    def synchronize(self):
        cuda_runtime.synchronize_stream(self.stream)

    def record_event(self, event=None):
        if event is None:
            event = Event()
        event.record(self)
        return event

    def __del__(self):
        if getattr(self, "_owns_stream", False) and getattr(self, "stream", 0):
            try:
                cuda_runtime.destroy_stream(self.stream)
            except Exception:
                pass


class Event:
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        self.enable_timing = bool(enable_timing)
        self.blocking = bool(blocking)
        self.interprocess = bool(interprocess)
        self.event = cuda_runtime.create_event()

    def record(self, stream=None):
        stream_handle = None if stream is None else stream.stream
        cuda_runtime.record_event(self.event, stream_handle)
        return self

    def synchronize(self):
        cuda_runtime.synchronize_event(self.event)

    def __del__(self):
        if getattr(self, "event", 0):
            try:
                cuda_runtime.destroy_event(self.event)
            except Exception:
                pass


class device:
    def __init__(self, dev):
        self.dev = _normalize_cuda_device(dev)
        self._prev = None

    def __enter__(self):
        self._prev = current_device()
        set_device(self.dev)
        return self.dev

    def __exit__(self, exc_type, exc, tb):
        if self._prev is not None:
            set_device(self._prev)
        return False


__all__ = [
    "is_available",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "Stream",
    "Event",
    "device",
]

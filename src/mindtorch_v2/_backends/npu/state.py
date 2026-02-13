import threading
from contextlib import contextmanager

from .runtime import get_runtime

_tls = threading.local()


def _state():
    if not hasattr(_tls, "current_device"):
        _tls.current_device = 0
        _tls.current_streams = {}
        _tls.default_streams = {}
    return _tls


def current_device():
    return _state().current_device


def set_device(device_id):
    device_id = int(device_id)
    _state().current_device = device_id
    get_runtime(device_id)


@contextmanager
def device_guard(device_id):
    state = _state()
    prev = state.current_device
    set_device(device_id)
    try:
        yield
    finally:
        set_device(prev)


def default_stream(device_id=None):
    from .streams import Stream

    state = _state()
    dev = current_device() if device_id is None else int(device_id)
    stream = state.default_streams.get(dev)
    if stream is None:
        stream = Stream(device=f"npu:{dev}")
        state.default_streams[dev] = stream
    return stream


def current_stream(device_id=None):
    state = _state()
    dev = current_device() if device_id is None else int(device_id)
    stream = state.current_streams.get(dev)
    if stream is None:
        stream = default_stream(dev)
        state.current_streams[dev] = stream
    return stream


def set_current_stream(stream):
    state = _state()
    state.current_streams[stream.device.index or 0] = stream

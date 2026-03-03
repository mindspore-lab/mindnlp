import threading
from contextlib import contextmanager

from .runtime import get_runtime

_tls = threading.local()
_default_streams = {}
_default_stream_runtime_ids = {}
_default_streams_lock = threading.Lock()


def _state():
    if not hasattr(_tls, "current_device"):
        _tls.current_device = 0
        _tls.current_streams = {}
    if not hasattr(_tls, "default_streams"):
        _tls.default_streams = _default_streams
    return _tls


def _sync_default_streams_from_state():
    state = _state()
    streams = getattr(state, "default_streams", None)
    if streams is None or streams is _default_streams:
        return
    with _default_streams_lock:
        _default_streams.clear()
        _default_stream_runtime_ids.clear()
    state.default_streams = _default_streams


def _reset_state_for_test():
    state = _state()
    state.current_device = 0
    state.current_streams = {}
    with _default_streams_lock:
        _default_streams.clear()
        _default_stream_runtime_ids.clear()
    state.default_streams = _default_streams


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

    _sync_default_streams_from_state()
    dev = current_device() if device_id is None else int(device_id)
    runtime_id = id(get_runtime(dev))
    with _default_streams_lock:
        stream = _default_streams.get(dev)
        cached_runtime_id = _default_stream_runtime_ids.get(dev)
        if stream is None or cached_runtime_id != runtime_id:
            stream = Stream(device=f"npu:{dev}")
            _default_streams[dev] = stream
            _default_stream_runtime_ids[dev] = runtime_id
    return stream


def current_stream(device_id=None):
    _sync_default_streams_from_state()
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

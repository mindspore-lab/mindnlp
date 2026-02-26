from . import runtime as npu_runtime
from ..._device import device as Device


class Stream:
    def __init__(self, device=None, priority=0):
        dev = Device(device or "npu")
        self.device = dev
        self.priority = int(priority)
        runtime = npu_runtime.get_runtime(self.device.index or 0)
        self._stream = runtime.create_stream(self.priority)

    @property
    def stream(self):
        return self._stream

    def synchronize(self):
        runtime = npu_runtime.get_runtime(self.device.index or 0)
        runtime.synchronize_stream(self._stream)

    def wait_event(self, event):
        runtime = npu_runtime.get_runtime(self.device.index or 0)
        runtime.stream_wait_event(self._stream, event.event)

    def wait_stream(self, stream):
        event = Event()
        event.record(stream)
        self.wait_event(event)
        self._event = event

    def record_event(self, event=None):
        if event is None:
            event = Event()
        event.record(self)
        return event


class Event:
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        self.enable_timing = bool(enable_timing)
        self.blocking = bool(blocking)
        self.interprocess = bool(interprocess)
        from . import state as npu_state

        dev = Device("npu", index=npu_state.current_device())
        runtime = npu_runtime.get_runtime(dev.index or 0)
        self.device = dev
        self._event = runtime.create_event(
            self.enable_timing,
            self.blocking,
            self.interprocess,
        )

    @property
    def event(self):
        return self._event

    def record(self, stream=None):
        if stream is None:
            from . import state as npu_state

            stream = npu_state.current_stream()
        runtime = npu_runtime.get_runtime(stream.device.index or 0)
        runtime.record_event(self._event, stream.stream)

    def wait(self, stream=None):
        if stream is None:
            from . import state as npu_state

            stream = npu_state.current_stream()
        stream.wait_event(self)

    def synchronize(self):
        runtime = npu_runtime.get_runtime(self.device.index or 0)
        runtime.synchronize_event(self._event)

    def query(self):
        runtime = npu_runtime.get_runtime(self.device.index or 0)
        return runtime.query_event(self._event)

    def elapsed_time(self, other):
        runtime = npu_runtime.get_runtime(self.device.index or 0)
        return runtime.event_elapsed_time(self._event, other.event)

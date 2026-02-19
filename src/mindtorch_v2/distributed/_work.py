class Work:
    def __init__(self, stream=None, device_id=None, source_rank=-1):
        self._completed = False
        self._stream = stream
        self._device_id = device_id
        self._exception = None
        self._source_rank = source_rank

    def wait(self, timeout=None):
        if not self._completed and self._stream is not None:
            try:
                from .._backends.npu import runtime as npu_runtime
                dev = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev).synchronize_stream(self._stream)
            except Exception as e:
                self._exception = e
                raise
        self._completed = True
        return True

    def is_completed(self):
        return self._completed

    def is_success(self):
        return self._completed and self._exception is None

    def exception(self):
        return self._exception

    def source_rank(self):
        return self._source_rank

    def result(self):
        return []

    def synchronize(self):
        self.wait()

    def get_future(self):
        raise NotImplementedError("get_future is not supported")

import contextlib
import threading


_TLS = threading.local()


class Pipeline:
    def __init__(self):
        self.queue = []
        self.outputs = {}

    def record(self, entry, *, pending=None):
        self.queue.append(entry)
        if pending is not None:
            if isinstance(pending, (tuple, list)):
                for item in pending:
                    self.outputs[id(item)] = item
            else:
                self.outputs[id(pending)] = pending
            entry.out = pending

    def flush(self):
        pending = list(self.queue)
        self.queue.clear()
        for entry in pending:
            entry.execute()
        self.outputs.clear()


def _get_current():
    return getattr(_TLS, "current", None)


def _set_current(pipe):
    _TLS.current = pipe


@contextlib.contextmanager
def pipeline_context():
    prev = _get_current()
    pipe = Pipeline()
    _set_current(pipe)
    try:
        yield pipe
    finally:
        pipe.flush()
        _set_current(prev)


def current_pipeline():
    return _get_current()

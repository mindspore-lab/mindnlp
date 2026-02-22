import contextlib


_CURRENT = None


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


@contextlib.contextmanager
def pipeline_context():
    global _CURRENT
    prev = _CURRENT
    _CURRENT = Pipeline()
    try:
        yield _CURRENT
    finally:
        _CURRENT.flush()
        _CURRENT = prev


def current_pipeline():
    return _CURRENT

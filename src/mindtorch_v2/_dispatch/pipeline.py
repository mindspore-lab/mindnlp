import contextlib


_CURRENT = None


class Pipeline:
    def __init__(self):
        self.queue = []

    def record(self, entry):
        self.queue.append(entry)

    def flush(self):
        for entry in self.queue:
            entry.execute()
        self.queue.clear()


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

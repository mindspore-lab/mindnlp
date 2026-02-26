import threading


_GRAD_MODE_STATE = threading.local()


def _get_enabled():
    return getattr(_GRAD_MODE_STATE, "enabled", True)


def _set_enabled(mode):
    _GRAD_MODE_STATE.enabled = bool(mode)


class GradMode:
    @property
    def enabled(self):
        return _get_enabled()

    @enabled.setter
    def enabled(self, mode):
        _set_enabled(mode)


GradMode = GradMode()


def is_grad_enabled():
    return _get_enabled()


class set_grad_enabled:
    def __init__(self, mode):
        self.mode = bool(mode)
        self._prev = _get_enabled()
        _set_enabled(self.mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)


class no_grad:
    def __enter__(self):
        self._prev = _get_enabled()
        _set_enabled(False)

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)


class enable_grad:
    def __enter__(self):
        self._prev = _get_enabled()
        _set_enabled(True)

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)

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


def _decorate_with_grad_mode(fn, enabled):
    def wrapped(*args, **kwargs):
        with set_grad_enabled(enabled):
            return fn(*args, **kwargs)
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped.__module__ = getattr(fn, "__module__", None)
    return wrapped


class _NoGradContext:
    def __enter__(self):
        self._prev = _get_enabled()
        _set_enabled(False)
        return self

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)

    def __call__(self, fn):
        return _decorate_with_grad_mode(fn, False)


class _EnableGradContext:
    def __enter__(self):
        self._prev = _get_enabled()
        _set_enabled(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)

    def __call__(self, fn):
        return _decorate_with_grad_mode(fn, True)


def no_grad(func=None):
    ctx = _NoGradContext()
    if func is None:
        return ctx
    return ctx(func)


def enable_grad(func=None):
    ctx = _EnableGradContext()
    if func is None:
        return ctx
    return ctx(func)


def inference_mode(mode=True):
    return no_grad() if mode else enable_grad()

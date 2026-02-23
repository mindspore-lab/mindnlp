class GradMode:
    enabled = True


def is_grad_enabled():
    return GradMode.enabled


class set_grad_enabled:
    def __init__(self, mode):
        self.mode = bool(mode)
        self._prev = GradMode.enabled
        GradMode.enabled = self.mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        GradMode.enabled = self._prev


class no_grad:
    def __enter__(self):
        self._prev = GradMode.enabled
        GradMode.enabled = False

    def __exit__(self, exc_type, exc, tb):
        GradMode.enabled = self._prev


class enable_grad:
    def __enter__(self):
        self._prev = GradMode.enabled
        GradMode.enabled = True

    def __exit__(self, exc_type, exc, tb):
        GradMode.enabled = self._prev

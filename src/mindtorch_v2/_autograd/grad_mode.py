class GradMode:
    enabled = True


def is_grad_enabled():
    return GradMode.enabled


def set_grad_enabled(mode):
    GradMode.enabled = bool(mode)


class no_grad:
    def __enter__(self):
        GradMode.enabled = False

    def __exit__(self, exc_type, exc, tb):
        GradMode.enabled = True


class enable_grad:
    def __enter__(self):
        GradMode.enabled = True

    def __exit__(self, exc_type, exc, tb):
        pass

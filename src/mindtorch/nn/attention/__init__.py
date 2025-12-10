import contextlib

class SDPBackend:
    MATH = 0

@contextlib.contextmanager
def sdpa_kernel(*args, **kwargs):
    yield {}
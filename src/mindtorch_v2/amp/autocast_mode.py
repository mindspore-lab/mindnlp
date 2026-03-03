import functools


class autocast:
    """No-op autocast context manager/decorator for API compatibility.

    A proper implementation would require DispatchKey.Autocast kernels.
    For now, users can manually cast models to fp16/bf16.
    """

    def __init__(self, device_type=None, dtype=None, enabled=True, cache_enabled=None):
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled
        self.cache_enabled = cache_enabled

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

import functools

def disable(fn=None, recursive=True, *, reason=None):
    def wrap_func(func):
        @functools.wraps(func)
        def staging_specialize(*args, **kwargs):
            return func(*args, **kwargs)

        return staging_specialize

    if fn is not None:
        return wrap_func(fn)
    return wrap_func

def reset(): pass

def is_compiling():
    return False
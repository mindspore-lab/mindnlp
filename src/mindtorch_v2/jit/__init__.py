from ._trace import _script_if_tracing


def is_tracing():
    return False


def is_scripting():
    return False


def script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None):
    return obj


def ignore(drop=False, **kwargs):
    if callable(drop):
        return drop

    def decorator(fn):
        return fn

    return decorator


def unused(fn):
    return fn


def _overload_method(func):
    return func


def _script_if_tracing_wrapper(fn):
    return _script_if_tracing(fn)


def script_if_tracing(fn):
    return _script_if_tracing(fn)

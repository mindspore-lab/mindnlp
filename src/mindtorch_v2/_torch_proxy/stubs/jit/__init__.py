"""Stub for torch.jit - JIT compilation.

Tier 2 stub: called but returns functions unchanged (no-op).
"""

from functools import wraps


def script(obj=None, optimize=None, _frames_up=0, _rcb=None):
    """JIT script decorator/function - returns input unchanged."""
    if obj is None:
        # Used as decorator with arguments
        def decorator(fn):
            return fn
        return decorator
    return obj


def trace(func, example_inputs=None, optimize=None, check_trace=True,
          check_inputs=None, check_tolerance=1e-5, strict=True, _force_outplace=False,
          _module_class=None, _compilation_unit=None):
    """JIT trace function - returns input unchanged."""
    return func


def trace_module(mod, inputs=None, optimize=None, check_trace=True,
                 check_inputs=None, check_tolerance=1e-5, strict=True,
                 _force_outplace=False, _module_class=None, _compilation_unit=None):
    """JIT trace module - returns input unchanged."""
    return mod


def is_scripting():
    """Check if currently in script mode."""
    return False


def is_tracing():
    """Check if currently tracing."""
    return False


def fork(func, *args, **kwargs):
    """Fork execution - just calls function."""
    return func(*args, **kwargs)


def wait(future):
    """Wait for future - returns input."""
    return future


def save(m, f, _extra_files=None):
    """Save JIT module - no-op."""
    pass


def load(f, map_location=None, _extra_files=None):
    """Load JIT module - raises error."""
    raise NotImplementedError("JIT load not supported in mindtorch_v2")


def freeze(mod, preserved_attrs=None, optimize_numerics=True):
    """Freeze module - returns input unchanged."""
    return mod


def optimize_for_inference(mod, other_methods=None):
    """Optimize for inference - returns input unchanged."""
    return mod


class ScriptModule:
    """Stub for ScriptModule."""
    pass


class ScriptFunction:
    """Stub for ScriptFunction."""
    pass


class RecursiveScriptModule:
    """Stub for RecursiveScriptModule."""
    pass


def unused(fn):
    """Mark function as unused in script - returns unchanged."""
    return fn


def ignore(drop=False, **kwargs):
    """Ignore decorator - returns function unchanged."""
    def decorator(fn):
        return fn
    return decorator


def export(fn):
    """Export decorator - returns function unchanged."""
    return fn


def interface(obj):
    """Interface decorator - returns unchanged."""
    return obj


class Attribute:
    """JIT attribute descriptor."""

    def __init__(self, value, type=None):
        self.value = value
        self.type = type


class Final:
    """JIT Final type hint."""

    def __class_getitem__(cls, item):
        return item


def isinstance(obj, target_type):
    """JIT isinstance - uses builtin."""
    import builtins
    return builtins.isinstance(obj, target_type)


# Annotations
def annotate(the_type, the_value):
    """Type annotation helper."""
    return the_value


# Error classes
class Error(Exception):
    """JIT error."""
    pass


class FrontendError(Error):
    """JIT frontend error."""
    pass


# Additional functions that might be called
def get_trace_graph(f, args=(), kwargs=None, strict=True, _force_outplace=False,
                    return_inputs=False, _return_inputs_states=False):
    """Get trace graph - not implemented."""
    raise NotImplementedError("get_trace_graph not supported")


def _get_trace_graph(*args, **kwargs):
    """Internal get trace graph - not implemented."""
    raise NotImplementedError("_get_trace_graph not supported")


class _IgnoreContextManager:
    """Context manager for ignore regions."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _script_if_tracing(fn):
    """Decorator that scripts if tracing - returns unchanged."""
    return fn


def set_fusion_strategy(strategy):
    """Set fusion strategy - no-op."""
    pass


def strict_fusion(fn):
    """Strict fusion decorator - returns unchanged."""
    return fn


def _overload(fn):
    """Overload decorator - returns unchanged."""
    return fn


def _overload_method(fn):
    """Overload method decorator - returns unchanged."""
    return fn


class Future:
    """JIT Future stub."""

    def __init__(self, value=None):
        self._value = value

    def wait(self):
        return self._value

    def then(self, callback):
        return Future(callback(self._value))


# async helpers
def _awaitable(fn):
    """Awaitable decorator - returns unchanged."""
    return fn


def _awaitable_wait(fn):
    """Awaitable wait - returns unchanged."""
    return fn

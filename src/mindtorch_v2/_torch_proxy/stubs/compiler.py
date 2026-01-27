"""Stub for torch.compiler module."""


def disable(fn=None, recursive=True):
    """Disable compiler on a function - returns identity decorator."""
    def decorator(func):
        return func
    if fn is not None:
        return fn
    return decorator


def is_compiling():
    """Check if currently compiling."""
    return False


def is_dynamo_compiling():
    """Check if dynamo is compiling."""
    return False


def allow_in_graph(fn):
    """Allow function in graph - returns identity decorator."""
    return fn


def assume_constant_result(fn):
    """Assume constant result - returns identity decorator."""
    return fn


def cudagraph_mark_step_begin():
    """Mark CUDA graph step begin - no-op."""
    pass


def reset():
    """Reset compiler state - no-op."""
    pass


__all__ = [
    'disable',
    'is_compiling',
    'is_dynamo_compiling',
    'allow_in_graph',
    'assume_constant_result',
    'cudagraph_mark_step_begin',
    'reset',
]

"""Stub for torch.fx._compatibility module."""


def compatibility(is_backward_compatible=True):
    """Decorator that marks functions for compatibility tracking.

    This is a no-op stub that just returns the function unchanged.
    """
    def decorator(fn):
        return fn
    return decorator


__all__ = ['compatibility']

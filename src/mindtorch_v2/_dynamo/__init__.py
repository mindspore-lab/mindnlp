"""Stub _dynamo module for compatibility.

This provides dummy implementations for torch._dynamo APIs
that are checked by libraries like accelerate and transformers.
"""

from .eval_frame import OptimizedModule


def allow_in_graph(fn):
    """Decorator that allows a function to be traced through dynamo.

    This is a no-op stub since we don't have torch.compile.
    """
    return fn


def disable(fn=None, recursive=True):
    """Decorator that disables dynamo for a function.

    This is a no-op stub since we don't have torch.compile.
    """
    if fn is None:
        # Called as @disable() with parens
        def decorator(fn):
            return fn
        return decorator
    return fn


def forbid_in_graph(fn):
    """Decorator that forbids a function from being traced through dynamo.

    This is a no-op stub since we don't have torch.compile.
    """
    return fn


__all__ = ['OptimizedModule', 'eval_frame', 'allow_in_graph', 'disable', 'forbid_in_graph']

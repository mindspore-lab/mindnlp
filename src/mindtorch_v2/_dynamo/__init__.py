"""Stub _dynamo module for compatibility.

This provides dummy implementations for torch._dynamo APIs
that are checked by various PyTorch-based libraries.
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


def mark_static_address(tensor, guard=True):
    """Mark a tensor's memory address as static for compilation.

    This is a no-op stub since we don't have torch.compile.

    Args:
        tensor: The tensor to mark
        guard: Whether to guard on the tensor's address
    """
    pass


def reset():
    """Reset the dynamo state.

    This is a no-op stub since we don't have torch.compile.
    """
    pass


__all__ = ['OptimizedModule', 'eval_frame', 'allow_in_graph', 'disable', 'forbid_in_graph', 'mark_static_address', 'reset']

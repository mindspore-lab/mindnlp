"""Stub for torch.library module."""


class Library:
    """Library stub for custom operator registration."""

    def __init__(self, ns, kind, dispatch_key=''):
        self.ns = ns
        self.kind = kind
        self.dispatch_key = dispatch_key

    def define(self, schema, alias_analysis=''):
        """Define a custom op - no-op stub."""
        def decorator(fn):
            return fn
        return decorator

    def impl(self, name, fn=None, dispatch_key=''):
        """Implement a custom op - no-op stub."""
        if fn is not None:
            return fn
        def decorator(func):
            return func
        return decorator

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def define(qualname, schema, *, lib=None, tags=()):
    """Define a custom operator - no-op stub."""
    pass


def impl(qualname, types, fn=None, *, lib=None):
    """Implement a custom operator - no-op stub."""
    def decorator(func):
        return func
    if fn is not None:
        return fn
    return decorator


def impl_abstract(qualname, fn=None, *, lib=None):
    """Register abstract implementation - no-op stub."""
    def decorator(func):
        return func
    if fn is not None:
        return fn
    return decorator


def register_fake(qualname, fn=None, *, lib=None):
    """Register fake/meta implementation - no-op stub decorator."""
    def decorator(func):
        return func
    if fn is not None:
        return fn
    return decorator


def custom_op(qualname, fn=None, *, mutates_args=(), device_types=None, schema=None):
    """Register custom op - no-op stub decorator."""
    def decorator(func):
        return func
    if fn is not None:
        return fn
    return decorator


def get_ctx():
    """Get library context - returns stub."""
    return _LibraryContext()


class _LibraryContext:
    """Library context stub."""

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Dispatch key constants
class _DispatchKey:
    CPU = 'CPU'
    CUDA = 'CUDA'
    Meta = 'Meta'
    Autograd = 'Autograd'
    CompositeExplicitAutograd = 'CompositeExplicitAutograd'
    CompositeImplicitAutograd = 'CompositeImplicitAutograd'


DispatchKey = _DispatchKey()


__all__ = [
    'Library',
    'define',
    'impl',
    'impl_abstract',
    'register_fake',
    'custom_op',
    'get_ctx',
    'DispatchKey',
]

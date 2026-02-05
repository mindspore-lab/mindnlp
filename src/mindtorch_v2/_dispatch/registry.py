"""Op registration system for dispatch."""

from typing import Callable, Dict, Optional, Tuple
from .keys import DispatchKey

# Global registry: (op_name, dispatch_key) -> implementation
_OP_REGISTRY: Dict[Tuple[str, DispatchKey], Callable] = {}


def register_op(op_name: str, dispatch_key: DispatchKey):
    """Decorator to register an op implementation for a dispatch key.

    Usage:
        @register_op("add", DispatchKey.Backend_CPU)
        def add_cpu(a, b):
            return prim_add(a, b)
    """
    def decorator(func: Callable) -> Callable:
        _OP_REGISTRY[(op_name, dispatch_key)] = func
        return func
    return decorator


def get_op_impl(op_name: str, dispatch_key: DispatchKey) -> Optional[Callable]:
    """Get the implementation of an op for a dispatch key."""
    return _OP_REGISTRY.get((op_name, dispatch_key))


def list_registered_ops() -> list:
    """List all registered (op_name, dispatch_key) pairs."""
    return list(_OP_REGISTRY.keys())


def clear_registry():
    """Clear all registered ops (for testing)."""
    _OP_REGISTRY.clear()

"""Op registry - maps op names to Op classes."""
from typing import Dict, Type, Optional, Any, Tuple
from .base import Op


_OP_REGISTRY: Dict[str, Type[Op]] = {}


def register_op(name: str):
    """Decorator to register an op class."""
    def decorator(op_class: Type[Op]):
        _OP_REGISTRY[name] = op_class
        return op_class
    return decorator


def get_op(name: str) -> Optional[Op]:
    """Get op instance by name."""
    op_class = _OP_REGISTRY.get(name)
    if op_class is None:
        return None
    return op_class()


def execute_op(name: str, *args, return_backward_info: bool = False, **kwargs) -> Any:
    """Execute an op by name.

    Args:
        name: Op name
        *args: Arguments to forward
        return_backward_info: If True, return (result, backward_info)
        **kwargs: Keyword arguments to forward

    Returns:
        result or (result, backward_info)
    """
    op = get_op(name)
    if op is None:
        raise NotImplementedError(f"Op '{name}' not registered")

    result = op.forward(*args, **kwargs)

    if return_backward_info:
        backward_info = {
            'op': op,
            'saved': args,
        }
        return result, backward_info

    return result


def _register_builtins():
    """Register all built-in ops.

    Currently unused - all ops go through backend dispatch (ascend.py / cpu.py).
    """
    pass

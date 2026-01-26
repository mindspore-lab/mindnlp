"""Dispatcher routes ops to correct implementations based on dispatch keys."""

from typing import Any, Tuple
from .keys import DispatchKey
from .registry import get_op_impl


def _get_backend_key(args: Tuple) -> DispatchKey:
    """Determine backend dispatch key from arguments.

    Looks at tensor arguments to determine which backend to use.
    """
    from .._tensor import Tensor

    for arg in args:
        if isinstance(arg, Tensor):
            device_type = arg.device.type
            if device_type == "cpu":
                return DispatchKey.Backend_CPU
            elif device_type == "cuda":
                return DispatchKey.Backend_CUDA
            elif device_type in ("npu", "ascend"):
                return DispatchKey.Backend_Ascend

    # Default to CPU
    return DispatchKey.Backend_CPU


def dispatch(op_name: str, *args, **kwargs) -> Any:
    """Dispatch an operation to the correct implementation.

    Args:
        op_name: Name of the operation (e.g., "add", "matmul")
        *args: Positional arguments to pass to the op
        **kwargs: Keyword arguments to pass to the op

    Returns:
        Result of the operation

    Raises:
        NotImplementedError: If no implementation found for the op
    """
    # Determine which backend to use
    backend_key = _get_backend_key(args)

    # Try to get implementation for this backend
    impl = get_op_impl(op_name, backend_key)

    if impl is None:
        # Try composite fallback
        impl = get_op_impl(op_name, DispatchKey.CompositeExplicit)

    if impl is None:
        raise NotImplementedError(
            f"No implementation found for op '{op_name}' with dispatch key {backend_key}"
        )

    return impl(*args, **kwargs)

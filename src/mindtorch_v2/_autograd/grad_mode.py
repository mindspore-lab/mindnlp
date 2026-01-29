"""Gradient mode context managers.

This module provides context managers and decorators for controlling
gradient computation in mindtorch_v2.
"""

import threading
from functools import wraps

# Thread-local storage for gradient mode
_grad_mode = threading.local()


def _get_grad_mode() -> bool:
    """Get current gradient mode (default True)."""
    if not hasattr(_grad_mode, 'enabled'):
        _grad_mode.enabled = True
    return _grad_mode.enabled


def _set_grad_mode(enabled: bool) -> None:
    """Set current gradient mode."""
    _grad_mode.enabled = enabled


def is_grad_enabled() -> bool:
    """Returns True if gradient computation is currently enabled.

    Returns:
        bool: True if gradients are enabled, False otherwise.

    Example:
        >>> import mindtorch_v2 as torch
        >>> torch.is_grad_enabled()
        True
        >>> with torch.no_grad():
        ...     torch.is_grad_enabled()
        False
    """
    return _get_grad_mode()


class set_grad_enabled:
    """Context manager / decorator to set gradient computation on or off.

    This context manager allows temporarily enabling or disabling gradient
    computation. It can be used as both a context manager and a decorator.

    Args:
        mode (bool): Whether to enable or disable gradient computation.

    Example:
        >>> import mindtorch_v2 as torch
        >>> # As context manager
        >>> with torch.set_grad_enabled(False):
        ...     # gradient computation disabled here
        ...     pass
        >>> # As decorator
        >>> @torch.set_grad_enabled(False)
        ... def my_func():
        ...     pass
    """

    def __init__(self, mode: bool):
        self.prev = _get_grad_mode()
        self.mode = mode

    def __enter__(self):
        _set_grad_mode(self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_mode(self.prev)
        return False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class no_grad(set_grad_enabled):
    """Context manager / decorator that disables gradient computation.

    Disabling gradient computation is useful for inference, when you are sure
    that you will not call backward(). It will reduce memory consumption for
    computations that would otherwise have requires_grad=True.

    This context manager is thread local; it will not affect computation
    in other threads.

    Can be used with or without parentheses:
        @torch.no_grad
        def func(): ...

        @torch.no_grad()
        def func(): ...

    Example:
        >>> import mindtorch_v2 as torch
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def inference_function(x):
        ...     return x * 2
    """

    def __init__(self, func=None):
        super().__init__(False)
        self._func = func
        if func is not None:
            # Used as @torch.no_grad without parentheses
            self._wrapped = self._wrap_func(func)

    def _wrap_func(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return wrapper

    def __call__(self, func_or_args=None, *args, **kwargs):
        if self._func is not None:
            # We were used as @torch.no_grad (without parens), now being called
            return self._wrapped(func_or_args, *args, **kwargs)
        elif callable(func_or_args):
            # Used as @torch.no_grad() and now decorating a function
            @wraps(func_or_args)
            def wrapper(*args, **kwargs):
                with self:
                    return func_or_args(*args, **kwargs)
            return wrapper
        else:
            # Regular call through context manager
            raise TypeError("no_grad() takes no arguments")


class enable_grad(set_grad_enabled):
    """Context manager / decorator that enables gradient computation.

    Enables gradient computation, if it has been disabled via no_grad
    or set_grad_enabled.

    This context manager is thread local; it will not affect computation
    in other threads.

    Can be used with or without parentheses:
        @torch.enable_grad
        def func(): ...

        @torch.enable_grad()
        def func(): ...

    Example:
        >>> import mindtorch_v2 as torch
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     with torch.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True
    """

    def __init__(self, func=None):
        super().__init__(True)
        self._func = func
        if func is not None:
            # Used as @torch.enable_grad without parentheses
            self._wrapped = self._wrap_func(func)

    def _wrap_func(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return wrapper

    def __call__(self, func_or_args=None, *args, **kwargs):
        if self._func is not None:
            # We were used as @torch.enable_grad (without parens), now being called
            return self._wrapped(func_or_args, *args, **kwargs)
        elif callable(func_or_args):
            # Used as @torch.enable_grad() and now decorating a function
            @wraps(func_or_args)
            def wrapper(*args, **kwargs):
                with self:
                    return func_or_args(*args, **kwargs)
            return wrapper
        else:
            # Regular call through context manager
            raise TypeError("enable_grad() takes no arguments")

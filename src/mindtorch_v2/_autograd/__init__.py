"""Autograd module for mindtorch_v2.

This module provides automatic differentiation functionality including:
- Gradient mode context managers (no_grad, enable_grad, set_grad_enabled)
- Node base class for autograd graph nodes
- Backward engine for computing gradients
- Backward functions for ops
"""

from .grad_mode import (
    is_grad_enabled,
    set_grad_enabled,
    no_grad,
    enable_grad,
)
from .node import Node, AccumulateGrad
from .engine import backward
from . import functions


class FunctionCtx:
    """Context object for autograd Function."""
    def __init__(self):
        self._saved_tensors = ()
        self.needs_input_grad = ()

    def save_for_backward(self, *tensors):
        self._saved_tensors = tensors

    @property
    def saved_tensors(self):
        return self._saved_tensors

    def mark_dirty(self, *args):
        pass

    def mark_non_differentiable(self, *args):
        pass

    def set_materialize_grads(self, value):
        pass


class FunctionMeta(type):
    """Metaclass for Function that enables static method calling."""
    def __call__(cls, *args, **kwargs):
        raise RuntimeError(
            "Legacy autograd Function is not supported. Use Function.apply instead."
        )


class Function(metaclass=FunctionMeta):
    """Base class for autograd Functions.

    Subclasses should implement forward() and backward() as static methods.
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = FunctionCtx()
        result = cls.forward(ctx, *args, **kwargs)
        return result


__all__ = [
    'is_grad_enabled',
    'set_grad_enabled',
    'no_grad',
    'enable_grad',
    'Node',
    'AccumulateGrad',
    'backward',
    'functions',
    'Function',
    'FunctionCtx',
]

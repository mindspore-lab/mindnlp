# src/mindtorch_v2/_autograd/node.py
"""Autograd Node - base class for gradient functions."""

from typing import Tuple, Optional, Any


class Node:
    """Base class for all autograd graph nodes (grad_fn).

    Each node represents an operation in the computational graph.
    During backward, nodes compute gradients for their inputs.

    Attributes:
        _next_functions: Tuple of (Node, output_idx) pairs linking to input nodes
        _saved_tensors: Tensors saved for backward computation
        _needs_input_grad: Tuple of bools indicating which inputs need gradients
    """

    __slots__ = ('_next_functions', '_saved_tensors', '_needs_input_grad', '_name')

    def __init__(self):
        self._next_functions: Tuple[Tuple[Optional['Node'], int], ...] = ()
        self._saved_tensors: Tuple[Any, ...] = ()
        self._needs_input_grad: Tuple[bool, ...] = ()
        self._name: str = self.__class__.__name__

    @property
    def next_functions(self) -> Tuple[Tuple[Optional['Node'], int], ...]:
        """Return the next functions in the graph (inputs to this op)."""
        return self._next_functions

    @property
    def name(self) -> str:
        """Return the name of this node."""
        return self._name

    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass."""
        self._saved_tensors = tensors

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return tensors saved for backward."""
        return self._saved_tensors

    def backward(self, grad_outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Compute gradients for inputs given gradients of outputs.

        Must be overridden by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.backward is not implemented"
        )

    def __repr__(self):
        return f"<{self._name}>"


class AccumulateGrad(Node):
    """Special node for leaf tensors that accumulates gradients.

    This node doesn't have a backward() - instead, the backward engine
    directly accumulates gradients into the tensor's .grad attribute.
    """

    __slots__ = ('_variable',)

    def __init__(self, variable):
        super().__init__()
        self._variable = variable
        self._name = "AccumulateGrad"

    @property
    def variable(self):
        return self._variable

    def backward(self, grad_outputs):
        raise RuntimeError("AccumulateGrad.backward should not be called directly")

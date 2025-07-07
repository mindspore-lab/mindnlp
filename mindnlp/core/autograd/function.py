"""functional autograd"""
from collections.abc import Generator
from dataclasses import dataclass
from typing import Tuple, Any, Optional, Type, Sequence
import functools

@dataclass(unsafe_hash=True)
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, *grad_out):
        return cls.backward(ctx, *grad_out)  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps, **kwargs):
        return cls.forward(ctx, *inps, **kwargs)  # type: ignore

    @classmethod
    def apply(cls, *vals, **kwargs):
        # Create the context.
        ctx = Context(not requires_grad)
        # Call forward with the variables.
        results = cls._forward(ctx, *vals, **kwargs)
        requires_grad = any([x.requires_grad for x in vals])

        if requires_grad: # cut useless nodes
            generation = max([x.generation for x in vals])
            ctx.outputs = [weakref.ref(output) for output in outputs]
            back = History(cls, ctx, generation)
            for output in outputs:
                output.set_creator(back)

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

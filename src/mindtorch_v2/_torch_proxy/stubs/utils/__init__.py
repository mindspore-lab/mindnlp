"""Stub for torch.utils module."""

from types import ModuleType
import sys


class checkpoint:
    """Stub for torch.utils.checkpoint."""

    @staticmethod
    def checkpoint(fn, *args, use_reentrant=True, **kwargs):
        """Run function without gradient checkpointing."""
        return fn(*args, **kwargs)

    @staticmethod
    def checkpoint_sequential(functions, segments, *inputs, **kwargs):
        """Run sequential functions without checkpointing."""
        for fn in functions:
            inputs = fn(*inputs) if isinstance(inputs, tuple) else fn(inputs)
        return inputs


# Import data as proper submodule
from . import data


class hooks:
    """Stub for torch.utils.hooks."""

    class RemovableHandle:
        def __init__(self):
            pass

        def remove(self):
            pass


# Import _pytree as submodule
from . import _pytree


# Make this module have the submodules accessible
_pytree_module = _pytree

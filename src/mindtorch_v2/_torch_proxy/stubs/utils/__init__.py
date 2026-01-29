"""Stub for torch.utils module."""

from types import ModuleType
import sys

# Import checkpoint as proper submodule
from . import checkpoint

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

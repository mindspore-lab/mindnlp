"""Torch proxy system for mindtorch_v2.

This module provides a proxy loader that intercepts `import torch`
and returns mindtorch_v2 instead, enabling PyTorch-based libraries
to work with mindtorch_v2 as the backend.

Usage:
    from mindtorch_v2._torch_proxy import install
    install()  # Now `import torch` returns mindtorch_v2

    # PyTorch-based libraries now use mindtorch_v2
"""

from .finder import MindTorchV2Finder
from .loader import MindTorchV2Loader

_installed = False


def install():
    """Install the torch proxy into sys.meta_path.

    After calling this, `import torch` will return mindtorch_v2.
    """
    global _installed
    if _installed:
        return

    import sys

    # Insert our finder at the beginning of meta_path
    finder = MindTorchV2Finder()
    sys.meta_path.insert(0, finder)
    _installed = True


def uninstall():
    """Remove the torch proxy from sys.meta_path."""
    global _installed
    if not _installed:
        return

    import sys

    # Remove all MindTorchV2Finder instances
    sys.meta_path = [f for f in sys.meta_path if not isinstance(f, MindTorchV2Finder)]

    # Clear torch modules from sys.modules
    torch_modules = [k for k in sys.modules.keys() if k == 'torch' or k.startswith('torch.')]
    for mod in torch_modules:
        del sys.modules[mod]

    _installed = False

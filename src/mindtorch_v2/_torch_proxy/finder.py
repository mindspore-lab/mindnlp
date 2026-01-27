"""MindTorchV2Finder - Intercepts torch imports."""

import sys


class MindTorchV2Finder:
    """Meta path finder that intercepts all torch.* imports.

    When installed in sys.meta_path, this finder will catch any
    import of torch or torch.* and delegate to MindTorchV2Loader.
    """

    def find_module(self, fullname, path=None):
        """Find module hook for the import system.

        Args:
            fullname: Full module name being imported (e.g., 'torch.nn')
            path: Optional path for the module

        Returns:
            MindTorchV2Loader if this is a torch import, None otherwise
        """
        if fullname == 'torch' or fullname.startswith('torch.'):
            from .loader import MindTorchV2Loader
            return MindTorchV2Loader()
        return None

    def find_spec(self, fullname, path=None, target=None):
        """Find spec hook for the import system (Python 3.4+).

        Returns None to indicate we handle this via find_module/load_module.
        """
        if fullname == 'torch' or fullname.startswith('torch.'):
            from importlib.machinery import ModuleSpec
            from .loader import MindTorchV2Loader
            return ModuleSpec(fullname, MindTorchV2Loader())
        return None

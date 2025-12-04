"""
Common utilities for patch system
"""

import sys
from typing import Optional


class MissingLibraryErrorModule:
    """A module that raises a friendly error when the library is not installed."""
    
    def __init__(self, library_name: str):
        self._library_name = library_name
        self.__name__ = f"mindnlp.{library_name}"
    
    def __getattr__(self, name: str):
        raise ImportError(
            f"`{self._library_name}` is required but not installed. "
            f"Please install it with: `pip install {self._library_name}`\n"
            f"Note: The usage 'from mindnlp.{self._library_name} import ...' is deprecated. "
            f"Please use 'import mindnlp; from {self._library_name} import ...' instead."
        )
    
    def __dir__(self):
        return []


def setup_missing_library_error_module(library_name: str, module_name: Optional[str] = None):
    """
    Set up an error module when a library is not installed.
    
    Args:
        library_name: The name of the library (e.g., 'transformers', 'diffusers')
        module_name: The full module name in sys.modules (e.g., 'mindnlp.transformers').
                    If None, defaults to f'mindnlp.{library_name}'
    """
    if module_name is None:
        module_name = f'mindnlp.{library_name}'
    
    if module_name not in sys.modules:
        sys.modules[module_name] = MissingLibraryErrorModule(library_name)

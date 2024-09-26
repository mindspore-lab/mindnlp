"""accelerate import utilities."""
from mindnlp.utils.import_utils import _is_package_available


def is_mindformers_available():
    """
    Checks if the MindFormers library is available in the current environment.
    
    Returns:
        bool: True if MindFormers library is available, False otherwise.
    """
    _mindformers_available = _is_package_available("mindformers")
    return _mindformers_available

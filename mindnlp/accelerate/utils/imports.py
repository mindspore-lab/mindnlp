"""accelerate import utilities."""
from mindnlp.utils.import_utils import _is_package_available


def is_mindformers_available():
    """
    Checks if the MindFormers library is available in the current environment.
    
    Returns:
        bool: True if MindFormers library is available, False otherwise.
    """
    _, _mindformers_available = _is_package_available(
        "mindformers", return_version=True
    )
    return _mindformers_available

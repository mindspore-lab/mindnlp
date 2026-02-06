"""Backend implementations for different devices."""

from ..configs import DEVICE_TARGET
from . import cpu

__all__ = ['cpu']

# Conditionally import Ascend backend if running on Ascend
if DEVICE_TARGET == 'Ascend':
    try:
        from . import ascend
        __all__.append('ascend')
    except ImportError:
        # Ascend backend not available (e.g., running on CPU-only system)
        pass

"""Backend implementations for different devices."""

from ..configs import DEVICE_TARGET
from . import cpu

# Conditionally import Ascend backend if running on Ascend
if DEVICE_TARGET == 'Ascend':
    from . import ascend

__all__ = ['cpu']
if DEVICE_TARGET == 'Ascend':
    __all__.append('ascend')

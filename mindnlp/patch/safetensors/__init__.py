"""
Safetensors patches

This module automatically registers and applies patches for safetensors library.
Importing this module will trigger patch registration.
"""

from ..registry import register_safetensors_patch, apply_safetensors_patches

# Import patches to trigger registration
from . import common


def setup_safetensors_module():
    """Setup safetensors patches"""
    apply_safetensors_patches()

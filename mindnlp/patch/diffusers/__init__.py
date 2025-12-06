"""
Diffusers patches

This module automatically registers and applies patches for diffusers library.
"""

import sys
from ..registry import register_diffusers_patch, apply_diffusers_patches
from ..common import setup_missing_library_error_module
from . import common


def setup_diffusers_module():
    """Setup diffusers patches"""
    try:
        import diffusers
    except ImportError:
        setup_missing_library_error_module('diffusers')
        return
    
    apply_diffusers_patches()


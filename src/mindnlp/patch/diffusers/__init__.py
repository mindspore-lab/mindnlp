"""
Diffusers patches

This module automatically registers and applies patches for diffusers library.
"""

import sys
from ..registry import register_diffusers_patch, apply_diffusers_patches
from ..common import setup_missing_library_error_module
from . import common


def setup_diffusers_module():
    """Setup mindnlp.diffusers module to redirect to patched diffusers"""
    try:
        import diffusers
    except ImportError:
        setup_missing_library_error_module('diffusers', 'mindnlp.diffusers')
        return

    apply_diffusers_patches()

    # Redirect mindnlp.diffusers to diffusers
    if 'mindnlp.diffusers' not in sys.modules:
        sys.modules['mindnlp.diffusers'] = diffusers

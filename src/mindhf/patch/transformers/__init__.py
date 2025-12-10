"""
Transformers patches

This module automatically registers and applies patches for transformers library.
Importing this module will trigger patch registration.
"""

import sys
from ..registry import register_transformers_patch, apply_transformers_patches
from ..common import setup_missing_library_error_module
from mindhf.utils.import_utils import _LazyModule

# Import version-specific patches to trigger registration
from . import common, v4_55, v4_56


def setup_transformers_module():
    """Setup mindhf.transformers and mindnlp.transformers modules to redirect to patched transformers"""
    try:
        import transformers
    except ImportError:
        setup_missing_library_error_module('transformers', 'mindhf.transformers')
        setup_missing_library_error_module('transformers', 'mindnlp.transformers')
        return
    
    # Apply patches first (if not already applied)
    apply_transformers_patches()
    
    # Create lazy module for transformers
    lazy_module = _LazyModule(
        'transformers',
        transformers.__file__,
        transformers._import_structure,
        module_spec=None,  # Will be set by _LazyModule
        extra_objects={"__version__": transformers.__version__},
    )
    
    # Redirect mindhf.transformers to transformers
    transformers_module_name_hf = 'mindhf.transformers'
    if transformers_module_name_hf not in sys.modules or not isinstance(sys.modules[transformers_module_name_hf], _LazyModule):
        sys.modules[transformers_module_name_hf] = lazy_module
    
    # Redirect mindnlp.transformers to transformers for backward compatibility
    transformers_module_name_nlp = 'mindnlp.transformers'
    if transformers_module_name_nlp not in sys.modules or not isinstance(sys.modules[transformers_module_name_nlp], _LazyModule):
        sys.modules[transformers_module_name_nlp] = lazy_module


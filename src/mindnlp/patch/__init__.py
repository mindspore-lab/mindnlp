"""
MindNLP Patch System

This module provides a versioned patch system for HuggingFace libraries.
Patches are automatically applied when importing mindnlp.
"""

from .registry import apply_all_patches

__all__ = ['apply_all_patches']

# Patch Utils

Utility functions for the MindNLP patch system.

## Overview

This module provides internal utilities used by the patch system to manage version compatibility and apply patches correctly.

## Functions

### Version Management

The patch system uses version-aware patching to ensure compatibility across different versions of HuggingFace libraries.

### Patch Registry

Internal registry for managing which patches are applied:

- Track applied patches
- Handle patch dependencies
- Manage patch order

## Usage

These utilities are primarily used internally. Users typically don't need to interact with them directly - patches are applied automatically when importing `mindnlp`.

```python
import mindnlp  # All patches are applied here

# Check if patches were applied
from mindnlp.patch import apply_all_patches
# Patches are already applied, this is a no-op
```

## Notes

- Patches are version-specific to maintain compatibility
- The system handles graceful degradation for unsupported versions
- Debug information is available through logging

# Utils

MindNLP utilities module provides common helper functions and utilities.

## Overview

The utils module contains:

- **Generic utilities**: Common helper functions
- **Import utilities**: Dynamic import helpers for checking library availability

## Usage

```python
from mindnlp.utils import is_mindspore_available

if is_mindspore_available():
    import mindspore
```

## Import Utilities

The import utilities help check for optional dependencies:

- `is_mindspore_available()` - Check if MindSpore is installed
- `is_torch_available()` - Check if PyTorch is installed (for compatibility)

## Constants

- `DUMMY_INPUTS` - Example input IDs for testing
- `DUMMY_MASK` - Example attention masks for testing
- `SENTENCEPIECE_UNDERLINE` - The sentencepiece underscore character

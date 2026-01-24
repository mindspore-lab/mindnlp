# Safetensors Patch

This module patches the `safetensors` library to work with MindSpore tensors.

## Overview

Safetensors is a secure and fast format for storing and loading model weights. MindNLP patches safetensors to support loading weights directly into MindSpore tensors.

## Usage

```python
import mindnlp  # Patches are applied automatically

# Now safetensors works with MindSpore
from safetensors import safe_open
from safetensors.mindspore import load_file, save_file

# Load safetensors file
tensors = load_file("model.safetensors")

# Save tensors to safetensors format
save_file(tensors, "output.safetensors")
```

## Key Functions

### load_file

Load tensors from a safetensors file:

```python
from safetensors.mindspore import load_file

# Load all tensors
tensors = load_file("model.safetensors")

# Access individual tensors
weight = tensors["model.weight"]
```

### save_file

Save tensors to a safetensors file:

```python
from safetensors.mindspore import save_file
import mindspore

tensors = {
    "weight": mindspore.Tensor([1.0, 2.0, 3.0]),
    "bias": mindspore.Tensor([0.1, 0.2, 0.3])
}

save_file(tensors, "output.safetensors")
```

## Notes

- Safetensors format is the recommended format for storing model weights
- It provides security against arbitrary code execution (unlike pickle)
- Loading is memory-mapped for efficient large model handling

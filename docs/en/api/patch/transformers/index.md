# Transformers Patch

This module patches HuggingFace Transformers to work with MindSpore as the backend.

## Overview

When you import `mindnlp`, it automatically patches the `transformers` library to use MindSpore operations instead of PyTorch. This enables you to use all HuggingFace models directly.

## Usage

```python
import mindspore
import mindnlp  # Patches are applied automatically

# Now use transformers as usual
from transformers import AutoModel, AutoTokenizer, pipeline

# Load any model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Use pipelines
pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B", ms_dtype=mindspore.float16)
```

## Key Features

### Model Loading

All `from_pretrained()` methods work seamlessly:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    ms_dtype=mindspore.bfloat16,  # MindSpore dtype
    device_map="auto"
)
```

### Pipelines

All transformers pipelines are supported:

- `text-generation`
- `text-classification`
- `token-classification`
- `question-answering`
- `fill-mask`
- `summarization`
- `translation`
- And more...

### Special Parameters

MindNLP adds support for MindSpore-specific parameters:

- `ms_dtype`: Specify MindSpore data type (e.g., `mindspore.float16`, `mindspore.bfloat16`)
- `device_map`: Device placement strategy ("auto", "cuda", "cpu")

## Patched Components

The patch system modifies:

- Model classes (Auto*, *Model, *ForCausalLM, etc.)
- Tokenizers
- Pipelines
- Generation utilities
- Configuration classes

## Notes

- For training, consider using `mindnlp.engine.Trainer` for optimal gradient computation
- The patch is automatically applied on `import mindnlp`
- All 200,000+ models on HuggingFace Hub are supported

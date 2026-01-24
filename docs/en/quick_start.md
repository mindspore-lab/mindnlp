# Quick Start

Welcome to MindNLP! This page will help you get started quickly.

## Getting Started

For a comprehensive guide on how to use MindNLP, including loading pretrained models and fine-tuning them for your specific tasks, please visit our detailed tutorial:

**[Quick Start Tutorial](tutorials/quick_start.md)** - Learn how to fine-tune BERT for sentiment classification

## Quick Examples

### Using HuggingFace Transformers with MindSpore

```python
import mindspore
import mindnlp
from transformers import pipeline

# Create a text generation pipeline with Qwen
pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen3-8B",
    ms_dtype=mindspore.bfloat16,
    device_map="auto"
)

chat = [
    {"role": "user", "content": "Hello, how are you?"}
]
response = pipe(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

### Using MindNLP Native Interface

```python
from mindnlp.transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="ms")
outputs = model(**inputs)
```

## More Tutorials

- [Use Trainer](tutorials/use_trainer.md) - Training with MindNLP's Trainer API
- [PEFT/LoRA](tutorials/peft.md) - Parameter-efficient fine-tuning
- [Data Preprocessing](tutorials/data_preprocess.md) - Dataset handling and processing
- [Use Mirror](tutorials/use_mirror.md) - Using model mirrors for faster downloads

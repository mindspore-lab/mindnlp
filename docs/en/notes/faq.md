# Frequently Asked Questions

## Installation

### What versions of MindSpore are supported?

| MindNLP Version | MindSpore Version | Python Version |
|-----------------|-------------------|----------------|
| 0.6.x           | >=2.7.1           | 3.10-3.11      |
| 0.5.x           | 2.5.0-2.7.0       | 3.10-3.11      |
| 0.4.x           | 2.2.x-2.5.0       | 3.9-3.11       |
| 0.3.x           | 2.1.0-2.3.1       | 3.8-3.9        |

### What platforms are supported?

MindNLP supports:

- **Linux**: Ubuntu 18.04/20.04/22.04 (recommended)
- **macOS**: Intel and Apple Silicon (CPU only)
- **Windows**: Windows 10/11 (limited support)

Hardware accelerators:

- **Ascend NPU**: Full support on Linux
- **NVIDIA GPU**: CUDA 11.x support on Linux
- **CPU**: All platforms

### How do I install MindNLP?

```bash
# From PyPI (recommended)
pip install mindnlp

# From source
pip install git+https://github.com/mindspore-lab/mindnlp.git

# Daily build
# Download from: https://repo.mindspore.cn/mindspore-lab/mindnlp/newest/any/
```

### Installation fails with MindSpore errors

Ensure MindSpore is properly installed first:

```bash
# Check MindSpore installation
python -c "import mindspore; print(mindspore.__version__)"

# Install MindSpore if needed (example for CPU)
pip install mindspore
```

See [MindSpore installation guide](https://www.mindspore.cn/install) for platform-specific instructions.

## Model Compatibility

### Does MindNLP support all HuggingFace models?

Yes! MindNLP provides full compatibility with the HuggingFace ecosystem through its patching mechanism. When you import `mindnlp`, it automatically patches `transformers` and `diffusers` to use MindSpore as the backend.

```python
import mindspore
import mindnlp  # This patches HuggingFace libraries
from transformers import AutoModel

# Now loads with MindSpore backend
model = AutoModel.from_pretrained("bert-base-uncased")
```

### What's the difference between inference and training support?

**Inference**: All HuggingFace models work for inference out of the box.

**Training**: For training, use MindNLP's native APIs (`mindnlp.transformers`) to ensure proper gradient computation:

```python
# For training, use MindNLP APIs
from mindnlp.transformers import AutoModelForSequenceClassification
from mindnlp.engine import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# ... training code
```

### How do I use a specific model precision?

Use the `ms_dtype` parameter:

```python
import mindspore
from transformers import AutoModel

# Load in float16
model = AutoModel.from_pretrained(
    "model-name",
    ms_dtype=mindspore.float16
)

# Load in bfloat16
model = AutoModel.from_pretrained(
    "model-name",
    ms_dtype=mindspore.bfloat16
)
```

### Model loading is slow or fails

1. **Check network connectivity**: Model weights are downloaded from HuggingFace Hub
2. **Use a mirror**: See [Use Mirror tutorial](../tutorials/use_mirror.md) for faster downloads in China
3. **Check disk space**: Large models require significant storage
4. **Try a smaller model first**: Verify your setup works with a small model like `bert-base-uncased`

## Training Issues

### How do I fine-tune a model?

Use the MindNLP Trainer API:

```python
from mindnlp.transformers import AutoModelForSequenceClassification
from mindnlp.engine import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

See [Quick Start Tutorial](../tutorials/quick_start.md) for a complete example.

### Training runs out of memory

1. **Reduce batch size**: Lower `per_device_train_batch_size`
2. **Use gradient accumulation**: Set `gradient_accumulation_steps`
3. **Use mixed precision**: Set `ms_dtype=mindspore.float16`
4. **Use PEFT/LoRA**: See [PEFT tutorial](../tutorials/peft.md) for parameter-efficient fine-tuning

### Gradients are not computed correctly

Ensure you're using MindNLP's native APIs for training:

```python
# Correct: Use mindnlp.transformers for training
from mindnlp.transformers import AutoModel

# Incorrect: Using transformers directly may have gradient issues
from transformers import AutoModel  # Only for inference
```

## Common Errors

### `RuntimeError: Device mismatch`

Ensure tensors are on the same device:

```python
import mindspore
mindspore.set_context(device_target="CPU")  # or "GPU", "Ascend"
```

### `KeyError: 'model_type'`

The model configuration may be incompatible. Try:

```python
from mindnlp.transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("model-name")
model = AutoModel.from_config(config)
```

### `ImportError: cannot import name 'xxx'`

1. Update MindNLP to the latest version: `pip install -U mindnlp`
2. Ensure MindSpore version is compatible (see version table above)
3. Check if the feature is available in your version

### Model outputs differ from PyTorch

Small numerical differences are expected due to different backends. For significant differences:

1. Ensure you're using the same model weights
2. Check input preprocessing is identical
3. Verify model configuration matches

## Performance

### How do I speed up inference?

1. **Use GPU/Ascend**: Set appropriate device context
2. **Use lower precision**: `ms_dtype=mindspore.float16`
3. **Enable graph mode**: `mindspore.set_context(mode=mindspore.GRAPH_MODE)`
4. **Batch your inputs**: Process multiple samples together

### How do I enable distributed training?

MindNLP supports MindSpore's distributed training capabilities:

```python
from mindspore.communication import init
from mindspore import set_auto_parallel_context

init()
set_auto_parallel_context(parallel_mode="data_parallel")
```

## Getting Help

- **GitHub Issues**: [mindspore-lab/mindnlp/issues](https://github.com/mindspore-lab/mindnlp/issues)
- **Documentation**: [mindnlp.cqu.ai](https://mindnlp.cqu.ai)
- **MindSpore Forum**: [bbs.huaweicloud.com/forum/forum-1076-1.html](https://bbs.huaweicloud.com/forum/forum-1076-1.html)

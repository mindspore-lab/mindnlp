---
hide:
  - navigation
---

# Supported Models

## HuggingFace Compatibility

MindNLP provides **full compatibility with all HuggingFace Transformers and Diffusers models** through its patching mechanism. When you import `mindnlp`, it automatically patches the HuggingFace libraries to use MindSpore as the backend.

```python
import mindspore
import mindnlp  # Patches HuggingFace libraries

# Now any HuggingFace model works with MindSpore
from transformers import AutoModel
model = AutoModel.from_pretrained("any-model-on-huggingface")
```

This means you can use:

- **All 200,000+ models** on the [HuggingFace Model Hub](https://huggingface.co/models)
- **Latest model architectures** as soon as they're released in transformers
- **Diffusion models** through the diffusers library

## Usage Modes

### Inference Mode

For inference, you can use HuggingFace APIs directly after importing `mindnlp`:

```python
import mindspore
import mindnlp
from transformers import pipeline

# Text generation
pipe = pipeline("text-generation", model="Qwen/Qwen3-8B", ms_dtype=mindspore.bfloat16)
result = pipe("Hello, how are you?")

# Image generation
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
image = pipe("A beautiful sunset").images[0]
```

### Training Mode

For training, use MindNLP's native APIs to ensure proper gradient computation:

```python
from mindnlp.transformers import AutoModelForSequenceClassification
from mindnlp.engine import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

## Verified Models

The following models have been extensively tested with MindNLP. However, **all models in the transformers library are supported**.

### Large Language Models (LLMs)

| Model Family | Inference | Training | Notes |
|-------------|:---------:|:--------:|-------|
| Qwen/Qwen2/Qwen3 | ✅ | ✅ | Full series support |
| Llama/Llama2/Llama3 | ✅ | ✅ | Including CodeLlama |
| ChatGLM/GLM | ✅ | ✅ | ChatGLM, ChatGLM2, ChatGLM3 |
| Mistral/Mixtral | ✅ | ✅ | MoE models supported |
| Phi-2/Phi-3 | ✅ | ✅ | |
| Gemma | ✅ | ✅ | |
| BLOOM | ✅ | ✅ | |
| Falcon | ✅ | ✅ | |
| GPT-2/GPT-Neo/GPT-NeoX | ✅ | ✅ | |
| OPT | ✅ | ✅ | |
| Mamba | ✅ | ✅ | |
| RWKV | ✅ | ✅ | |

### Encoder Models

| Model | Inference | Training | Notes |
|-------|:---------:|:--------:|-------|
| BERT | ✅ | ✅ | All variants |
| RoBERTa | ✅ | ✅ | |
| ALBERT | ✅ | ✅ | |
| DeBERTa | ✅ | ✅ | |
| ELECTRA | ✅ | ✅ | |
| XLNet | ✅ | ✅ | |
| Longformer | ✅ | ✅ | |
| BigBird | ✅ | ✅ | |

### Encoder-Decoder Models

| Model | Inference | Training | Notes |
|-------|:---------:|:--------:|-------|
| T5 | ✅ | ✅ | Including mT5, Flan-T5 |
| BART | ✅ | ✅ | |
| mBART | ✅ | ✅ | |
| Pegasus | ✅ | ✅ | |

### Vision Models

| Model | Inference | Training | Notes |
|-------|:---------:|:--------:|-------|
| ViT | ✅ | ✅ | |
| CLIP | ✅ | ✅ | |
| BLIP/BLIP-2 | ✅ | ✅ | |
| SAM | ✅ | ⚠️ | Inference recommended |
| Swin Transformer | ✅ | ✅ | |
| ConvNeXt | ✅ | ✅ | |
| ResNet | ✅ | ✅ | |

### Multimodal Models

| Model | Inference | Training | Notes |
|-------|:---------:|:--------:|-------|
| LLaVA | ✅ | ⚠️ | |
| Qwen-VL | ✅ | ✅ | |
| ALIGN | ✅ | ⚠️ | |
| AltCLIP | ✅ | ⚠️ | |
| BridgeTower | ✅ | ⚠️ | |

### Audio Models

| Model | Inference | Training | Notes |
|-------|:---------:|:--------:|-------|
| Whisper | ✅ | ✅ | |
| Wav2Vec2 | ✅ | ✅ | |
| HuBERT | ✅ | ✅ | |
| MusicGen | ✅ | ⚠️ | |
| Bark | ✅ | ⚠️ | |

### Diffusion Models

| Model | Inference | Training | Notes |
|-------|:---------:|:--------:|-------|
| Stable Diffusion | ✅ | ⚠️ | v1.5, v2.x, XL |
| SDXL | ✅ | ⚠️ | |
| ControlNet | ✅ | ⚠️ | |
| LoRA adapters | ✅ | ✅ | |

**Legend:**
- ✅ Full support
- ⚠️ Experimental or inference-only recommended

## Device Support

| Device | Status | Notes |
|--------|:------:|-------|
| CPU | ✅ | All platforms |
| NVIDIA GPU | ✅ | CUDA 11.x on Linux |
| Ascend NPU | ✅ | Full support on Linux |

## Known Limitations

1. **Graph Mode**: Some dynamic models may require PyNative mode
2. **Custom CUDA kernels**: Models with custom CUDA ops may need adaptation
3. **Quantization**: Some quantization methods may have limited support

## Reporting Issues

If you encounter issues with a specific model:

1. Check [GitHub Issues](https://github.com/mindspore-lab/mindnlp/issues) for known issues
2. Try updating to the latest version: `pip install -U mindnlp`
3. Report new issues with model name, error message, and reproduction steps

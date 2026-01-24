---
hide:
  - navigation
---

<p align="center">
  <span style="font-size: 3rem; font-weight: bold;">🚀 MindNLP</span>
</p>

<p align="center">
  <strong>Run HuggingFace Models on MindSpore with Zero Code Changes</strong>
</p>

<p align="center">
  <em>The easiest way to use 200,000+ HuggingFace models on Ascend NPU, GPU, and CPU</em>
</p>

<p align="center">
  <a href="https://github.com/mindspore-lab/mindnlp/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/mindspore-lab/mindnlp?style=for-the-badge&logo=github&color=yellow">
  </a>
  <a href="https://pypi.org/project/mindnlp/">
    <img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/mindnlp?style=for-the-badge&logo=pypi&color=blue">
  </a>
  <a href="https://github.com/mindspore-lab/mindnlp/blob/master/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/mindspore-lab/mindnlp?style=for-the-badge&color=green">
  </a>
</p>

---

## 🎯 What is MindNLP?

**MindNLP** bridges the gap between HuggingFace's massive model ecosystem and MindSpore's hardware acceleration. With just `import mindnlp`, you can run any HuggingFace model on **Ascend NPU**, **NVIDIA GPU**, or **CPU** - no code changes required.

```python
import mindnlp  # That's it! HuggingFace now runs on MindSpore
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B")
print(pipe("Hello, I am")[0]["generated_text"])
```

## ⚡ Quick Start

### Text Generation with LLMs

```python
import mindspore
import mindnlp
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen3-8B",
    ms_dtype=mindspore.bfloat16,
    device_map="auto"
)

messages = [{"role": "user", "content": "Write a haiku about coding"}]
print(pipe(messages, max_new_tokens=100)[0]["generated_text"][-1]["content"])
```

### Image Generation with Stable Diffusion

```python
import mindspore
import mindnlp
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ms_dtype=mindspore.float16
)
image = pipe("A sunset over mountains, oil painting style").images[0]
image.save("sunset.png")
```

## ✨ Features

=== "🤗 HuggingFace Compatibility"

    - **200,000+ models** from HuggingFace Hub
    - **Transformers** - All model architectures
    - **Diffusers** - Stable Diffusion, SDXL, ControlNet
    - **Zero code changes** - Just `import mindnlp`

=== "🚀 Hardware Acceleration"

    - **Ascend NPU** - Full support for Huawei AI chips
    - **NVIDIA GPU** - CUDA acceleration
    - **CPU** - Optimized CPU execution
    - **Multi-device** - Automatic device placement

=== "🔧 Advanced Capabilities"

    - **Mixed precision** - FP16/BF16 training & inference
    - **Quantization** - INT8/INT4 with BitsAndBytes
    - **Distributed** - Multi-GPU/NPU training
    - **PEFT/LoRA** - Parameter-efficient fine-tuning

=== "📦 Easy Integration"

    - **PyTorch-compatible API** via mindtorch
    - **Safetensors** support for fast loading
    - **Model Hub mirrors** for faster downloads
    - **Comprehensive documentation**

## 📦 Installation

```bash
# From PyPI (recommended)
pip install mindnlp

# From source (latest features)
pip install git+https://github.com/mindspore-lab/mindnlp.git
```

### Version Compatibility

| MindNLP | MindSpore | Python |
|---------|-----------|--------|
| 0.6.x   | ≥2.7.1    | 3.10-3.11 |
| 0.5.x   | 2.5.0-2.7.0 | 3.10-3.11 |
| 0.4.x   | 2.2.x-2.5.0 | 3.9-3.11 |
| 0.3.x   | 2.1.0-2.3.1 | 3.8-3.9 |

## 💡 Why MindNLP?

| Feature | MindNLP | PyTorch + HF | TensorFlow + HF |
|---------|---------|--------------|-----------------|
| HuggingFace Models | ✅ 200K+ | ✅ 200K+ | ⚠️ Limited |
| Ascend NPU Support | ✅ Native | ❌ | ❌ |
| Zero Code Migration | ✅ | - | ❌ |
| Chinese Model Support | ✅ Excellent | ✅ Good | ⚠️ Limited |

!!! success "Key Advantages"

    1. **Instant Migration**: Your existing HuggingFace code works immediately
    2. **Ascend Optimization**: Native support for Huawei NPU hardware
    3. **Production Ready**: Battle-tested in enterprise deployments
    4. **Active Community**: Regular updates and responsive support

## 🗺️ Supported Models

MindNLP supports **all models** from HuggingFace Transformers and Diffusers:

| Category | Models |
|----------|--------|
| **LLMs** | Qwen, Llama, ChatGLM, Mistral, Phi, Gemma, BLOOM, Falcon |
| **Vision** | ViT, CLIP, Swin, ConvNeXt, SAM, BLIP |
| **Audio** | Whisper, Wav2Vec2, HuBERT, MusicGen |
| **Diffusion** | Stable Diffusion, SDXL, ControlNet |
| **Multimodal** | LLaVA, Qwen-VL, ALIGN |

👉 [View all supported models](supported_models.md)

## 📚 Next Steps

!!! tip "Tutorials"

    - [Quick Start](tutorials/quick_start.md) - Fine-tune BERT for sentiment analysis
    - [Use Trainer](tutorials/use_trainer.md) - Configure training with Trainer API
    - [PEFT/LoRA](tutorials/peft.md) - Parameter-efficient fine-tuning
    - [Data Processing](tutorials/data_preprocess.md) - Dataset handling

!!! info "Resources"

    - [API Reference](api/patch/index.md) - API documentation
    - [FAQ](notes/faq.md) - Frequently asked questions
    - [Contributing](contribute.md) - How to contribute
    - [Changelog](notes/changelog.md) - Version history

## 🤝 Community

Join the **MindSpore NLP SIG** for discussions and collaboration:

<p align="center">
  <img src="assets/qrcode_qq_group.jpg" width="200" alt="QQ Group"/>
</p>

## 📄 License

MindNLP is released under the [Apache 2.0 License](https://github.com/mindspore-lab/mindnlp/blob/master/LICENSE).

## 📖 Citation

```bibtex
@misc{mindnlp2022,
    title={MindNLP: Easy-to-use and High-performance NLP and LLM Framework Based on MindSpore},
    author={MindNLP Contributors},
    howpublished={\url{https://github.com/mindspore-lab/mindnlp}},
    year={2022}
}
```

---

<p align="center">
  Made with ❤️ by the <a href="https://github.com/mindspore-lab">MindSpore Lab</a> team
</p>

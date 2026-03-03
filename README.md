<p align="center">
  <img src="https://raw.githubusercontent.com/mindspore-lab/mindnlp/master/assets/mindnlp_logo.png" width="400" alt="MindNLP Logo"/>
</p>

<h1 align="center">MindNLP</h1>

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

<p align="center">
  <a href="https://mindnlp.cqu.ai/en/latest/">
    <img alt="Documentation" src="https://img.shields.io/badge/docs-latest-blue?style=flat-square">
  </a>
  <a href="https://github.com/mindspore-lab/mindnlp/actions">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/mindspore-lab/mindnlp/ci_pipeline.yaml?style=flat-square&label=CI">
  </a>
  <a href="https://github.com/mindspore-lab/mindnlp/pulls">
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square">
  </a>
  <a href="https://github.com/mindspore-lab/mindnlp/issues">
    <img alt="Issues" src="https://img.shields.io/github/issues/mindspore-lab/mindnlp?style=flat-square">
  </a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-why-mindnlp">Why MindNLP</a> •
  <a href="https://mindnlp.cqu.ai">Documentation</a>
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

### BERT for Text Classification

```python
import mindnlp
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("MindNLP is awesome!", return_tensors="pt")
outputs = model(**inputs)
```

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤗 Full HuggingFace Compatibility

- **200,000+ models** from HuggingFace Hub
- **Transformers** - All model architectures
- **Diffusers** - Stable Diffusion, SDXL, ControlNet
- **Zero code changes** - Just `import mindnlp`

</td>
<td width="50%">

### 🚀 Hardware Acceleration

- **Ascend NPU** - Full support for Huawei AI chips
- **NVIDIA GPU** - CUDA acceleration
- **CPU** - Optimized CPU execution
- **Multi-device** - Automatic device placement

</td>
</tr>
<tr>
<td width="50%">

### 🔧 Advanced Capabilities

- **Mixed precision** - FP16/BF16 training & inference
- **Quantization** - INT8/INT4 with BitsAndBytes
- **Distributed** - Multi-GPU/NPU training
- **PEFT/LoRA** - Parameter-efficient fine-tuning

</td>
<td width="50%">

### 📦 Easy Integration

- **PyTorch-compatible API** via mindtorch
- **Safetensors** support for fast loading
- **Model Hub mirrors** for faster downloads
- **Comprehensive documentation**

</td>
</tr>
</table>

## 🧪 Mindtorch NPU Debugging

Mindtorch NPU ops are async by default. Use `torch.npu.synchronize()` when you need to block on results.
For debugging, set `ACL_LAUNCH_BLOCKING=1` to force per-op synchronization.

## 📦 Installation

```bash
# From PyPI (recommended)
pip install mindnlp

# From source (latest features)
pip install git+https://github.com/mindspore-lab/mindnlp.git
```

<details>
<summary><b>📋 Version Compatibility</b></summary>

| MindNLP | MindSpore | Python |
|---------|-----------|--------|
| 0.6.x   | ≥2.7.1    | 3.10-3.11 |
| 0.5.x   | 2.5.0-2.7.0 | 3.10-3.11 |
| 0.4.x   | 2.2.x-2.5.0 | 3.9-3.11 |

</details>

## 💡 Why MindNLP?

| Feature | MindNLP | PyTorch + HF | TensorFlow + HF |
|---------|---------|--------------|-----------------|
| HuggingFace Models | ✅ 200K+ | ✅ 200K+ | ⚠️ Limited |
| Ascend NPU Support | ✅ Native | ❌ | ❌ |
| Zero Code Migration | ✅ | - | ❌ |
| Unified API | ✅ | ✅ | ❌ |
| Chinese Model Support | ✅ Excellent | ✅ Good | ⚠️ Limited |

### 🏆 Key Advantages

1. **Instant Migration**: Your existing HuggingFace code works immediately
2. **Ascend Optimization**: Native support for Huawei NPU hardware
3. **Production Ready**: Battle-tested in enterprise deployments
4. **Active Community**: Regular updates and responsive support

## 🗺️ Supported Models

MindNLP supports **all models** from HuggingFace Transformers and Diffusers. Here are some popular ones:

| Category | Models |
|----------|--------|
| **LLMs** | Qwen, Llama, ChatGLM, Mistral, Phi, Gemma, BLOOM, Falcon |
| **Vision** | ViT, CLIP, Swin, ConvNeXt, SAM, BLIP |
| **Audio** | Whisper, Wav2Vec2, HuBERT, MusicGen |
| **Diffusion** | Stable Diffusion, SDXL, ControlNet |
| **Multimodal** | LLaVA, Qwen-VL, ALIGN |

👉 [View all supported models](https://mindnlp.cqu.ai/supported_models)

## 📚 Resources

- 📖 [Documentation](https://mindnlp.cqu.ai)
- 🚀 [Quick Start Guide](https://mindnlp.cqu.ai/quick_start)
- 📝 [Tutorials](https://mindnlp.cqu.ai/tutorials/quick_start)
- 💬 [GitHub Discussions](https://github.com/mindspore-lab/mindnlp/discussions)
- 🐛 [Issue Tracker](https://github.com/mindspore-lab/mindnlp/issues)

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](https://mindnlp.cqu.ai/contribute) for details.

```bash
# Clone and install for development
git clone https://github.com/mindspore-lab/mindnlp.git
cd mindnlp
pip install -e ".[dev]"
```

## 👥 Community

<p align="center">
  <a href="https://github.com/mindspore-lab/mindnlp/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mindspore-lab/mindnlp" />
  </a>
</p>

Join the **MindSpore NLP SIG** (Special Interest Group) for discussions, events, and collaboration:

<p align="center">
  <img src="./assets/qrcode_qq_group.jpg" width="200" alt="QQ Group"/>
</p>

## ⭐ Star History

<p align="center">
  <a href="https://star-history.com/#mindspore-lab/mindnlp&Date">
    <img src="https://api.star-history.com/svg?repos=mindspore-lab/mindnlp&type=Date" alt="Star History Chart" width="600">
  </a>
</p>

**If you find MindNLP useful, please consider giving it a star ⭐ - it helps the project grow!**

## 📄 License

MindNLP is released under the [Apache 2.0 License](LICENSE).

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

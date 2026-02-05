---
hide:
  - navigation
---

<p align="center">
  <span style="font-size: 3rem; font-weight: bold;">🚀 MindNLP</span>
</p>

<p align="center">
  <strong>零代码改动，在 MindSpore 上运行 HuggingFace 模型</strong>
</p>

<p align="center">
  <em>在昇腾 NPU、GPU 和 CPU 上使用 200,000+ HuggingFace 模型的最简单方式</em>
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

## 🎯 什么是 MindNLP？

**MindNLP** 连接了 HuggingFace 庞大的模型生态系统和 MindSpore 的硬件加速能力。只需 `import mindnlp`，您就可以在**昇腾 NPU**、**NVIDIA GPU** 或 **CPU** 上运行任何 HuggingFace 模型——无需修改代码。

```python
import mindnlp  # 就这么简单！HuggingFace 现在运行在 MindSpore 上
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B")
print(pipe("你好，我是")[0]["generated_text"])
```

## ⚡ 快速开始

### 使用大语言模型生成文本

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

messages = [{"role": "user", "content": "写一首关于编程的俳句"}]
print(pipe(messages, max_new_tokens=100)[0]["generated_text"][-1]["content"])
```

### 使用 Stable Diffusion 生成图像

```python
import mindspore
import mindnlp
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ms_dtype=mindspore.float16
)
image = pipe("山间日落，油画风格").images[0]
image.save("sunset.png")
```

## ✨ 特性

=== "🤗 完全兼容 HuggingFace"

    - **200,000+ 模型** 来自 HuggingFace Hub
    - **Transformers** - 所有模型架构
    - **Diffusers** - Stable Diffusion、SDXL、ControlNet
    - **零代码改动** - 只需 `import mindnlp`

=== "🚀 硬件加速"

    - **昇腾 NPU** - 完全支持华为 AI 芯片
    - **NVIDIA GPU** - CUDA 加速
    - **CPU** - 优化的 CPU 执行
    - **多设备** - 自动设备分配

=== "🔧 高级功能"

    - **混合精度** - FP16/BF16 训练和推理
    - **量化** - 使用 BitsAndBytes 的 INT8/INT4
    - **分布式** - 多 GPU/NPU 训练
    - **PEFT/LoRA** - 参数高效微调

=== "📦 易于集成"

    - **PyTorch 兼容 API**（通过 mindtorch）
    - **Safetensors** 支持快速加载
    - **模型镜像** 加速下载
    - **完善的文档**

## 📦 安装

```bash
# 从 PyPI 安装（推荐）
pip install mindnlp

# 从源码安装（最新功能）
pip install git+https://github.com/mindspore-lab/mindnlp.git
```

### 版本兼容性

| MindNLP | MindSpore | Python |
|---------|-----------|--------|
| 0.6.x   | ≥2.7.1    | 3.10-3.11 |
| 0.5.x   | 2.5.0-2.7.0 | 3.10-3.11 |
| 0.4.x   | 2.2.x-2.5.0 | 3.9-3.11 |
| 0.3.x   | 2.1.0-2.3.1 | 3.8-3.9 |

## 💡 为什么选择 MindNLP？

| 特性 | MindNLP | PyTorch + HF | TensorFlow + HF |
|------|---------|--------------|-----------------|
| HuggingFace 模型 | ✅ 200K+ | ✅ 200K+ | ⚠️ 有限 |
| 昇腾 NPU 支持 | ✅ 原生 | ❌ | ❌ |
| 零代码迁移 | ✅ | - | ❌ |
| 中文模型支持 | ✅ 优秀 | ✅ 良好 | ⚠️ 有限 |

!!! success "核心优势"

    1. **即时迁移**：您现有的 HuggingFace 代码立即可用
    2. **昇腾优化**：原生支持华为 NPU 硬件
    3. **生产就绪**：在企业部署中经过实战检验
    4. **活跃社区**：定期更新和响应迅速的支持

## 🗺️ 支持的模型

MindNLP 支持 HuggingFace Transformers 和 Diffusers 的**所有模型**：

| 类别 | 模型 |
|------|------|
| **大语言模型** | Qwen、Llama、ChatGLM、Mistral、Phi、Gemma、BLOOM、Falcon |
| **视觉** | ViT、CLIP、Swin、ConvNeXt、SAM、BLIP |
| **音频** | Whisper、Wav2Vec2、HuBERT、MusicGen |
| **扩散模型** | Stable Diffusion、SDXL、ControlNet |
| **多模态** | LLaVA、Qwen-VL、ALIGN |

👉 [查看所有支持的模型](supported_models.md)

## 📚 下一步

!!! tip "教程"

    - [快速入门](tutorials/quick_start.md) - 微调 BERT 进行情感分析
    - [使用 Trainer](tutorials/use_trainer.md) - 使用 Trainer API 配置训练
    - [PEFT/LoRA](tutorials/peft.md) - 参数高效微调
    - [数据处理](tutorials/data_preprocess.md) - 数据集处理

!!! info "资源"

    - [API 参考](api/patch/index.md) - API 文档
    - [常见问题](notes/faq.md) - 常见问题解答
    - [贡献指南](contribute.md) - 如何贡献
    - [更新日志](notes/changelog.md) - 版本历史

## 🤝 社区

加入 **MindSpore NLP SIG** 参与讨论和协作：

<p align="center">
  <img src="assets/qrcode_qq_group.jpg" width="200" alt="QQ 群"/>
</p>

## 📄 许可证

MindNLP 基于 [Apache 2.0 许可证](https://github.com/mindspore-lab/mindnlp/blob/master/LICENSE) 发布。

## 📖 引用

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
  由 <a href="https://github.com/mindspore-lab">MindSpore Lab</a> 团队用 ❤️ 打造
</p>

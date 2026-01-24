---
hide:
  - navigation
---

# 支持的模型

## HuggingFace 兼容性

MindNLP 通过其补丁机制提供**与所有 HuggingFace Transformers 和 Diffusers 模型的完全兼容性**。当您导入 `mindnlp` 时，它会自动将 HuggingFace 库补丁为使用 MindSpore 作为后端。

```python
import mindspore
import mindnlp  # 对 HuggingFace 库打补丁

# 现在任何 HuggingFace 模型都可以使用 MindSpore
from transformers import AutoModel
model = AutoModel.from_pretrained("any-model-on-huggingface")
```

这意味着您可以使用：

- [HuggingFace 模型中心](https://huggingface.co/models)上的**所有 200,000+ 模型**
- **最新的模型架构**，在 transformers 发布后立即可用
- 通过 diffusers 库使用**扩散模型**

## 使用模式

### 推理模式

对于推理，您可以在导入 `mindnlp` 后直接使用 HuggingFace API：

```python
import mindspore
import mindnlp
from transformers import pipeline

# 文本生成
pipe = pipeline("text-generation", model="Qwen/Qwen3-8B", ms_dtype=mindspore.bfloat16)
result = pipe("你好，最近怎么样？")

# 图像生成
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
image = pipe("美丽的日落").images[0]
```

### 训练模式

对于训练，请使用 MindNLP 的原生 API 以确保正确的梯度计算：

```python
from mindnlp.transformers import AutoModelForSequenceClassification
from mindnlp.engine import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

## 已验证的模型

以下模型已在 MindNLP 上进行了广泛测试。但是，**transformers 库中的所有模型都受支持**。

### 大型语言模型（LLMs）

| 模型系列 | 推理 | 训练 | 备注 |
|---------|:----:|:----:|------|
| Qwen/Qwen2/Qwen3 | ✅ | ✅ | 完整系列支持 |
| Llama/Llama2/Llama3 | ✅ | ✅ | 包括 CodeLlama |
| ChatGLM/GLM | ✅ | ✅ | ChatGLM、ChatGLM2、ChatGLM3 |
| Mistral/Mixtral | ✅ | ✅ | 支持 MoE 模型 |
| Phi-2/Phi-3 | ✅ | ✅ | |
| Gemma | ✅ | ✅ | |
| BLOOM | ✅ | ✅ | |
| Falcon | ✅ | ✅ | |
| GPT-2/GPT-Neo/GPT-NeoX | ✅ | ✅ | |
| OPT | ✅ | ✅ | |
| Mamba | ✅ | ✅ | |
| RWKV | ✅ | ✅ | |

### 编码器模型

| 模型 | 推理 | 训练 | 备注 |
|-----|:----:|:----:|------|
| BERT | ✅ | ✅ | 所有变体 |
| RoBERTa | ✅ | ✅ | |
| ALBERT | ✅ | ✅ | |
| DeBERTa | ✅ | ✅ | |
| ELECTRA | ✅ | ✅ | |
| XLNet | ✅ | ✅ | |
| Longformer | ✅ | ✅ | |
| BigBird | ✅ | ✅ | |

### 编码器-解码器模型

| 模型 | 推理 | 训练 | 备注 |
|-----|:----:|:----:|------|
| T5 | ✅ | ✅ | 包括 mT5、Flan-T5 |
| BART | ✅ | ✅ | |
| mBART | ✅ | ✅ | |
| Pegasus | ✅ | ✅ | |

### 视觉模型

| 模型 | 推理 | 训练 | 备注 |
|-----|:----:|:----:|------|
| ViT | ✅ | ✅ | |
| CLIP | ✅ | ✅ | |
| BLIP/BLIP-2 | ✅ | ✅ | |
| SAM | ✅ | ⚠️ | 建议仅用于推理 |
| Swin Transformer | ✅ | ✅ | |
| ConvNeXt | ✅ | ✅ | |
| ResNet | ✅ | ✅ | |

### 多模态模型

| 模型 | 推理 | 训练 | 备注 |
|-----|:----:|:----:|------|
| LLaVA | ✅ | ⚠️ | |
| Qwen-VL | ✅ | ✅ | |
| ALIGN | ✅ | ⚠️ | |
| AltCLIP | ✅ | ⚠️ | |
| BridgeTower | ✅ | ⚠️ | |

### 音频模型

| 模型 | 推理 | 训练 | 备注 |
|-----|:----:|:----:|------|
| Whisper | ✅ | ✅ | |
| Wav2Vec2 | ✅ | ✅ | |
| HuBERT | ✅ | ✅ | |
| MusicGen | ✅ | ⚠️ | |
| Bark | ✅ | ⚠️ | |

### 扩散模型

| 模型 | 推理 | 训练 | 备注 |
|-----|:----:|:----:|------|
| Stable Diffusion | ✅ | ⚠️ | v1.5、v2.x、XL |
| SDXL | ✅ | ⚠️ | |
| ControlNet | ✅ | ⚠️ | |
| LoRA 适配器 | ✅ | ✅ | |

**图例：**
- ✅ 完全支持
- ⚠️ 实验性或仅建议用于推理

## 设备支持

| 设备 | 状态 | 备注 |
|-----|:----:|------|
| CPU | ✅ | 所有平台 |
| NVIDIA GPU | ✅ | Linux 上支持 CUDA 11.x |
| 昇腾 NPU | ✅ | Linux 上完全支持 |

## 已知限制

1. **图模式**：某些动态模型可能需要 PyNative 模式
2. **自定义 CUDA 算子**：具有自定义 CUDA 操作的模型可能需要适配
3. **量化**：某些量化方法可能支持有限

## 问题反馈

如果您在特定模型上遇到问题：

1. 查看 [GitHub Issues](https://github.com/mindspore-lab/mindnlp/issues) 了解已知问题
2. 尝试更新到最新版本：`pip install -U mindnlp`
3. 报告新问题时请提供模型名称、错误消息和复现步骤

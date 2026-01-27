# 快速开始

欢迎使用 MindNLP！本页面将帮助您快速入门。

## 入门指南

如需了解如何使用 MindNLP 加载预训练模型并针对特定任务进行微调的详细教程，请访问：

**[快速入门教程](tutorials/quick_start.md)** - 学习如何微调 BERT 进行情感分类

## 快速示例

### 在 MindSpore 中使用 HuggingFace Transformers

```python
import mindspore
import mindnlp
from transformers import pipeline

# 使用 Qwen 创建文本生成 pipeline
pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen3-8B",
    ms_dtype=mindspore.bfloat16,
    device_map="auto"
)

chat = [
    {"role": "user", "content": "你好，最近怎么样？"}
]
response = pipe(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

### 使用 MindNLP 原生接口

```python
from mindnlp.transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="ms")
outputs = model(**inputs)
```

## 更多教程

- [使用 Trainer](tutorials/use_trainer.md) - 使用 MindNLP 的 Trainer API 进行训练
- [PEFT/LoRA](tutorials/peft.md) - 参数高效微调
- [数据预处理](tutorials/data_preprocess.md) - 数据集处理
- [使用镜像](tutorials/use_mirror.md) - 使用模型镜像加速下载

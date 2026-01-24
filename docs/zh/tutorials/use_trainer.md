# 使用 Trainer

在[快速入门](./quick_start.md)教程中，我们学习了如何使用 Trainer API 来微调模型。本教程将全面介绍如何配置 `Trainer` 以获得最佳训练效果。

## 概述

MindNLP 对 HuggingFace `transformers` 库进行补丁以支持 MindSpore。这意味着您可以直接使用标准的 HuggingFace `Trainer` 和 `TrainingArguments` 类：

```python
import mindspore
import mindnlp  # 应用补丁

from transformers import Trainer, TrainingArguments
```

`TrainingArguments` 类允许您配置基本的训练参数，`Trainer` 使用 MindSpore 作为后端处理整个训练循环。

## 配置训练参数

创建 `TrainingArguments` 对象来指定训练配置：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50
)
```

### 基本参数

- **output_dir**：保存模型检查点和训练输出的目录。
- **num_train_epochs**：训练轮数。更多轮数可以更好地学习，但可能导致过拟合。

### 优化器参数

- **optim**：优化器类型。默认是 "adamw_torch"（映射到 MindSpore 的 AdamW）。
- **learning_rate**：初始学习率。这对收敛至关重要——太高会导致不稳定，太低会导致收敛缓慢。
- **weight_decay**：正则化，通过惩罚大权重来防止过拟合。
- **adam_beta1** / **adam_beta2**：Adam 优化器的动量参数。
- **adam_epsilon**：Adam 中数值稳定性的小值。
- **max_grad_norm**：梯度裁剪阈值，防止梯度爆炸。

### 批次大小参数

- **per_device_train_batch_size**：训练批次大小。较大的批次更快但需要更多内存。
- **per_device_eval_batch_size**：评估批次大小。
- **gradient_accumulation_steps**：在多个步骤中累积梯度，以在有限内存下模拟更大的批次大小。

### 策略参数

#### 评估策略

`eval_strategy` 参数控制何时进行评估：

- `"no"`：不评估
- `"steps"`：每 `eval_steps` 个训练步骤评估一次
- `"epoch"`：每个 epoch 结束时评估

#### 保存策略

`save_strategy` 参数控制何时保存检查点：

- `"no"`：不保存
- `"steps"`：每 `save_steps` 个训练步骤保存一次
- `"epoch"`：每个 epoch 结束时保存

#### 日志策略

`logging_strategy` 参数控制何时记录指标：

- `"no"`：不记录
- `"steps"`：每 `logging_steps` 个训练步骤记录一次
- `"epoch"`：每个 epoch 结束时记录

### MindSpore 特定参数

MindNLP 在模型加载中添加了对 MindSpore 特定参数的支持：

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    ms_dtype=mindspore.float16  # 使用 MindSpore 数据类型
)
```

## 创建 Trainer

使用模型、数据集和配置创建 `Trainer` 实例：

```python
import mindnlp
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 创建 trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

### Trainer 参数

- **model**：要训练的模型
- **args**：带有训练配置的 `TrainingArguments` 实例
- **train_dataset**：训练数据集
- **eval_dataset**：评估数据集（可选）
- **compute_metrics**：计算评估指标的函数（可选）

### 定义 compute_metrics

`compute_metrics` 函数从模型预测计算指标：

```python
import evaluate
import numpy as np

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

## 开始训练

配置好 trainer 后，开始训练：

```python
trainer.train()
```

## 完整示例

```python
import mindspore
import mindnlp
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import evaluate
import numpy as np

# 加载数据集
dataset = load_dataset("imdb")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 对数据集进行分词
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 定义指标
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
)

# 创建 trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # 演示用子集
    eval_dataset=tokenized_datasets["test"].select(range(200)),
    compute_metrics=compute_metrics,
)

# 训练
trainer.train()
```

## 高级功能

### 混合精度训练

使用较低精度以加快训练并减少内存：

```python
training_args = TrainingArguments(
    output_dir="./output",
    fp16=True,  # 启用 FP16 训练
    # 或使用 bf16=True 启用 bfloat16
)
```

### 梯度检查点

通过在反向传播时重新计算激活来节省内存：

```python
training_args = TrainingArguments(
    output_dir="./output",
    gradient_checkpointing=True,
)
```

### 从检查点恢复

从保存的检查点恢复训练：

```python
trainer.train(resume_from_checkpoint="./results/checkpoint-500")
```

## 注意事项

- Trainer 通过 MindNLP 的补丁系统自动使用 MindSpore 操作
- 所有标准的 HuggingFace Trainer 功能都应该可以使用
- 对于生产训练，建议使用完整数据集而不是子集
- 通过在 TrainingArguments 中设置 `logging_dir` 可以使用 TensorBoard 监控训练

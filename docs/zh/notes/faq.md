# 常见问题

## 安装

### 支持哪些版本的 MindSpore？

| MindNLP 版本 | MindSpore 版本 | Python 版本 |
|-------------|----------------|-------------|
| 0.6.x       | >=2.7.1        | 3.10-3.11   |
| 0.5.x       | 2.5.0-2.7.0    | 3.10-3.11   |
| 0.4.x       | 2.2.x-2.5.0    | 3.9-3.11    |
| 0.3.x       | 2.1.0-2.3.1    | 3.8-3.9     |

### 支持哪些平台？

MindNLP 支持：

- **Linux**：Ubuntu 18.04/20.04/22.04（推荐）
- **macOS**：Intel 和 Apple Silicon（仅 CPU）
- **Windows**：Windows 10/11（有限支持）

硬件加速器：

- **昇腾 NPU**：Linux 上完全支持
- **NVIDIA GPU**：Linux 上支持 CUDA 11.x
- **CPU**：所有平台

### 如何安装 MindNLP？

```bash
# 从 PyPI 安装（推荐）
pip install mindnlp

# 从源码安装
pip install git+https://github.com/mindspore-lab/mindnlp.git

# 每日构建版本
# 下载地址：https://repo.mindspore.cn/mindspore-lab/mindnlp/newest/any/
```

### 安装失败，提示 MindSpore 错误

请确保先正确安装 MindSpore：

```bash
# 检查 MindSpore 安装
python -c "import mindspore; print(mindspore.__version__)"

# 如需安装 MindSpore（CPU 示例）
pip install mindspore
```

请参阅 [MindSpore 安装指南](https://www.mindspore.cn/install) 获取特定平台的安装说明。

## 模型兼容性

### MindNLP 支持所有 HuggingFace 模型吗？

是的！MindNLP 通过其补丁机制提供与 HuggingFace 生态系统的完全兼容性。当您导入 `mindnlp` 时，它会自动将 `transformers` 和 `diffusers` 补丁为使用 MindSpore 作为后端。

```python
import mindspore
import mindnlp  # 这会对 HuggingFace 库打补丁

# 现在可以使用 MindSpore 后端加载模型
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
```

### 推理和训练支持有什么区别？

**推理**：所有 HuggingFace 模型开箱即用支持推理。

**训练**：进行训练时，请使用 MindNLP 的原生 API（`mindnlp.transformers`）以确保正确的梯度计算：

```python
# 训练时使用 MindNLP API
from mindnlp.transformers import AutoModelForSequenceClassification
from mindnlp.engine import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# ... 训练代码
```

### 如何使用特定的模型精度？

使用 `ms_dtype` 参数：

```python
import mindspore
from transformers import AutoModel

# 以 float16 加载
model = AutoModel.from_pretrained(
    "model-name",
    ms_dtype=mindspore.float16
)

# 以 bfloat16 加载
model = AutoModel.from_pretrained(
    "model-name",
    ms_dtype=mindspore.bfloat16
)
```

### 模型加载很慢或失败

1. **检查网络连接**：模型权重从 HuggingFace Hub 下载
2. **使用镜像**：参阅[使用镜像教程](../tutorials/use_mirror.md)在国内加速下载
3. **检查磁盘空间**：大型模型需要大量存储空间
4. **先尝试小模型**：使用小模型如 `bert-base-uncased` 验证您的设置是否正常

## 训练问题

### 如何微调模型？

使用 MindNLP Trainer API：

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

完整示例请参阅[快速入门教程](../tutorials/quick_start.md)。

### 训练时内存不足

1. **减小批次大小**：降低 `per_device_train_batch_size`
2. **使用梯度累积**：设置 `gradient_accumulation_steps`
3. **使用混合精度**：设置 `ms_dtype=mindspore.float16`
4. **使用 PEFT/LoRA**：参阅 [PEFT 教程](../tutorials/peft.md)了解参数高效微调

### 梯度计算不正确

确保您使用 MindNLP 的原生 API 进行训练：

```python
# 正确：训练时使用 mindnlp.transformers
from mindnlp.transformers import AutoModel

# 错误：直接使用 transformers 可能存在梯度问题
from transformers import AutoModel  # 仅用于推理
```

## 常见错误

### `RuntimeError: Device mismatch`

确保张量在同一设备上：

```python
import mindspore
mindspore.set_context(device_target="CPU")  # 或 "GPU"、"Ascend"
```

### `KeyError: 'model_type'`

模型配置可能不兼容。请尝试：

```python
from mindnlp.transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("model-name")
model = AutoModel.from_config(config)
```

### `ImportError: cannot import name 'xxx'`

1. 更新 MindNLP 到最新版本：`pip install -U mindnlp`
2. 确保 MindSpore 版本兼容（参见上方版本表）
3. 检查该功能在您的版本中是否可用

### 模型输出与 PyTorch 不同

由于后端不同，微小的数值差异是正常的。对于显著差异：

1. 确保使用相同的模型权重
2. 检查输入预处理是否相同
3. 验证模型配置是否匹配

## 性能

### 如何加速推理？

1. **使用 GPU/昇腾**：设置相应的设备上下文
2. **使用低精度**：`ms_dtype=mindspore.float16`
3. **启用图模式**：`mindspore.set_context(mode=mindspore.GRAPH_MODE)`
4. **批量处理输入**：同时处理多个样本

### 如何启用分布式训练？

MindNLP 支持 MindSpore 的分布式训练功能：

```python
from mindspore.communication import init
from mindspore import set_auto_parallel_context

init()
set_auto_parallel_context(parallel_mode="data_parallel")
```

## 获取帮助

- **GitHub Issues**：[mindspore-lab/mindnlp/issues](https://github.com/mindspore-lab/mindnlp/issues)
- **文档**：[mindnlp.cqu.ai](https://mindnlp.cqu.ai)
- **MindSpore 论坛**：[bbs.huaweicloud.com/forum/forum-1076-1.html](https://bbs.huaweicloud.com/forum/forum-1076-1.html)

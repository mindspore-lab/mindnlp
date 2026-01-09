# OCR Fine-tuning Example

本目录包含 Qwen2-VL OCR 模型微调的示例代码和数据。

## 文件说明

- `finetune_example.py`: 完整的微调示例代码,包含 LoRA、QLoRA、评估和推理
- `sample_data.json`: 示例训练数据 (3个样本)
- `README.md`: 本文件

## 快速开始

### 1. 准备数据

将你的图片放在 `images/` 目录下,并准备训练数据 JSON 文件:

```json
[
  {
    "image_path": "images/your_image.jpg",
    "conversations": [
      {
        "role": "user",
        "content": "请识别这张图片中的文字"
      },
      {
        "role": "assistant",
        "content": "识别结果文本"
      }
    ],
    "task_type": "general"
  }
]
```

### 2. LoRA 微调

```bash
python -m mindnlp.ocr.finetune.train_lora \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --data_path ./sample_data.json \
  --image_folder ./images \
  --output_dir ./output/lora_model \
  --lora_r 16 \
  --lora_alpha 32 \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

### 3. QLoRA 微调 (低显存)

```bash
python -m mindnlp.ocr.finetune.train_qlora \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --data_path ./sample_data.json \
  --image_folder ./images \
  --output_dir ./output/qlora_model \
  --load_in_4bit \
  --lora_r 16 \
  --num_epochs 3 \
  --batch_size 4
```

### 4. 模型评估

```bash
python -m mindnlp.ocr.finetune.evaluate \
  --model_path ./output/lora_model/final_model \
  --test_data_path ./test_data.json \
  --image_folder ./images \
  --output_file ./evaluation_results.json
```

### 5. Python API 使用

```python
from mindnlp.ocr.finetune import train_lora, evaluate_model

# 训练
model, processor = train_lora(
    model_name_or_path="Qwen/Qwen2-VL-7B-Instruct",
    data_path="./sample_data.json",
    image_folder="./images",
    output_dir="./output/lora_model",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
)

# 评估
metrics = evaluate_model(
    model=model,
    processor=processor,
    test_data_path="./test_data.json",
    image_folder="./images",
    device="cuda"
)

print(f"CER: {metrics['cer']:.4f}")
print(f"WER: {metrics['wer']:.4f}")
```

## 完整文档

详细的微调文档请查看: [docs/ocr_finetuning.md](../../docs/ocr_finetuning.md)

## Issue #2379 要求

根据 [Issue #2379](https://github.com/mindspore-ai/mindnlp/issues/2379) 的要求:

- ✅ 支持 LoRA 微调 (rank 8-64, alpha 16-128)
- ✅ 支持 QLoRA 微调 (4-bit 量化 + LoRA)
- ✅ 支持 OCR 数据集格式 (JSON with image_path and conversations)
- ✅ 实现 CER/WER 评估指标
- ✅ 任务特定准确率计算 (表格、公式识别)
- ✅ 完整的文档和示例代码

### 验收标准

微调后的模型应达到:
- CER 降低 20% 以上 (相比基础模型)
- 表格识别准确率 ≥ 95%
- 公式识别准确率 ≥ 90%

## 依赖

```bash
pip install mindnlp>=0.3.0
pip install transformers>=4.37.0
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0  # QLoRA
pip install datasets
pip install accelerate
pip install editdistance  # 评估
```

## 目录结构

```
examples/ocr_finetune_example/
├── README.md                 # 本文件
├── finetune_example.py      # 示例代码
├── sample_data.json         # 示例数据
└── images/                  # 图片目录 (需自行创建)
    ├── receipt_001.jpg
    ├── table_001.jpg
    └── formula_001.jpg
```

## 相关资源

- [Qwen2-VL 官方文档](https://github.com/QwenLM/Qwen2-VL)
- [PEFT 库文档](https://github.com/huggingface/peft)
- [MindNLP 文档](https://mindnlp.cqu.edu.cn/)
- [OCR 微调完整文档](../../docs/ocr_finetuning.md)

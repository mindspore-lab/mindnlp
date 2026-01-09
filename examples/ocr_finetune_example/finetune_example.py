"""
OCR Fine-tuning Example
演示如何使用 MindNLP 微调 Qwen2-VL OCR 模型
"""

# ============================================================
# 示例 1: LoRA 微调
# ============================================================

from mindnlp.ocr.finetune import train_lora

# 训练参数
model_name = "Qwen/Qwen2-VL-7B-Instruct"
data_path = "./sample_data.json"  # 训练数据
image_folder = "./images"          # 图片文件夹
output_dir = "./output/lora_model" # 输出目录

# 执行 LoRA 微调
model, processor = train_lora(
    model_name_or_path=model_name,
    data_path=data_path,
    image_folder=image_folder,
    output_dir=output_dir,
    # LoRA 配置
    lora_r=16,              # LoRA 秩
    lora_alpha=32,          # LoRA alpha
    lora_dropout=0.1,       # Dropout
    # 训练配置
    num_epochs=3,           # 训练轮数
    batch_size=4,           # 批大小
    gradient_accumulation_steps=4,  # 梯度累积
    learning_rate=2e-4,     # 学习率
    max_length=2048,        # 最大序列长度
    # 保存和日志
    save_steps=100,
    logging_steps=10,
    eval_steps=100,
    # 其他
    fp16=True,              # 使用 FP16
    gradient_checkpointing=True,  # 梯度检查点
)

print("LoRA 微调完成!")


# ============================================================
# 示例 2: QLoRA 微调 (低显存)
# ============================================================

from mindnlp.ocr.finetune import train_qlora

# 执行 QLoRA 微调
model, processor = train_qlora(
    model_name_or_path=model_name,
    data_path=data_path,
    image_folder=image_folder,
    output_dir="./output/qlora_model",
    # 量化配置
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # LoRA 配置
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    # 训练配置
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
)

print("QLoRA 微调完成!")


# ============================================================
# 示例 3: 模型评估
# ============================================================

from mindnlp.ocr.finetune import evaluate_model
from mindnlp.ocr.finetune.train_lora import load_lora_model

# 加载微调后的模型
model, processor = load_lora_model(
    base_model_path=model_name,
    lora_path="./output/lora_model/final_model",
    device="cuda"
)

# 执行评估
metrics = evaluate_model(
    model=model,
    processor=processor,
    test_data_path="./test_data.json",
    image_folder="./images",
    output_file="./output/evaluation_results.json",
    device="cuda"
)

# 打印结果
print("=" * 60)
print("评估结果:")
print(f"  CER (字符错误率): {metrics['cer']:.4f}")
print(f"  WER (词错误率): {metrics['wer']:.4f}")
print(f"  完全匹配率: {metrics['exact_match']:.4f}")
print("=" * 60)

# 检查是否满足 Issue #2379 要求
baseline_cer = 0.0850  # 假设基础模型的 CER
cer_reduction = (baseline_cer - metrics['cer']) / baseline_cer * 100

if cer_reduction >= 20:
    print(f"✓ CER 降低 {cer_reduction:.2f}% (要求 ≥20%)")
else:
    print(f"✗ CER 降低 {cer_reduction:.2f}% (要求 ≥20%)")


# ============================================================
# 示例 4: 推理使用
# ============================================================

import torch
from PIL import Image
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 加载模型
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    "./output/lora_model/final_model"
)
model.eval()

# 推理
image = Image.open("test_image.jpg").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "请识别这张图片中的文字"}
    ]
}]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    padding=True,
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
    )

response = processor.batch_decode(
    outputs,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("识别结果:")
print(response)


# ============================================================
# 示例 5: 计算 CER/WER
# ============================================================

from mindnlp.ocr.finetune import calculate_cer, calculate_wer, calculate_accuracy

# 单个样本
reference = "这是一段测试文本"
hypothesis = "这是一段测试文字"

cer = calculate_cer(reference, hypothesis)
wer = calculate_wer(reference, hypothesis)

print(f"CER: {cer:.4f}")
print(f"WER: {wer:.4f}")

# 多个样本
references = [
    "第一段文本",
    "第二段文本",
    "第三段文本",
]
hypotheses = [
    "第一段文字",
    "第二段文本",
    "第三个文本",
]

metrics = calculate_accuracy(references, hypotheses)
print(f"平均 CER: {metrics['cer']:.4f}")
print(f"平均 WER: {metrics['wer']:.4f}")
print(f"完全匹配率: {metrics['exact_match']:.4f}")


# ============================================================
# 示例 6: 命令行使用
# ============================================================

"""
# LoRA 微调
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

# QLoRA 微调
python -m mindnlp.ocr.finetune.train_qlora \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --data_path ./sample_data.json \
  --image_folder ./images \
  --output_dir ./output/qlora_model \
  --load_in_4bit \
  --lora_r 16 \
  --num_epochs 3 \
  --batch_size 4

# 模型评估
python -m mindnlp.ocr.finetune.evaluate \
  --model_path ./output/lora_model/final_model \
  --test_data_path ./test_data.json \
  --image_folder ./images \
  --output_file ./evaluation_results.json
"""

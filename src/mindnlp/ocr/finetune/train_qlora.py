"""
Qwen2-VL QLoRA Fine-tuning Script
使用 4-bit 量化 + LoRA 实现低显存微调
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # pylint: disable=import-error

from .dataset import OCRDataset, OCRDataCollator, split_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    创建量化配置

    Args:
        load_in_4bit: 是否使用 4-bit 量化
        bnb_4bit_compute_dtype: 计算数据类型
        bnb_4bit_quant_type: 量化类型 (nf4 or fp4)
        bnb_4bit_use_double_quant: 是否使用双重量化

    Returns:
        BitsAndBytesConfig 实例
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    logger.info(f"Quantization config created: 4bit={load_in_4bit}, dtype={bnb_4bit_compute_dtype}, quant_type={bnb_4bit_quant_type}")

    return config


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """
    创建 LoRA 配置

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: Dropout 比例
        target_modules: 目标模块列表

    Returns:
        LoraConfig 实例
    """
    if target_modules is None:
        # Qwen2-VL 默认目标模块
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    logger.info(f"LoRA config created: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")

    return config


def train_qlora(
    model_name_or_path: str,
    data_path: str,
    output_dir: str,
    image_folder: Optional[str] = None,
    # 量化参数
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    # LoRA 参数
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    # 训练参数
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    # 其他参数
    save_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 100,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    gradient_checkpointing: bool = True,
    # 数据集分割
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
):
    """
    执行 QLoRA 微调

    Args:
        model_name_or_path: 预训练模型路径或名称
        data_path: 训练数据 JSON 文件路径
        output_dir: 输出目录
        image_folder: 图片文件夹路径
        load_in_4bit: 是否使用 4-bit 量化
        bnb_4bit_compute_dtype: 计算数据类型
        bnb_4bit_quant_type: 量化类型
        bnb_4bit_use_double_quant: 是否使用双重量化
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        num_epochs: 训练轮数
        batch_size: 批大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        max_length: 最大序列长度
        save_steps: 保存步数
        logging_steps: 日志步数
        eval_steps: 评估步数
        warmup_ratio: Warmup 比例
        weight_decay: 权重衰减
        gradient_checkpointing: 是否使用梯度检查点
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(random_seed)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting QLoRA Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Quantization: 4bit={load_in_4bit}, quant_type={bnb_4bit_quant_type}")
    logger.info(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Training: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

    # 1. 加载 processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    # 2. 加载数据集
    logger.info("Loading dataset...")
    full_dataset = OCRDataset(
        data_path=data_path,
        processor=processor,
        max_length=max_length,
        image_folder=image_folder,
    )

    # 分割数据集
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    # 3. 创建量化配置
    logger.info("Creating quantization config...")
    quantization_config = create_quantization_config(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    # 4. 加载量化模型
    logger.info("Loading quantized model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 准备模型用于 k-bit 训练
    model = prepare_model_for_kbit_training(model)

    # 启用梯度检查点
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # 5. 创建 LoRA 模型
    logger.info("Creating LoRA model...")
    lora_config = create_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. 创建数据整理器
    data_collator = OCRDataCollator(
        processor=processor,
        padding=True,
        max_length=max_length,
    )

    # 7. 配置训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=3,
        fp16=False,  # QLoRA 使用 4-bit,不需要额外的 fp16
        dataloader_num_workers=4,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=random_seed,
    )

    # 8. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 9. 开始训练
    logger.info("Starting training...")
    trainer.train()

    # 10. 保存最终模型
    final_output_dir = output_dir / "final_model"
    trainer.save_model(str(final_output_dir))
    processor.save_pretrained(str(final_output_dir))

    logger.info(f"Model saved to {final_output_dir}")
    logger.info("Training completed!")

    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL QLoRA Fine-tuning")

    # 基础参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型路径或名称")
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="图片文件夹路径")

    # 量化参数
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="使用 4-bit 量化")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],
                        help="计算数据类型")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                        choices=["nf4", "fp4"],
                        help="量化类型")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True,
                        help="使用双重量化")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志步数")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="评估步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="不使用梯度检查点")

    # 数据集分割
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 执行训练
    train_qlora(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        image_folder=args.image_folder,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()

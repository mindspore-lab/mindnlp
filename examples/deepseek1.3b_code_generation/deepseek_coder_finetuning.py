#!/usr/bin/env python
# coding=utf-8
"""
DeepSeek Coder 模型在特定代码数据集上的微调示例
"""

import logging
import os
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union

import mindspore
from mindspore import nn
from mindspore.dataset import GeneratorDataset

from mindnlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from mindnlp.transformers.data.data_collator import DataCollatorForLanguageModeling
from mindnlp.transformers.optimization import get_scheduler, AdamWeightDecay

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    模型参数
    """
    model_name_or_path: str = field(
        default="deepseek-ai/deepseek-coder-1.3b-base",
        metadata={"help": "预训练模型的路径或标识符"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用LoRA进行参数高效微调"}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA的秩"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA的alpha参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA的dropout率"}
    )

@dataclass
class DataTrainingArguments:
    """
    数据训练参数
    """
    train_file: Optional[str] = field(
        default=None, metadata={"help": "训练数据文件的路径"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "验证数据文件的路径"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "单个样本的最大总序列长度。序列将被截断为该长度。"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于数据预处理的进程数"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖缓存的预处理数据集"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "用于划分训练样本的可选输入序列长度"
        },
    )

class CodeDataset:
    """代码数据集类"""
    
    def __init__(self, file_path, tokenizer, block_size):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 读取并处理数据
        logger.info(f"正在读取数据文件: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        
        # 分割为代码样本
        code_samples = text.split("# ---NEW SAMPLE---")
        
        for code in code_samples:
            if len(code.strip()) > 0:
                tokenized_code = self.tokenizer.encode(code.strip())
                self.examples.extend(self._get_chunks(tokenized_code))
    
    def _get_chunks(self, tokenized_code):
        chunks = []
        for i in range(0, len(tokenized_code), self.block_size):
            chunk = tokenized_code[i:i + self.block_size]
            if len(chunk) == self.block_size:
                chunks.append({"input_ids": chunk})
        return chunks
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    parser = argparse.ArgumentParser(description="微调 DeepSeek Coder 模型")
    
    # 添加模型参数
    model_args_group = parser.add_argument_group("模型参数")
    model_args_group.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    model_args_group.add_argument("--use_lora", action="store_true")
    model_args_group.add_argument("--lora_rank", type=int, default=8)
    model_args_group.add_argument("--lora_alpha", type=int, default=16)
    model_args_group.add_argument("--lora_dropout", type=float, default=0.05)
    
    # 添加数据参数
    data_args_group = parser.add_argument_group("数据参数")
    data_args_group.add_argument("--train_file", type=str, required=True)
    data_args_group.add_argument("--validation_file", type=str)
    data_args_group.add_argument("--max_seq_length", type=int, default=512)
    data_args_group.add_argument("--block_size", type=int, default=None)
    data_args_group.add_argument("--overwrite_cache", action="store_true")
    data_args_group.add_argument("--preprocessing_num_workers", type=int, default=None)
    
    # 添加训练参数
    training_args_group = parser.add_argument_group("训练参数")
    training_args_group.add_argument("--output_dir", type=str, required=True)
    training_args_group.add_argument("--num_train_epochs", type=int, default=3)
    training_args_group.add_argument("--per_device_train_batch_size", type=int, default=8)
    training_args_group.add_argument("--per_device_eval_batch_size", type=int, default=8)
    training_args_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    training_args_group.add_argument("--learning_rate", type=float, default=5e-5)
    training_args_group.add_argument("--weight_decay", type=float, default=0.01)
    training_args_group.add_argument("--warmup_ratio", type=float, default=0.1)
    training_args_group.add_argument("--logging_steps", type=int, default=10)
    training_args_group.add_argument("--save_steps", type=int, default=500)
    training_args_group.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载模型和分词器
    logger.info(f"加载模型和分词器: {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    # 如果使用LoRA进行参数高效微调
    if args.use_lora:
        # 注意：这里需要实现LoRA的集成，这是一个简化版
        logger.info(f"使用LoRA进行参数高效微调，rank={args.lora_rank}, alpha={args.lora_alpha}")
        # 这里应添加LoRA相关配置和实现
        
    # 确定block_size
    block_size = args.block_size
    if block_size is None:
        block_size = min(tokenizer.model_max_length, args.max_seq_length)
    
    # 准备数据集
    train_dataset = CodeDataset(args.train_file, tokenizer, block_size)
    eval_dataset = None
    if args.validation_file:
        eval_dataset = CodeDataset(args.validation_file, tokenizer, block_size)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 优化器
    optimizer = AdamWeightDecay(
        params=model.trainable_params(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # (optimizer, scheduler)
    )
    
    # 开始训练
    logger.info("开始微调")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存微调后的模型到 {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main() 
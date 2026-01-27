# -*- coding: utf-8 -*-
"""
OCR Fine-tuning Module
支持 LoRA、QLoRA 和全参数微调
"""
from .dataset import OCRDataset, OCRDataCollator
from .train_lora import train_lora, load_lora_model
from .train_qlora import train_qlora
from .train_full import train_full
from .evaluate import (
    evaluate_model,
    calculate_cer,
    calculate_wer,
    calculate_accuracy,
    evaluate_table_accuracy,
    evaluate_formula_accuracy,
    evaluate_handwriting_accuracy,
)
from .prepare_dataset import (
    convert_icdar2015,
    convert_funsd,
    convert_custom_format,
    merge_and_split_datasets,
)

__all__ = [
    # 数据
    'OCRDataset',
    'OCRDataCollator',
    # 训练函数
    'train_lora',
    'train_qlora',
    'train_full',
    'load_lora_model',
    # 评估函数
    'evaluate_model',
    'calculate_cer',
    'calculate_wer',
    'calculate_accuracy',
    'evaluate_table_accuracy',
    'evaluate_formula_accuracy',
    'evaluate_handwriting_accuracy',
    # 数据准备
    'convert_icdar2015',
    'convert_funsd',
    'convert_custom_format',
    'merge_and_split_datasets',
]

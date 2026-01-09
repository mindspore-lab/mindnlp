"""
OCR Fine-tuning Module
支持 LoRA 和 QLoRA 微调
"""
from .dataset import OCRDataset, OCRDataCollator
from .train_lora import train_lora
from .train_qlora import train_qlora
from .evaluate import evaluate_model, calculate_cer, calculate_wer

__all__ = [
    'OCRDataset',
    'OCRDataCollator',
    'train_lora',
    'train_qlora',
    'evaluate_model',
    'calculate_cer',
    'calculate_wer',
]

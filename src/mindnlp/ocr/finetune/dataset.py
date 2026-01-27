"""
OCR Dataset Module for Fine-tuning
支持 OCR 数据集的加载、预处理和批处理
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    """
    OCR 数据集类

    支持的数据格
    [
        {
            "image_path": "path/to/image.jpg",
            "conversations": [
                {"role": "user", "content": "识别这张图片中的文字"},
                {"role": "assistant", "content": "这是识别结果"}
            ]
        }
    ]
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        processor: AutoProcessor,
        max_length: int = 2048,
        image_folder: Optional[Union[str, Path]] = None,
    ):
        """
        初始OCR 数据

        Args:
            data_path: JSON 数据文件路径
            processor: Qwen2VL processor
            max_length: 最大序列长
            image_folder: 图片文件夹路径（如果 image_path 是相对路径）
        """
        self.data_path = Path(data_path)
        self.processor = processor
        self.max_length = max_length
        self.image_folder = Path(image_folder) if image_folder else self.data_path.parent

        # 加载数据
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")

    def _load_data(self) -> List[Dict]:
        """加载 JSON 数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 验证数据格式
        if not isinstance(data, list):
            raise ValueError("Data must be a list of samples")

        for idx, sample in enumerate(data):
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {idx} must be a dict")
            if 'image_path' not in sample:
                raise ValueError(f"Sample {idx} missing 'image_path'")
            if 'conversations' not in sample:
                raise ValueError(f"Sample {idx} missing 'conversations'")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本

        Returns:
            包含 pixel_values, input_ids, attention_mask, labels 的字
        """
        sample = self.data[idx]

        # 加载图片 - 如果image_path是绝对路径或已包含完整路径，直接使用；否则拼接image_folder
        image_path_str = sample['image_path']
        if Path(image_path_str).is_absolute() or Path(image_path_str).exists():
            # 绝对路径或相对于当前目录的完整路
            image_path = Path(image_path_str)
        else:
            # 相对路径 - 仅当不包含目录分隔符时才拼接image_folder
            # 如果已经images/xxx.png"格式，说明路径相对于数据集根目录
            if '/' in image_path_str or '\\' in image_path_str:
                # 已包含目录，相对于数据集根目录（data_path的父目录
                image_path = self.data_path.parent / image_path_str
            else:
                # 纯文件名，拼接image_folder
                image_path = self.image_folder / image_path_str

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

        # 构建对话文本
        conversations = sample['conversations']
        messages = []

        # 第一条消息需要包含图
        first_message = conversations[0]
        messages.append({
            "role": first_message["role"],
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": first_message["content"]}
            ]
        })

        # 其余消息只包含文
        for conv in conversations[1:]:
            messages.append({
                "role": conv["role"],
                "content": conv["content"]
            })

        # 使用 processor 处理
        # 参Qwen2-VL 官方示例
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize - 确保在CPU上处理（避免mindtorch device错误
        # 临时设置mindtorch和MindSpore为CPU模式
        from mindtorch._bind import set_default_device, get_default_device
        import mindspore

        original_device = get_default_device()
        original_ms_context = mindspore.get_context('device_target')

        try:
            # 强制使用CPU进行数据预处理（避免device:3等错误）
            set_default_device('cpu')
            mindspore.set_context(device_target='CPU')

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=False,  # 不做padding，collate时再处理
                truncation=False  # 禁用截断以避免image token不匹
            )
        finally:
            # 恢复原设备设
            set_default_device(original_device)
            if original_ms_context == 'Ascend':
                mindspore.set_context(device_target='Ascend')

        # 提取单个样本 (去除 batch 维度) - 强制所有tensor都在CPU

        result = {
            'input_ids': inputs['input_ids'][0].cpu(),
            'attention_mask': inputs['attention_mask'][0].cpu(),
            'pixel_values': inputs['pixel_values'][0].cpu() if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'][0].cpu() if 'image_grid_thw' in inputs else None,
        }

        # 构建 labels (用于计算 loss)
        # 对于训练,只计assistant 回复部分loss
        labels = result['input_ids'].clone()

        # 找到 assistant 开始的位置,之前token 设为 -100
        # 这是一个简化实实际可能需要根tokenizer 的特token 来判
        # TODO: 改进 label masking 逻辑

        result['labels'] = labels

        return result


@dataclass
class OCRDataCollator:
    """
    OCR 数据批处理器

    用于 DataLoader,将多个样本打包成 batch
    """
    processor: AutoProcessor
    padding: bool = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        批处理多个样

        Args:
            features: 样本列表

        Returns:
            批处理后的字
        """
        # 提取各个字段
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]

        # 图片可能不存纯文本样
        pixel_values = None
        if 'pixel_values' in features[0] and features[0]['pixel_values'] is not None:
            pixel_values = torch.stack([f['pixel_values'] for f in features])

        # Padding
        if self.padding:
            max_len = max(len(ids) for ids in input_ids)
            if self.max_length:
                max_len = min(max_len, self.max_length)

            # Pad input_ids and attention_mask
            input_ids = [
                torch.cat([ids, torch.full((max_len - len(ids),), self.processor.tokenizer.pad_token_id)])
                if len(ids) < max_len else ids[:max_len]
                for ids in input_ids
            ]
            attention_mask = [
                torch.cat([mask, torch.zeros(max_len - len(mask))])
                if len(mask) < max_len else mask[:max_len]
                for mask in attention_mask
            ]
            labels = [
                torch.cat([lbl, torch.full((max_len - len(lbl),), -100)])
                if len(lbl) < max_len else lbl[:max_len]
                for lbl in labels
            ]

        batch = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
        }

        if pixel_values is not None:
            batch['pixel_values'] = pixel_values

        # 添加image_grid_thw（Qwen2VL需要）
        if 'image_grid_thw' in features[0] and features[0]['image_grid_thw'] is not None:
            batch['image_grid_thw'] = torch.stack([f['image_grid_thw'] for f in features])

        return batch


def load_ocr_dataset(
    data_path: Union[str, Path],
    processor: AutoProcessor,
    max_length: int = 2048,
    image_folder: Optional[Union[str, Path]] = None,
) -> OCRDataset:
    """
    便捷函数: 加载 OCR 数据

    Args:
        data_path: JSON 数据文件路径
        processor: Qwen2VL processor
        max_length: 最大序列长
        image_folder: 图片文件夹路

    Returns:
        OCRDataset 实例
    """
    return OCRDataset(
        data_path=data_path,
        processor=processor,
        max_length=max_length,
        image_folder=image_folder,
    )


def split_dataset(
    dataset: OCRDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple:
    """
    分割数据集为训练集、验证集和测试集

    Args:
        dataset: OCRDataset 实例
        train_ratio: 训练集比
        val_ratio: 验证集比
        test_ratio: 测试集比
        random_seed: 随机种子

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import random_split

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    logger.info(f"Split dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

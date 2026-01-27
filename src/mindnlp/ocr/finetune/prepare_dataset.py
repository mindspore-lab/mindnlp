# -*- coding: utf-8 -*-
"""
OCR Dataset Preparation Utility
支持多种数据源和格式转换
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import shutil

from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_icdar2015(
    icdar_root: str,
    output_dir: str,
    split: str = "train"
) -> List[Dict]:
    """
    转换 ICDAR 2015 数据集格

    Args:
        icdar_root: ICDAR 2015 数据集根目录
        output_dir: 输出目录
        split: 数据集分("train" "test")

    Returns:
        转换后的数据列表
    """
    logger.info(f"Converting ICDAR 2015 {split} dataset...")

    icdar_root = Path(icdar_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ICDAR 2015 格式：图片文件夹 + 标注文件
    img_dir = icdar_root / split / "images"
    gt_dir = icdar_root / split / "gt"

    if not img_dir.exists() or not gt_dir.exists():
        logger.error(f"ICDAR 2015 directory not found: {icdar_root}")
        return []

    data = []

    # 遍历所有图
    for img_file in tqdm(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        img_id = img_file.stem
        gt_file = gt_dir / f"gt_{img_id}.txt"

        if not gt_file.exists():
            logger.warning(f"GT file not found for {img_id}")
            continue

        # 读取标注
        with open(gt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 提取所有文
        texts = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 9:  # 坐标 (8 + 文本
                text = ','.join(parts[8:])  # 文本可能包含逗号
                if text and text != "###":  # 忽略 "###" (难以识别)
                    texts.append(text)

        if not texts:
            continue

        # 拼接所有文本
        full_text = ' '.join(texts)

        # 复制图片
        new_img_name = f"icdar_{split}_{img_id}{img_file.suffix}"
        shutil.copy(img_file, output_dir / new_img_name)

        # 构建数据
        data.append({
            "image_path": new_img_name,
            "conversations": [
                {
                    "role": "user",
                    "content": "请识别图像中的所有文字"
                },
                {
                    "role": "assistant",
                    "content": full_text
                }
            ],
            "task_type": "general"
        })

    logger.info(f"Converted {len(data)} samples from ICDAR 2015 {split}")
    return data


def convert_funsd(
    funsd_root: str,
    output_dir: str,
    split: str = "train"
) -> List[Dict]:
    """
    转换 FUNSD 数据集格(表单理解)

    Args:
        funsd_root: FUNSD 数据集根目录
        output_dir: 输出目录
        split: 数据集分

    Returns:
        转换后的数据列表
    """
    logger.info(f"Converting FUNSD {split} dataset...")

    funsd_root = Path(funsd_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = funsd_root / split / "images"
    anno_dir = funsd_root / split / "annotations"

    if not img_dir.exists() or not anno_dir.exists():
        logger.error(f"FUNSD directory not found: {funsd_root}")
        return []

    data = []

    for anno_file in tqdm(list(anno_dir.glob("*.json"))):
        img_id = anno_file.stem
        img_file = img_dir / f"{img_id}.png"

        if not img_file.exists():
            logger.warning(f"Image not found for {img_id}")
            continue

        # 读取标注
        with open(anno_file, 'r', encoding='utf-8') as f:
            anno = json.load(f)

        # 提取所有文
        texts = []
        for item in anno.get('form', []):
            text = item.get('text', '').strip()
            if text:
                texts.append(text)

        if not texts:
            continue

        full_text = ' '.join(texts)

        # 复制图片
        new_img_name = f"funsd_{split}_{img_id}.png"
        shutil.copy(img_file, output_dir / new_img_name)

        # 构建数据
        data.append({
            "image_path": new_img_name,
            "conversations": [
                {
                    "role": "user",
                    "content": "请识别这份表单中的所有文字"
                },
                {
                    "role": "assistant",
                    "content": full_text
                }
            ],
            "task_type": "table"
        })

    logger.info(f"Converted {len(data)} samples from FUNSD {split}")
    return data


def convert_sroie(
    sroie_root: str,
    output_dir: str,
    split: str = "train"
) -> List[Dict]:
    """
    转换 SROIE 数据集格(收据文本提取)

    Args:
        sroie_root: SROIE 数据集根目录
        output_dir: 输出目录
        split: 数据集分

    Returns:
        转换后的数据列表
    """
    logger.info(f"Converting SROIE {split} dataset...")

    sroie_root = Path(sroie_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = sroie_root / split / "img"
    box_dir = sroie_root / split / "box"
    entities_dir = sroie_root / split / "entities"

    if not img_dir.exists():
        logger.error(f"SROIE directory not found: {sroie_root}")
        return []

    data = []

    for img_file in tqdm(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        img_id = img_file.stem

        # 优先使用entities文件，否则使用box文件
        txt_file = None
        if entities_dir and entities_dir.exists():
            txt_file = entities_dir / f"{img_id}.txt"
        if not txt_file or not txt_file.exists():
            if box_dir and box_dir.exists():
                txt_file = box_dir / f"{img_id}.txt"

        if not txt_file or not txt_file.exists():
            logger.warning(f"Text file not found for {img_id}")
            continue

        # 读取文本
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        texts = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 9:  # 坐标 + 文本
                text = ','.join(parts[8:])
                if text:
                    texts.append(text)
            elif len(parts) > 0:  # 只有文本
                texts.append(line.strip())

        if not texts:
            continue

        full_text = ' '.join(texts)

        # 复制图片
        new_img_name = f"sroie_{split}_{img_id}{img_file.suffix}"
        shutil.copy(img_file, output_dir / new_img_name)

        # 构建数据
        data.append({
            "image_path": new_img_name,
            "conversations": [
                {
                    "role": "user",
                    "content": "请识别这张收据中的所有文字"
                },
                {
                    "role": "assistant",
                    "content": full_text
                }
            ],
            "task_type": "general"
        })

    logger.info(f"Converted {len(data)} samples from SROIE {split}")
    return data


def convert_custom_format(
    data_dir: str,
    output_dir: str,
    task_type: str = "general"
) -> List[Dict]:
    """
    转换自定义格式数据集

    期望格式
    data_dir/
        images/
            img1.jpg
            img2.jpg
        labels.json  # {"img1.jpg": "text1", "img2.jpg": "text2", ...}

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        task_type: 任务类型

    Returns:
        转换后的数据列表
    """
    logger.info(f"Converting custom format dataset from {data_dir}...")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = data_dir / "images"
    label_file = data_dir / "labels.json"

    if not img_dir.exists() or not label_file.exists():
        logger.error(f"Custom format directory invalid: {data_dir}")
        return []

    # 读取标签
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    data = []

    for img_name, text in tqdm(labels.items()):
        img_file = img_dir / img_name

        if not img_file.exists():
            logger.warning(f"Image not found: {img_name}")
            continue

        # 复制图片
        new_img_name = f"custom_{task_type}_{img_name}"
        shutil.copy(img_file, output_dir / new_img_name)

        # 构建数据
        data.append({
            "image_path": new_img_name,
            "conversations": [
                {
                    "role": "user",
                    "content": "请识别图像中的所有文字"
                },
                {
                    "role": "assistant",
                    "content": text
                }
            ],
            "task_type": task_type
        })

    logger.info(f"Converted {len(data)} samples from custom format")
    return data


def validate_dataset(dataset_file: str) -> Tuple[int, int, int]:
    """
    验证数据集格式和质量

    Args:
        dataset_file: 数据集文件路

    Returns:
        (valid_count, error_count, missing_images)
    """
    logger.info(f"Validating dataset: {dataset_file}")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Total samples: {len(data)}")

    valid_count = 0
    error_count = 0
    missing_images = []

    for i, item in enumerate(tqdm(data, desc="Validating")):
        try:
            # 检查必需字段
            assert 'image_path' in item or 'image' in item, "缺少image_path字段"
            assert 'conversations' in item, "缺少conversations字段"
            assert len(item['conversations']) == 2, "conversations应包含2条消息"

            # 检查图像文件
            img_path = item.get('image_path') or item.get('image')
            dataset_dir = Path(dataset_file).parent
            full_img_path = dataset_dir / img_path

            if not full_img_path.exists():
                missing_images.append(img_path)
            else:
                # 尝试打开图像验证
                try:
                    Image.open(full_img_path).verify()
                except Exception as e:
                    logger.warning(f"Image corrupted: {img_path}")
                    error_count += 1
                    continue

            valid_count += 1
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                logger.error(f"Sample {i} error: {e}")

    logger.info(f"Valid samples: {valid_count}")
    logger.info(f"Error samples: {error_count}")
    logger.info(f"Missing images: {len(missing_images)}")

    if missing_images and len(missing_images) <= 10:
        logger.info("Missing image files:")
        for img in missing_images[:10]:
            logger.info(f"  - {img}")

    return valid_count, error_count, len(missing_images)


def merge_and_split_datasets(
    datasets: List[List[Dict]],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    合并多个数据集并分割为训验证/测试

    Args:
        datasets: 数据集列
        output_dir: 输出目录
        train_ratio: 训练集比
        val_ratio: 验证集比
        test_ratio: 测试集比
        random_seed: 随机种子
    """
    import random

    logger.info("Merging and splitting datasets...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 合并所有数
    all_data = []
    for dataset in datasets:
        all_data.extend(dataset)

    logger.info(f"Total samples: {len(all_data)}")

    # 打乱数据
    random.seed(random_seed)
    random.shuffle(all_data)

    # 分割数据
    total = len(all_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 保存
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_file = output_dir / f"{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="OCR Dataset Preparation")

    parser.add_argument("--format", type=str, required=True,
                        choices=["icdar2015", "funsd", "sroie", "custom"],
                        help="数据集格式")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据集目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--split", type=str, default="train",
                        help="数据集分(train/test/val)")
    parser.add_argument("--task_type", type=str, default="general",
                        choices=["general", "table", "formula", "handwriting"],
                        help="任务类型 (仅用custom 格式)")

    # 数据集分割参数
    parser.add_argument("--merge_and_split", action="store_true",
                        help="合并所有数据并重新分割")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")

    # 验证参数
    parser.add_argument("--validate", action="store_true",
                        help="转换后验证数据集")

    args = parser.parse_args()

    # 转换数据
    if args.format == "icdar2015":
        data = convert_icdar2015(args.data_dir, args.output_dir, args.split)
    elif args.format == "funsd":
        data = convert_funsd(args.data_dir, args.output_dir, args.split)
    elif args.format == "sroie":
        data = convert_sroie(args.data_dir, args.output_dir, args.split)
    elif args.format == "custom":
        data = convert_custom_format(args.data_dir, args.output_dir, args.task_type)
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    # 保存转换后的数据
    if not args.merge_and_split:
        output_file = Path(args.output_dir) / f"{args.split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} samples to {output_file}")

        # 验证数据
        if args.validate:
            validate_dataset(str(output_file))
    else:
        merge_and_split_datasets(
            [data],
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.random_seed,
        )

    logger.info("Dataset preparation completed!")


if __name__ == "__main__":
    main()

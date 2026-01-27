# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
OCR 数据集格式转换工
将公开数据集转换为 MindNLP OCR 微调格式
"""

import json
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_sroie(dataset_path: str, output_dir: str):
    """
    转换 SROIE 数据
    SROIE: 收据文字提取
    """
    print("转换 SROIE 数据..")

    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"加载数据集失 {e}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    samples = []

    for split in ['train', 'test']:
        if split not in dataset:
            continue

        print(f"处理 {split} ..")

        for idx, item in enumerate(tqdm(dataset[split])):
            # 保存图片
            if 'image' in item:
                image = item['image']
                if isinstance(image, Image.Image):
                    image_filename = f"{split}_{idx:05d}.jpg"
                    image_path = images_dir / image_filename
                    image.save(image_path)
                else:
                    image_filename = f"{split}_{idx:05d}.jpg"

            # 提取文本
            text_content = ""
            if 'text' in item:
                text_content = item['text']
            elif 'words' in item:
                text_content = ' '.join(item['words'])

            # 构建样本
            sample = {
                "image_path": f"images/{image_filename}",
                "conversations": [
                    {
                        "role": "user",
                        "content": "请识别这张收据中的所有文字"
                    },
                    {
                        "role": "assistant",
                        "content": text_content
                    }
                ],
                "task_type": "receipt"
            }
            samples.append(sample)

    # 保存 JSON
    output_file = output_dir / "train.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"转换完成: {output_file}")
    print(f"  样本 {len(samples)}")
    print(f"  图片 {len(list(images_dir.glob('*.jpg')))}")


def convert_funsd(dataset_path: str, output_dir: str):
    """
    转换 FUNSD 数据
    FUNSD: 表单理解
    """
    print("转换 FUNSD 数据..")

    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"加载数据集失 {e}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    samples = []

    for split in ['train', 'test']:
        if split not in dataset:
            continue

        print(f"处理 {split} ..")

        for idx, item in enumerate(tqdm(dataset[split])):
            # 保存图片
            if 'image' in item:
                image = item['image']
                if isinstance(image, Image.Image):
                    image_filename = f"{split}_{idx:05d}.png"
                    image_path = images_dir / image_filename
                    image.save(image_path)

            # 提取文本和标
            text_content = ""
            if 'words' in item:
                text_content = ' '.join(item['words'])

            # 构建样本
            sample = {
                "image_path": f"images/{image_filename}",
                "conversations": [
                    {
                        "role": "user",
                        "content": "请识别这个表单中的所有内容"
                    },
                    {
                        "role": "assistant",
                        "content": text_content
                    }
                ],
                "task_type": "form"
            }
            samples.append(sample)

    # 保存 JSON
    output_file = output_dir / "train.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"转换完成: {output_file}")
    print(f"  样本 {len(samples)}")


def convert_docvqa(dataset_path: str, output_dir: str):
    """
    转换 DocVQA 数据
    DocVQA: 文档问答
    """
    print("转换 DocVQA 数据..")

    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"加载数据集失 {e}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    samples = []

    for split in ['train', 'test', 'validation']:
        if split not in dataset:
            continue

        print(f"处理 {split} ..")

        for idx, item in enumerate(tqdm(dataset[split])):
            # 保存图片
            if 'image' in item:
                image = item['image']
                if isinstance(image, Image.Image):
                    image_filename = f"{split}_{idx:05d}.png"
                    image_path = images_dir / image_filename
                    image.save(image_path)

            # 提取问题和答
            question = item.get('question', '')
            answers = item.get('answers', [])
            answer_text = answers[0] if answers else ''

            # 构建样本
            sample = {
                "image_path": f"images/{image_filename}",
                "conversations": [
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": answer_text
                    }
                ],
                "task_type": "vqa"
            }
            samples.append(sample)

    # 保存 JSON
    output_file = output_dir / "train.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"转换完成: {output_file}")
    print(f"  样本 {len(samples)}")


def main():
    if len(sys.argv) < 4:
        print("用法: python convert_datasets.py <数据集类型> <输入路径> <输出路径>")
        print("")
        print("数据集类型:")
        print("  sroie   - SROIE 收据数据集")
        print("  funsd   - FUNSD 表单数据集")
        print("  docvqa  - DocVQA 问答数据集")
        print("")
        print("示例:")
        print("  python convert_datasets.py sroie ./datasets/sroie ./datasets/sroie_converted")
        sys.exit(1)

    dataset_type = sys.argv[1].lower()
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    print("=" * 60)
    print(f"数据集转换工具")
    print("=" * 60)
    print(f"类型: {dataset_type}")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print("")

    if dataset_type == 'sroie':
        convert_sroie(input_path, output_path)
    elif dataset_type == 'funsd':
        convert_funsd(input_path, output_path)
    elif dataset_type == 'docvqa':
        convert_docvqa(input_path, output_path)
    else:
        print(f"错误: 不支持的数据集类'{dataset_type}'")
        sys.exit(1)

    print("")
    print("=" * 60)
    print("转换完成")
    print("=" * 60)
    print("")
    print("后续步骤:")
    print(f"1. 检查数 ls {output_path}")
    print(f"2. 开始训")
    print(f"   export DATA_PATH='{output_path}/train.json'")
    print(f"   export IMAGE_FOLDER='{output_path}/images'")
    print(f"   ./scripts/ocr/train_lora_npu.sh")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
通过 OCR API 进行模型评估
避免直接推理时的 NPU 兼容性问
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
import requests
from tqdm import tqdm
import editdistance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """计算字符错误(Character Error Rate)"""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    distance = editdistance.eval(reference, hypothesis)
    return distance / len(reference)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """计算词错误率 (Word Error Rate)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    distance = editdistance.eval(ref_words, hyp_words)
    return distance / len(ref_words)


def load_test_data(test_data_path: str, image_folder: str = None) -> List[Dict]:
    """加载测试数据

    支持两种格式:
    1. 标准格式: {"image": "path", "text": "content"}
    2. FUNSD格式: {"image_path": "path", "conversations": [...]}
    """
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 标准化数据格
    standardized_data = []
    test_data_dir = Path(test_data_path).parent

    for item in data:
        # 提取图片路径
        if 'image_path' in item:
            image_path = item['image_path']
        elif 'image' in item:
            image_path = item['image']
        else:
            logger.warning(f"No image path found in item: {item}")
            continue

        # 如果是相对路径，拼接完整路径
        if not Path(image_path).is_absolute():
            if image_folder:
                image_path = str(Path(image_folder) / image_path)
            else:
                # 假设图片在测试数据同一目录
                image_path = str(test_data_dir / image_path)

        # 提取文本内容
        if 'conversations' in item:
            # FUNSD 格式
            for conv in item['conversations']:
                if conv.get('role') == 'assistant':
                    text = conv.get('content', '')
                    break
            else:
                text = ''
        elif 'text' in item:
            text = item['text']
        else:
            logger.warning(f"No text content found in item: {item}")
            text = ''

        standardized_data.append({
            'image': image_path,
            'text': text,
            'original': item
        })

    logger.info(f"Loaded {len(standardized_data)} test samples")
    return standardized_data


def evaluate_via_api(
    api_url: str,
    test_data: List[Dict],
    task_type: str = "document",
    output_file: str = None
) -> Dict:
    """通过 API 进行评估"""

    references = []
    hypotheses = []
    failed_count = 0

    logger.info(f"Starting evaluation via API: {api_url}")
    logger.info(f"Total samples: {len(test_data)}")

    for item in tqdm(test_data, desc="Evaluating"):
        try:
            image_path = item.get('image')
            reference = item.get('text', '')

            # 调试：检查路
            if not image_path:
                logger.error(f"Empty image path in item: {item.get('original', {}).get('image_path', 'N/A')}")
                failed_count += 1
                continue

            if not Path(image_path).exists():
                logger.error(f"Image not found: {image_path}")
                failed_count += 1
                continue

            # 调用 API - with 块内完成请求
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/png')}
                data = {'task_type': task_type}

                response = requests.post(
                    f"{api_url}/api/v1/ocr/predict",
                    files=files,
                    data=data,
                    timeout=120
                )

            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                failed_count += 1
                continue

            result = response.json()
            # API 返回 raw_output 字段包含完整文本
            hypothesis = result.get('raw_output', '')

            # 如果 raw_output 为空，尝试从 texts 数组获取
            if not hypothesis and 'texts' in result and result['texts']:
                hypothesis = result['texts'][0] if isinstance(result['texts'], list) else result['texts']

            # 移除 HTML 标签（如果存在）
            if hypothesis.startswith('<html>'):
                import re
                hypothesis = re.sub(r'<[^>]+>', '', hypothesis)
                hypothesis = hypothesis.strip()

            references.append(reference)
            hypotheses.append(hypothesis)

        except Exception as e:
            logger.error(f"Sample evaluation failed: {e}")
            failed_count += 1

    # 计算指标
    if not references:
        logger.error("No successful evaluations!")
        return {
            'cer': 0.0,
            'wer': 0.0,
            'exact_match': 0.0,
            'total_samples': len(test_data),
            'successful_samples': 0,
            'failed_samples': failed_count
        }

    cer_scores = [calculate_cer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    wer_scores = [calculate_wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    exact_matches = [ref == hyp for ref, hyp in zip(references, hypotheses)]

    metrics = {
        'cer': sum(cer_scores) / len(cer_scores),
        'wer': sum(wer_scores) / len(wer_scores),
        'exact_match': sum(exact_matches) / len(exact_matches),
        'total_samples': len(test_data),
        'successful_samples': len(references),
        'failed_samples': failed_count
    }

    logger.info("=" * 80)
    logger.info("Evaluation Results:")
    logger.info(f"  Total Samples: {metrics['total_samples']}")
    logger.info(f"  Successful: {metrics['successful_samples']}")
    logger.info(f"  Failed: {metrics['failed_samples']}")
    logger.info(f"  CER: {metrics['cer']:.4f}")
    logger.info(f"  WER: {metrics['wer']:.4f}")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info("=" * 80)

    # 保存结果
    if output_file:
        results = {
            'metrics': metrics,
            'samples': [
                {
                    'reference': ref,
                    'hypothesis': hyp,
                    'cer': cer,
                    'wer': wer
                }
                for ref, hyp, cer, wer in zip(references, hypotheses, cer_scores, wer_scores)
            ]
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR model via API")

    parser.add_argument("--api_url", type=str, default="http://localhost:8000",
                        help="OCR API URL")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="测试数据 JSON 文件路径")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="图片文件夹路径（如果需要覆盖 JSON 中的路径)")
    parser.add_argument("--task_type", type=str, default="document",
                        choices=["general", "document", "handwriting"],
                        help="OCR 任务类型")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出结果文件路径")

    args = parser.parse_args()

    # 加载测试数据
    test_data = load_test_data(args.test_data_path, args.image_folder)

    # 评估
    metrics = evaluate_via_api(
        api_url=args.api_url,
        test_data=test_data,
        task_type=args.task_type,
        output_file=args.output_file
    )

    return metrics


if __name__ == "__main__":
    main()

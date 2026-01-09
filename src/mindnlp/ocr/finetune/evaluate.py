"""
OCR Model Evaluation Module
实现 CER, WER 和任务特定准确率评估
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
import editdistance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    计算字符错误率 (Character Error Rate)
    
    CER = (S + D + I) / N
    S: 替换数, D: 删除数, I: 插入数, N: 参考字符数
    
    Args:
        reference: 参考文本
        hypothesis: 识别结果
        
    Returns:
        CER 值 (0-1)
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    # 使用 editdistance 计算编辑距离
    distance = editdistance.eval(reference, hypothesis)
    cer = distance / len(reference)
    
    return cer


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    计算词错误率 (Word Error Rate)
    
    WER = (S + D + I) / N
    S: 替换数, D: 删除数, I: 插入数, N: 参考词数
    
    Args:
        reference: 参考文本
        hypothesis: 识别结果
        
    Returns:
        WER 值 (0-1)
    """
    # 分词 (简单按空格分割)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    
    # 计算编辑距离
    distance = editdistance.eval(ref_words, hyp_words)
    wer = distance / len(ref_words)
    
    return wer


def calculate_accuracy(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    计算多个样本的平均准确率指标
    
    Args:
        references: 参考文本列表
        hypotheses: 识别结果列表
        
    Returns:
        指标字典: {
            "cer": 平均 CER,
            "wer": 平均 WER,
            "exact_match": 完全匹配率
        }
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Length mismatch: {len(references)} refs vs {len(hypotheses)} hyps")
    
    cer_scores = []
    wer_scores = []
    exact_matches = 0
    
    for ref, hyp in zip(references, hypotheses):
        cer = calculate_cer(ref, hyp)
        wer = calculate_wer(ref, hyp)
        
        cer_scores.append(cer)
        wer_scores.append(wer)
        
        if ref == hyp:
            exact_matches += 1
    
    metrics = {
        "cer": sum(cer_scores) / len(cer_scores) if cer_scores else 0.0,
        "wer": sum(wer_scores) / len(wer_scores) if wer_scores else 0.0,
        "exact_match": exact_matches / len(references) if references else 0.0,
    }
    
    return metrics


def calculate_task_accuracy(
    references: List[str],
    hypotheses: List[str],
    task_labels: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    按任务类型计算准确率
    
    Args:
        references: 参考文本列表
        hypotheses: 识别结果列表
        task_labels: 任务类型标签列表 (e.g., "table", "formula", "general")
        
    Returns:
        按任务分组的指标字典
    """
    # 按任务分组
    task_groups = defaultdict(lambda: {"refs": [], "hyps": []})
    
    for ref, hyp, task in zip(references, hypotheses, task_labels):
        task_groups[task]["refs"].append(ref)
        task_groups[task]["hyps"].append(hyp)
    
    # 计算每个任务的指标
    task_metrics = {}
    for task, data in task_groups.items():
        metrics = calculate_accuracy(data["refs"], data["hyps"])
        task_metrics[task] = metrics
        logger.info(f"Task '{task}': CER={metrics['cer']:.4f}, WER={metrics['wer']:.4f}, EM={metrics['exact_match']:.4f}")
    
    return task_metrics


@torch.no_grad()
def evaluate_model(
    model,
    processor: AutoProcessor,
    test_data_path: str,
    image_folder: Optional[str] = None,
    max_length: int = 2048,
    batch_size: int = 1,
    device: str = "cuda",
    output_file: Optional[str] = None,
) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: 微调后的模型
        processor: Processor
        test_data_path: 测试数据 JSON 文件路径
        image_folder: 图片文件夹路径
        max_length: 最大生成长度
        batch_size: 批大小
        device: 设备
        output_file: 输出结果文件路径 (可选)
        
    Returns:
        评估指标字典
    """
    logger.info("=" * 80)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 80)
    
    # 加载测试数据
    test_data_path = Path(test_data_path)
    image_folder = Path(image_folder) if image_folder else test_data_path.parent
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test samples from {test_data_path}")
    
    # 准备模型
    model.eval()
    model = model.to(device)
    
    # 推理
    references = []
    hypotheses = []
    task_labels = []
    
    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        # 加载图片
        image_path = image_folder / sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            continue
        
        # 构建输入
        conversations = sample['conversations']
        
        # 使用第一条用户消息作为输入 (带图片)
        user_message = conversations[0]
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_message["content"]}
            ]
        }]
        
        # 处理输入
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
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,  # 贪婪解码
            num_beams=1,
        )
        
        # 解码
        generated_text = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 提取 assistant 回复部分
        # 简化处理: 假设生成文本就是回复
        hypothesis = generated_text.strip()
        
        # 获取参考答案 (第一条 assistant 消息)
        reference = ""
        for conv in conversations:
            if conv["role"] == "assistant":
                reference = conv["content"]
                break
        
        # 任务标签 (如果有)
        task_label = sample.get("task_type", "general")
        
        references.append(reference)
        hypotheses.append(hypothesis)
        task_labels.append(task_label)
    
    # 计算整体指标
    overall_metrics = calculate_accuracy(references, hypotheses)
    
    logger.info("=" * 80)
    logger.info("Overall Metrics:")
    logger.info(f"  CER: {overall_metrics['cer']:.4f}")
    logger.info(f"  WER: {overall_metrics['wer']:.4f}")
    logger.info(f"  Exact Match: {overall_metrics['exact_match']:.4f}")
    logger.info("=" * 80)
    
    # 按任务类型计算指标
    task_metrics = calculate_task_accuracy(references, hypotheses, task_labels)
    
    logger.info("Task-specific Metrics:")
    for task, metrics in task_metrics.items():
        logger.info(f"  {task}:")
        logger.info(f"    CER: {metrics['cer']:.4f}")
        logger.info(f"    WER: {metrics['wer']:.4f}")
        logger.info(f"    Exact Match: {metrics['exact_match']:.4f}")
    
    # 保存结果
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "overall_metrics": overall_metrics,
            "task_metrics": task_metrics,
            "predictions": [
                {
                    "reference": ref,
                    "hypothesis": hyp,
                    "task": task,
                    "cer": calculate_cer(ref, hyp),
                    "wer": calculate_wer(ref, hyp),
                }
                for ref, hyp, task in zip(references, hypotheses, task_labels)
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    return overall_metrics


def main():
    parser = argparse.ArgumentParser(description="OCR Model Evaluation")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="微调后的模型路径")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="测试数据 JSON 文件路径")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="图片文件夹路径")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出结果文件路径")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大生成长度")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    
    args = parser.parse_args()
    
    # 加载模型和 processor
    from .train_lora import load_lora_model
    
    logger.info(f"Loading model from {args.model_path}")
    model, processor = load_lora_model(
        base_model_path=args.model_path,
        lora_path=args.model_path,
        device=args.device,
    )
    
    # 执行评估
    evaluate_model(
        model=model,
        processor=processor,
        test_data_path=args.test_data_path,
        image_folder=args.image_folder,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()

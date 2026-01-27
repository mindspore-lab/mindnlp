# -*- coding: utf-8 -*-
"""
OCR Model Evaluation Module
实现 CER, WER 和任务特定准确率评估
"""
# 必须在最开始阻止mindnlp patch的应确保使用纯PyTorch路径
import sys
import os

# 阻止mindnlp自动导入和patch应用
os.environ['MINDNLP_DISABLE_PATCHES'] = '1'

# 如果mindnlp已经被导移除它以避免patch生效
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('mindnlp')]
for module in modules_to_remove:
    del sys.modules[module]

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re

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
    计算字符错误(Character Error Rate)

    CER = (S + D + I) / N
    S: 替换 D: 删除 I: 插入 N: 参考字符数

    Args:
        reference: 参考文
        hypothesis: 识别结果

    Returns:
        CER (0-1)
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
    S: 替换 D: 删除 I: 插入 N: 参考词

    Args:
        reference: 参考文
        hypothesis: 识别结果

    Returns:
        WER (0-1)
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
        references: 参考文本列
        hypotheses: 识别结果列表

    Returns:
        指标字典: {
            "cer": 平均 CER,
            "wer": 平均 WER,
            "exact_match": 完全匹配
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
        references: 参考文本列
        hypotheses: 识别结果列表
        task_labels: 任务类型标签列表 (e.g., "table", "formula", "general")

    Returns:
        按任务分组的指标字典
    """
    # 按任务分
    task_groups = defaultdict(lambda: {"refs": [], "hyps": []})

    for ref, hyp, task in zip(references, hypotheses, task_labels):
        task_groups[task]["refs"].append(ref)
        task_groups[task]["hyps"].append(hyp)

    # 计算每个任务的指
    task_metrics = {}
    for task, data in task_groups.items():
        metrics = calculate_accuracy(data["refs"], data["hyps"])
        task_metrics[task] = metrics
        logger.info(f"Task '{task}': CER={metrics['cer']:.4f}, WER={metrics['wer']:.4f}, EM={metrics['exact_match']:.4f}")

    return task_metrics


def evaluate_table_accuracy(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    评估表格识别准确

    表格识别特定指标
    - Cell Accuracy: 单元格级别的准确
    - Row Accuracy: 行级别的准确
    - Structure Accuracy: 表格结构准确

    Args:
        references: 参考文本列(应包含表格标记，|, -, \n)
        hypotheses: 识别结果列表

    Returns:
        表格特定指标
    """
    cell_correct = 0
    cell_total = 0
    row_correct = 0
    row_total = 0
    structure_correct = 0

    for ref, hyp in zip(references, hypotheses):
        # 解析表格
        ref_rows = ref.split('\n')
        hyp_rows = hyp.split('\n')

        row_total += len(ref_rows)

        # 行级别比
        for ref_row, hyp_row in zip(ref_rows, hyp_rows):
            if ref_row.strip() == hyp_row.strip():
                row_correct += 1

            # 单元格级别比
            ref_cells = [c.strip() for c in ref_row.split('|') if c.strip()]
            hyp_cells = [c.strip() for c in hyp_row.split('|') if c.strip()]

            cell_total += len(ref_cells)

            for ref_cell, hyp_cell in zip(ref_cells, hyp_cells):
                if ref_cell == hyp_cell:
                    cell_correct += 1

        # 结构准确率：行数和列数是否匹
        ref_col_count = len(ref_rows[0].split('|')) if ref_rows else 0
        hyp_col_count = len(hyp_rows[0].split('|')) if hyp_rows else 0

        if len(ref_rows) == len(hyp_rows) and ref_col_count == hyp_col_count:
            structure_correct += 1

    metrics = {
        "cell_accuracy": cell_correct / cell_total if cell_total > 0 else 0.0,
        "row_accuracy": row_correct / row_total if row_total > 0 else 0.0,
        "structure_accuracy": structure_correct / len(references) if references else 0.0,
    }

    return metrics


def evaluate_formula_accuracy(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    评估公式识别准确

    公式识别特定指标
    - Symbol Accuracy: 符号级别准确
    - LaTeX Structure Accuracy: LaTeX 结构准确
    - Exact Match: 完全匹配

    Args:
        references: 参考公式文(LaTeX 格式)
        hypotheses: 识别结果

    Returns:
        公式特定指标
    """
    symbol_correct = 0
    symbol_total = 0
    structure_correct = 0
    exact_matches = 0

    # LaTeX 特殊符号
    latex_symbols = re.compile(r'\\[a-zA-Z]+|\{|\}|\^|_|\[|\]|\(|\)')

    for ref, hyp in zip(references, hypotheses):
        # 完全匹配
        if ref == hyp:
            exact_matches += 1

        # 提取 LaTeX 符号
        ref_symbols = latex_symbols.findall(ref)
        hyp_symbols = latex_symbols.findall(hyp)

        symbol_total += len(ref_symbols)

        # 符号级别比较
        for ref_sym, hyp_sym in zip(ref_symbols, hyp_symbols):
            if ref_sym == hyp_sym:
                symbol_correct += 1

        # 结构准确率：括号、上下标匹配
        ref_braces = ref.count('{') == ref.count('}')
        hyp_braces = hyp.count('{') == hyp.count('}')
        ref_parens = ref.count('(') == ref.count(')')
        hyp_parens = hyp.count('(') == hyp.count(')')

        if ref_braces == hyp_braces and ref_parens == hyp_parens:
            structure_correct += 1

    metrics = {
        "symbol_accuracy": symbol_correct / symbol_total if symbol_total > 0 else 0.0,
        "structure_accuracy": structure_correct / len(references) if references else 0.0,
        "exact_match": exact_matches / len(references) if references else 0.0,
    }

    return metrics


def load_lora_model(base_model_path: str, lora_path: str, device: Optional[str] = None):
    """
    加载LoRA模型 (纯PyTorch实现,避免mindnlp依赖)

    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA adapter路径
        device: 目标设备

    Returns:
        (model, processor)
    """
    from transformers import Qwen2VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, PeftModel
    import numpy as np

    # 自动检测设
    if device is None:
        try:
            import torch_npu
            if torch.npu.is_available():
                device = "npu:0"
                logger.info("Auto-detected NPU device")
        except ImportError:
            pass

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Auto-detected CUDA device")
            else:
                device = "cpu"
                logger.info("Using CPU device")

    logger.info(f"Using device: {device}")
    logger.info(f"Loading base model from {base_model_path}")

    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # 设置 left-padding 用于批量推理 (decoder-only 模型必需)
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'left'
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        # 确保 pad_token_id int 而不tensor
        if hasattr(processor.tokenizer.pad_token_id, 'item'):
            processor.tokenizer.pad_token_id = processor.tokenizer.pad_token_id.item()
        if hasattr(processor.tokenizer.eos_token_id, 'item'):
            processor.tokenizer.eos_token_id = processor.tokenizer.eos_token_id.item()

    # 加载基础模型
    logger.info("Loading base model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,  # 禁用以避免meta tensor
        attn_implementation="eager",  # NPU兼容
        trust_remote_code=True
    )

    # 加载 LoRA 权重
    logger.info(f"Loading LoRA weights from {lora_path}")
    lora_path = Path(lora_path).resolve()

    # 检查是否是.npz格式
    adapter_file = lora_path / "adapter_model.npz"
    if adapter_file.exists():
        logger.info("Loading MindSpore format (.npz) with merged LoRA weights")
        logger.info("Using direct NPZ loading (same as API) to avoid memory issues")

        # 直接NPZ 加载完整模型（基础权重 + LoRA 已合并）
        logger.info(f"Creating empty model architecture from {base_model_path}")

        # 创建空模型架
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

        # 针对 NPU 配置
        if "npu" in device:
            config.attn_implementation = "eager"

        base_model = Qwen2VLForConditionalGeneration(config)
        base_model = base_model.to(torch.float16)

        logger.info(f"Loading weights from NPZ: {adapter_file}")
        logger.info(f"File size: {adapter_file.stat().st_size / (1024**3):.2f} GB")

        # 加载 NPZ 数据
        data = np.load(str(adapter_file))
        logger.info(f"Found {len(data.files)} weight tensors in NPZ")

        # 构建模型参数字典
        param_dict = dict(base_model.named_parameters())
        buffer_dict = dict(base_model.named_buffers())

        # 分组权重：base_layer、lora_A、lora_B
        base_weights = {}
        lora_a_weights = {}
        lora_b_weights = {}

        # 收集所有权
        for npz_key in data.files:
            if 'lora_A.default' in npz_key:
                module_name = npz_key.replace('base_model.model.', '').replace('.lora_A.default.weight', '')
                lora_a_weights[module_name] = data[npz_key]
            elif 'lora_B.default' in npz_key:
                module_name = npz_key.replace('base_model.model.', '').replace('.lora_B.default.weight', '')
                lora_b_weights[module_name] = data[npz_key]
            elif '.base_layer.weight' in npz_key:
                module_name = npz_key.replace('base_model.model.', '').replace('.base_layer.weight', '.weight')
                base_weights[module_name] = data[npz_key]
            elif '.base_layer.bias' in npz_key:
                module_name = npz_key.replace('base_model.model.', '').replace('.base_layer.bias', '.bias')
                base_weights[module_name] = data[npz_key]
            else:
                module_name = npz_key.replace('base_model.model.', '')
                base_weights[module_name] = data[npz_key]

        logger.info(f"Found {len(base_weights)} base weights, {len(lora_a_weights)} LoRA-A, {len(lora_b_weights)} LoRA-B")

        # 加载并合并权
        loaded_count = 0
        merged_count = 0

        for module_name, base_weight in base_weights.items():
            # 查找对应的模型参
            target_param = None
            if module_name in param_dict:
                target_param = param_dict[module_name]
            elif module_name in buffer_dict:
                target_param = buffer_dict[module_name]

            if target_param is not None:
                # 转换torch tensor
                torch_weight = torch.from_numpy(base_weight).to(dtype=torch.float16)

                # 检查是否有对应LoRA 权重（只2D 权重合并
                module_base_name = module_name.replace('.weight', '').replace('.bias', '')

                if (module_base_name in lora_a_weights and
                    module_base_name in lora_b_weights and
                    len(base_weight.shape) == 2):

                    lora_a_np = lora_a_weights[module_base_name]
                    lora_b_np = lora_b_weights[module_base_name]

                    if len(lora_a_np.shape) == 2 and len(lora_b_np.shape) == 2:
                        lora_a = torch.from_numpy(lora_a_np).to(dtype=torch.float16)
                        lora_b = torch.from_numpy(lora_b_np).to(dtype=torch.float16)

                        # 合并 LoRA: weight = base + lora_B @ lora_A
                        lora_delta = torch.matmul(lora_b, lora_a)
                        torch_weight = torch_weight + lora_delta

                        del lora_a, lora_b, lora_delta
                        merged_count += 1

                # 替换参数
                with torch.no_grad():
                    target_param.copy_(torch_weight)

                del base_weight, torch_weight
                loaded_count += 1

        logger.info(f"Loaded {loaded_count} weights ({merged_count} LoRA merged)")
        model = base_model
    else:
        # 标准PEFT格式
        logger.info("Loading standard PEFT format LoRA weights")
        model = PeftModel.from_pretrained(
            base_model,
            str(lora_path),
            is_trainable=False
        )

    # 移动到设
    logger.info(f"Moving model to device: {device}")
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    return model, processor


def evaluate_handwriting_accuracy(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    评估手写体识别准确率

    手写体识别特定指标：
    - Character-level CER: 字符级别错误
    - Normalized Edit Distance: 归一化编辑距
    - Confusion Analysis: 常见混淆字符分析

    Args:
        references: 参考文
        hypotheses: 识别结果

    Returns:
        手写体特定指
    """
    cer_scores = []
    normalized_distances = []
    confusion_pairs = defaultdict(int)

    for ref, hyp in zip(references, hypotheses):
        # CER
        cer = calculate_cer(ref, hyp)
        cer_scores.append(cer)

        # 归一化编辑距
        distance = editdistance.eval(ref, hyp)
        max_len = max(len(ref), len(hyp))
        normalized_distance = distance / max_len if max_len > 0 else 0.0
        normalized_distances.append(normalized_distance)

        # 混淆字符分析
        # 简化版：只记录替换操作
        for i, (r, h) in enumerate(zip(ref, hyp)):
            if r != h:
                confusion_pairs[f"{r}->{h}"] += 1

    # 获取0个最常见的混淆对
    top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

    metrics = {
        "cer": sum(cer_scores) / len(cer_scores) if cer_scores else 0.0,
        "normalized_edit_distance": sum(normalized_distances) / len(normalized_distances) if normalized_distances else 0.0,
        "top_confusions": dict(top_confusions),
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
        references: 参考文本列
        hypotheses: 识别结果列表
        task_labels: 任务类型标签列表 (e.g., "table", "formula", "handwriting", "general")

    Returns:
        按任务分组的指标字典
    """
    # 按任务分
    task_groups = defaultdict(lambda: {"refs": [], "hyps": []})

    for ref, hyp, task in zip(references, hypotheses, task_labels):
        task_groups[task]["refs"].append(ref)
        task_groups[task]["hyps"].append(hyp)

    # 计算每个任务的指
    task_metrics = {}
    for task, data in task_groups.items():
        refs = data["refs"]
        hyps = data["hyps"]

        # 基础指标
        base_metrics = calculate_accuracy(refs, hyps)
        task_metrics[task] = base_metrics

        # 任务特定指标
        if task == "table":
            table_metrics = evaluate_table_accuracy(refs, hyps)
            task_metrics[task].update({"table_metrics": table_metrics})
            logger.info(f"Table Task - Cell Acc: {table_metrics['cell_accuracy']:.4f}, "
                       f"Row Acc: {table_metrics['row_accuracy']:.4f}, "
                       f"Structure Acc: {table_metrics['structure_accuracy']:.4f}")
        elif task == "formula":
            formula_metrics = evaluate_formula_accuracy(refs, hyps)
            task_metrics[task].update({"formula_metrics": formula_metrics})
            logger.info(f"Formula Task - Symbol Acc: {formula_metrics['symbol_accuracy']:.4f}, "
                       f"Structure Acc: {formula_metrics['structure_accuracy']:.4f}")
        elif task == "handwriting":
            handwriting_metrics = evaluate_handwriting_accuracy(refs, hyps)
            task_metrics[task].update({"handwriting_metrics": handwriting_metrics})
            logger.info(f"Handwriting Task - CER: {handwriting_metrics['cer']:.4f}, "
                       f"Normalized Dist: {handwriting_metrics['normalized_edit_distance']:.4f}")

        logger.info(f"Task '{task}': CER={base_metrics['cer']:.4f}, WER={base_metrics['wer']:.4f}, EM={base_metrics['exact_match']:.4f}")

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
        image_folder: 图片文件夹路
        max_length: 最大生成长
        batch_size: 批大
        device: 设备
        output_file: 输出结果文件路径 (可

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

    # 批量推理加
    num_batches = (len(test_data) + batch_size - 1) // batch_size
    logger.info(f"Using batch size: {batch_size}, total batches: {num_batches}")

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        # 每个批次前重新设padding (确保生效)
        processor.tokenizer.padding_side = 'left'

        # 准备批次数据
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_data))
        batch_samples = test_data[start_idx:end_idx]

        batch_images = []
        batch_texts = []
        batch_references = []
        batch_task_labels = []
        valid_indices = []

        for i, sample in enumerate(batch_samples):
            # 加载图片
            image_path = image_folder / sample['image_path']
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                continue

            # 构建输入
            conversations = sample['conversations']
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

            # 获取参考答
            reference = ""
            for conv in conversations:
                if conv["role"] == "assistant":
                    reference = conv["content"]
                    break

            task_label = sample.get("task_type", "general")

            batch_images.append(image)
            batch_texts.append(text)
            batch_references.append(reference)
            batch_task_labels.append(task_label)
            valid_indices.append(i)

        if not batch_images:
            continue

        # 批量处理
        try:
            # 确保 token IDs 是整
            pad_token_id = int(processor.tokenizer.pad_token_id) if processor.tokenizer.pad_token_id is not None else None
            eos_token_id = int(processor.tokenizer.eos_token_id) if processor.tokenizer.eos_token_id is not None else None

            inputs = processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 批量生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

            # 检查输
            if outputs is None or len(outputs) == 0:
                logger.error(f"Batch {batch_idx} generated empty outputs")
                raise ValueError("Empty generation outputs")

            # 批量解码
            generated_texts = processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # 验证解码结果
            if not generated_texts or len(generated_texts) != len(batch_images):
                logger.error(f"Batch {batch_idx} decode failed: expected {len(batch_images)}, got {len(generated_texts) if generated_texts else 0}")
                raise ValueError("Decode result mismatch")

            # 收集结果
            for hypothesis, reference, task_label in zip(generated_texts, batch_references, batch_task_labels):
                hypotheses.append(hypothesis.strip())
                references.append(reference)
                task_labels.append(task_label)

        except Exception as e:
            logger.error(f"Batch {batch_idx} inference failed: {e}")
            # 回退到逐个处理
            for i, (image, text, reference, task_label) in enumerate(zip(batch_images, batch_texts, batch_references, batch_task_labels)):
                try:
                    inputs = processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # 确保 token IDs 是整
                    pad_token_id = int(processor.tokenizer.pad_token_id) if processor.tokenizer.pad_token_id is not None else None
                    eos_token_id = int(processor.tokenizer.eos_token_id) if processor.tokenizer.eos_token_id is not None else None

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_length,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                        )

                    if outputs is None or len(outputs) == 0:
                        logger.warning(f"Sample {i} generated empty outputs, skipping")
                        continue

                    generated_texts = processor.batch_decode(
                        outputs,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    if not generated_texts or len(generated_texts) == 0:
                        logger.warning(f"Sample {i} decode failed, skipping")
                        continue

                    generated_text = generated_texts[0]

                    hypotheses.append(generated_text.strip())
                    references.append(reference)
                    task_labels.append(task_label)
                except Exception as e2:
                    logger.error(f"Sample inference failed: {e2}")
                    continue

    # 计算整体指标
    overall_metrics = calculate_accuracy(references, hypotheses)

    logger.info("=" * 80)
    logger.info("Overall Metrics:")
    logger.info(f"  CER: {overall_metrics['cer']:.4f}")
    logger.info(f"  WER: {overall_metrics['wer']:.4f}")
    logger.info(f"  Exact Match: {overall_metrics['exact_match']:.4f}")
    logger.info("=" * 80)

    # 按任务类型计算指
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
                        help="微调后的模型路径或LoRA adapter路径")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="基础模型路径（用于LoRA评估，如果不提供则使用model_path)")
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
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (None表示自动检测NPU/CUDA/CPU)")

    args = parser.parse_args()

    # 自动检测设
    if args.device is None:
        try:
            import torch_npu
            if torch.npu.is_available():
                args.device = "npu:0"
                logger.info("Auto-detected NPU device")
        except ImportError:
            pass

        if args.device is None:
            if torch.cuda.is_available():
                args.device = "cuda"
                logger.info("Auto-detected CUDA device")
            else:
                args.device = "cpu"
                logger.info("Using CPU device")

    # 确定基础模型路径和LoRA路径
    base_model = args.base_model_path if args.base_model_path else args.model_path
    lora_path = args.model_path if args.base_model_path else None

    if lora_path:
        logger.info(f"Loading LoRA model - Base: {base_model}, LoRA: {lora_path}")
        # 使用本地定义的load_lora_model,不依赖mindnlp
        model, processor = load_lora_model(
            base_model_path=base_model,
            lora_path=lora_path,
            device=args.device,
        )
    else:
        logger.info(f"Loading base model from {base_model}")
        from transformers import Qwen2VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(args.device)
        model.eval()

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

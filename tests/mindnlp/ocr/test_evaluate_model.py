"""
OCR模型评估测试 - 计算CER和WER指标

使用方法:
    pytest tests/ocr/test_evaluate_model.py
    python tests/ocr/test_evaluate_model.py --model_name "Qwen/Qwen2-VL-7B-Instruct" --test_data test_data.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import time
from tqdm import tqdm

# 添加mindnlp到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mindnlp.ocr.core.engine import VLMOCREngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    计算字符错误率 (Character Error Rate)
    
    CER = (S + D + I) / N
    其中: S=替换数, D=删除数, I=插入数, N=参考文本字符数
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    # 使用编辑距离算法 (Levenshtein距离)
    len_ref = len(reference)
    len_hyp = len(hypothesis)
    
    # 初始化距离矩阵
    dp = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]
    
    # 初始化第一行和第一列
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j
    
    # 计算编辑距离
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1,    # 插入
                    dp[i-1][j-1] + 1   # 替换
                )
    
    edit_distance = dp[len_ref][len_hyp]
    cer = edit_distance / len_ref if len_ref > 0 else 0.0
    
    return cer


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    计算词错误率 (Word Error Rate)
    
    WER = (S + D + I) / N
    其中: S=替换数, D=删除数, I=插入数, N=参考文本词数
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    
    len_ref = len(ref_words)
    len_hyp = len(hyp_words)
    
    # 初始化距离矩阵
    dp = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]
    
    # 初始化第一行和第一列
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j
    
    # 计算编辑距离
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1,    # 插入
                    dp[i-1][j-1] + 1   # 替换
                )
    
    edit_distance = dp[len_ref][len_hyp]
    wer = edit_distance / len_ref if len_ref > 0 else 0.0
    
    return wer


def normalize_text(text: str) -> str:
    """
    归一化文本 - 移除多余空格、换行符等
    """
    # 移除多余的空白字符
    text = ' '.join(text.split())
    return text.strip()


def load_test_data(test_data_path: str) -> List[Dict]:
    """
    加载测试数据
    """
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"✅ 加载了 {len(test_data)} 个测试样本")
    return test_data


def evaluate_model(
    model_name: str,
    test_data: List[Dict],
    lora_path: str = None,
    device: str = "npu:0"
) -> Dict:
    """
    评估OCR模型性能
    
    Returns:
        {
            "average_cer": float,
            "average_wer": float,
            "total_samples": int,
            "total_time": float,
            "avg_time_per_sample": float,
            "detailed_results": List[Dict]
        }
    """
    logger.info(f"📊 开始评估模型: {model_name}")
    if lora_path:
        logger.info(f"🔧 使用LoRA权重: {lora_path}")
    
    # 初始化OCR引擎
    logger.info("🚀 正在加载模型...")
    engine = VLMOCREngine(
        model_name=model_name,
        lora_weights_path=lora_path,
        device=device
    )
    logger.info("✅ 模型加载完成")
    
    # 评估结果
    results = {
        "total_samples": len(test_data),
        "successful_samples": 0,
        "failed_samples": 0,
        "total_cer": 0.0,
        "total_wer": 0.0,
        "total_time": 0.0,
        "detailed_results": []
    }
    
    # 逐个样本进行评估
    for idx, sample in enumerate(tqdm(test_data, desc="评估进度")):
        image_path = sample["image_path"]
        ground_truth = normalize_text(sample["ground_truth"])
        
        try:
            # 执行OCR识别
            start_time = time.time()
            prediction = engine.inference(
                image_path=image_path,
                prompt="识别图像中的所有文字内容"
            )
            inference_time = time.time() - start_time
            
            # 归一化预测结果
            prediction = normalize_text(prediction)
            
            # 计算CER和WER
            cer = calculate_cer(ground_truth, prediction)
            wer = calculate_wer(ground_truth, prediction)
            
            # 记录结果
            results["total_cer"] += cer
            results["total_wer"] += wer
            results["total_time"] += inference_time
            results["successful_samples"] += 1
            
            results["detailed_results"].append({
                "sample_id": idx,
                "image_path": image_path,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "cer": cer,
                "wer": wer,
                "inference_time": inference_time
            })
            
        except Exception as e:
            logger.error(f"❌ 样本 {idx} 评估失败: {str(e)}")
            results["failed_samples"] += 1
            results["detailed_results"].append({
                "sample_id": idx,
                "image_path": image_path,
                "error": str(e)
            })
    
    # 计算平均指标
    if results["successful_samples"] > 0:
        results["average_cer"] = results["total_cer"] / results["successful_samples"]
        results["average_wer"] = results["total_wer"] / results["successful_samples"]
        results["avg_time_per_sample"] = results["total_time"] / results["successful_samples"]
    else:
        results["average_cer"] = 1.0
        results["average_wer"] = 1.0
        results["avg_time_per_sample"] = 0.0
    
    return results


def print_results(results: Dict, output_file: str = None):
    """
    打印评估结果
    """
    print("\n" + "="*80)
    print("📊 评估结果汇总")
    print("="*80)
    print(f"总样本数: {results['total_samples']}")
    print(f"成功样本: {results['successful_samples']}")
    print(f"失败样本: {results['failed_samples']}")
    print(f"\n平均CER (字符错误率): {results['average_cer']:.4f} ({results['average_cer']*100:.2f}%)")
    print(f"平均WER (词错误率): {results['average_wer']:.4f} ({results['average_wer']*100:.2f}%)")
    print(f"\n总推理时间: {results['total_time']:.2f} 秒")
    print(f"平均每样本时间: {results['avg_time_per_sample']:.2f} 秒")
    print("="*80)
    
    # 保存详细结果到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 详细结果已保存到: {output_file}")


def compare_models(base_results: Dict, lora_results: Dict):
    """
    比较基础模型和LoRA微调模型的性能
    """
    print("\n" + "="*80)
    print("📈 模型对比分析")
    print("="*80)
    
    base_cer = base_results['average_cer']
    lora_cer = lora_results['average_cer']
    cer_improvement = (base_cer - lora_cer) / base_cer * 100
    
    base_wer = base_results['average_wer']
    lora_wer = lora_results['average_wer']
    wer_improvement = (base_wer - lora_wer) / base_wer * 100
    
    print(f"基础模型 CER: {base_cer:.4f} ({base_cer*100:.2f}%)")
    print(f"LoRA模型 CER: {lora_cer:.4f} ({lora_cer*100:.2f}%)")
    print(f"CER改进: {cer_improvement:+.2f}%")
    
    print(f"\n基础模型 WER: {base_wer:.4f} ({base_wer*100:.2f}%)")
    print(f"LoRA模型 WER: {lora_wer:.4f} ({lora_wer*100:.2f}%)")
    print(f"WER改进: {wer_improvement:+.2f}%")
    
    base_time = base_results['avg_time_per_sample']
    lora_time = lora_results['avg_time_per_sample']
    time_change = (lora_time - base_time) / base_time * 100
    
    print(f"\n基础模型平均时间: {base_time:.2f}秒")
    print(f"LoRA模型平均时间: {lora_time:.2f}秒")
    print(f"时间变化: {time_change:+.2f}%")
    
    print("="*80)
    
    # 检查是否满足Issue #2379要求
    print("\n✅ Issue #2379 要求检查:")
    if cer_improvement >= 20:
        print(f"✅ CER降低 {cer_improvement:.2f}% >= 20% (满足要求)")
    else:
        print(f"❌ CER降低 {cer_improvement:.2f}% < 20% (未满足要求)")


def main():
    parser = argparse.ArgumentParser(description="OCR模型评估脚本")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据JSON文件路径")
    parser.add_argument("--device", type=str, default="npu:0", help="设备 (npu:0 or cpu)")
    parser.add_argument("--output", type=str, default=None, help="结果输出文件路径")
    parser.add_argument("--compare_with", type=str, default=None, help="与另一个结果文件对比")
    
    args = parser.parse_args()
    
    # 加载测试数据
    test_data = load_test_data(args.test_data)
    
    # 评估模型
    results = evaluate_model(
        model_name=args.model_name,
        test_data=test_data,
        lora_path=args.lora_path,
        device=args.device
    )
    
    # 打印和保存结果
    output_file = args.output or f"eval_results_{int(time.time())}.json"
    print_results(results, output_file)
    
    # 如果指定了对比文件，进行对比分析
    if args.compare_with:
        with open(args.compare_with, 'r', encoding='utf-8') as f:
            compare_results = json.load(f)
        compare_models(compare_results, results)


if __name__ == "__main__":
    main()

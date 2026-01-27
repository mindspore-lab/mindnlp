"""
KV Cache 对比测试
测试启用和禁用 KV Cache 的性能差异
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position,import-error
from benchmark_kv_cache import (
    create_test_image,
    benchmark_single_inference,
    benchmark_batch_inference,
    benchmark_long_sequence,
)
from mindnlp.ocr.utils.cache_manager import CacheConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def run_comparison_benchmark(model_path: str,
                             device: str,
                             lora_path: str = None,
                             output_file: str = "comparison_results.json") -> Dict[str, Any]:
    """
    运行 KV Cache 启用/禁用对比测试

    Args:
        model_path: 模型路径
        device: 设备
        lora_path: LoRA权重路径
        output_file: 输出文件

    Returns:
        对比结果字典
    """
    from mindnlp.ocr.models.qwen2vl import Qwen2VLModel

    logger.info("="*80)
    logger.info("KV Cache Comparison Benchmark: Enabled vs Disabled")
    logger.info("="*80)

    test_image = create_test_image()

    results = {
        'test_config': {
            'model_path': model_path,
            'device': device,
            'lora_path': lora_path,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # ========== Test 1: KV Cache DISABLED ==========
    logger.info("%s", "\n" + "="*80)
    logger.info("Scenario 1: KV Cache DISABLED")
    logger.info("%s", "="*80)

    cache_config_disabled = CacheConfig(
        enable_kv_cache=False,
        enable_flash_attention=False,
    )

    logger.info("Loading model (KV Cache disabled)...")
    # pylint: disable=unexpected-keyword-arg
    model_disabled = Qwen2VLModel(
        model_name=model_path,
        device=device,
        lora_weights_path=lora_path,
        cache_config=cache_config_disabled
    )

    logger.info("Running tests with KV Cache disabled...")
    results['kv_cache_disabled'] = {
        'model_info': model_disabled.get_model_info(),
        'single_inference': benchmark_single_inference(model_disabled, test_image, num_runs=10),
        'batch_inference': benchmark_batch_inference(model_disabled, test_image, batch_sizes=[1, 2, 4]),
        'long_sequence': benchmark_long_sequence(model_disabled, test_image, max_tokens=1024),
    }

    # 释放模型
    del model_disabled
    import torch
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========== Test 2: KV Cache ENABLED ==========
    logger.info("%s", "\n" + "="*80)
    logger.info("Scenario 2: KV Cache ENABLED")
    logger.info("%s", "="*80)

    cache_config_enabled = CacheConfig(
        enable_kv_cache=True,
        max_cache_size_mb=2048.0,
        enable_lru=True,
        cache_ttl_seconds=300.0,
        enable_flash_attention=False,  # 先不启用 Flash Attention
    )

    logger.info("Loading model (KV Cache enabled)...")
    # pylint: disable=unexpected-keyword-arg
    model_enabled = Qwen2VLModel(
        model_name=model_path,
        device=device,
        lora_weights_path=lora_path,
        cache_config=cache_config_enabled
    )

    logger.info("Running tests with KV Cache enabled...")
    results['kv_cache_enabled'] = {
        'model_info': model_enabled.get_model_info(),
        'single_inference': benchmark_single_inference(model_enabled, test_image, num_runs=10),
        'batch_inference': benchmark_batch_inference(model_enabled, test_image, batch_sizes=[1, 2, 4]),
        'long_sequence': benchmark_long_sequence(model_enabled, test_image, max_tokens=1024),
    }

    # 释放模型
    del model_enabled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 计算性能提升
    results['improvement'] = calculate_improvement(
        results['kv_cache_disabled'],
        results['kv_cache_enabled']
    )

    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ Comparison benchmark completed! Results saved to: {output_path}")

    # 输出对比摘要
    print_comparison_summary(results)

    return results


def calculate_improvement(baseline: Dict, optimized: Dict) -> Dict[str, Any]:
    """
    计算性能提升百分比

    Args:
        baseline: 基准测试结果（KV Cache disabled）
        optimized: 优化后结果（KV Cache enabled）

    Returns:
        提升百分比字典
    """
    improvement = {}

    # 单图推理提升
    baseline_latency = baseline['single_inference']['latency_mean_ms']
    optimized_latency = optimized['single_inference']['latency_mean_ms']
    improvement['single_inference_speedup'] = (
        (baseline_latency - optimized_latency) / baseline_latency * 100
    )

    # 批量推理提升（batch=4）
    if 'batch_4' in baseline['batch_inference'] and 'batch_4' in optimized['batch_inference']:
        baseline_throughput = baseline['batch_inference']['batch_4']['throughput_images_per_sec']
        optimized_throughput = optimized['batch_inference']['batch_4']['throughput_images_per_sec']
        improvement['batch4_throughput_improvement'] = (
            (optimized_throughput - baseline_throughput) / baseline_throughput * 100
        )

    # 长序列提升
    baseline_tokens_per_sec = baseline['long_sequence']['tokens_per_second']
    optimized_tokens_per_sec = optimized['long_sequence']['tokens_per_second']
    improvement['long_sequence_speedup'] = (
        (optimized_tokens_per_sec - baseline_tokens_per_sec) / baseline_tokens_per_sec * 100
    )

    # 内存节省（如果有）
    if 'memory_peak_mb' in baseline['single_inference'] and 'memory_peak_mb' in optimized['single_inference']:
        baseline_memory = baseline['single_inference']['memory_peak_mb']
        optimized_memory = optimized['single_inference']['memory_peak_mb']
        improvement['memory_reduction'] = (
            (baseline_memory - optimized_memory) / baseline_memory * 100
        )

    return improvement


def print_comparison_summary(results: Dict[str, Any]):
    """打印对比摘要"""
    print("\n" + "="*80)
    print("KV CACHE COMPARISON SUMMARY")
    print("="*80)

    print(f"\nModel: {results['test_config']['model_path']}")
    print(f"Device: {results['test_config']['device']}")

    baseline = results['kv_cache_disabled']
    optimized = results['kv_cache_enabled']
    improvement = results['improvement']

    # 单图推理对比
    print("\n📊 Single Image Inference:")
    print(f"  KV Cache Disabled: {baseline['single_inference']['latency_mean_ms']:.2f} ms")
    print(f"  KV Cache Enabled:  {optimized['single_inference']['latency_mean_ms']:.2f} ms")
    print(f"  ⚡ Speedup: {improvement['single_inference_speedup']:.1f}%")

    # 批量推理对比
    print("\n📊 Batch Inference (batch=4):")
    if 'batch_4' in baseline['batch_inference']:
        baseline_b4 = baseline['batch_inference']['batch_4']
        optimized_b4 = optimized['batch_inference']['batch_4']
        print(f"  KV Cache Disabled: {baseline_b4['throughput_images_per_sec']:.2f} img/s")
        print(f"  KV Cache Enabled:  {optimized_b4['throughput_images_per_sec']:.2f} img/s")
        if 'batch4_throughput_improvement' in improvement:
            print(f"  ⚡ Throughput Improvement: {improvement['batch4_throughput_improvement']:.1f}%")

    # 长序列对比
    print("\n📊 Long Sequence Generation:")
    print(f"  KV Cache Disabled: {baseline['long_sequence']['tokens_per_second']:.2f} tokens/s")
    print(f"  KV Cache Enabled:  {optimized['long_sequence']['tokens_per_second']:.2f} tokens/s")
    print(f"  ⚡ Speedup: {improvement['long_sequence_speedup']:.1f}%")

    # 内存对比
    if 'memory_reduction' in improvement:
        print("\n💾 Memory Usage:")
        print(f"  KV Cache Disabled: {baseline['single_inference']['memory_peak_mb']:.2f} MB")
        print(f"  KV Cache Enabled:  {optimized['single_inference']['memory_peak_mb']:.2f} MB")
        print(f"  💾 Memory Reduction: {improvement['memory_reduction']:.1f}%")

    # 验收标准检查
    print("\n✅ Acceptance Criteria Check:")
    criteria_passed = 0
    criteria_total = 0

    # 1. 推理速度提升 20-30%
    criteria_total += 1
    if improvement['single_inference_speedup'] >= 20:
        print(f"  ✅ Inference speedup ≥20%: {improvement['single_inference_speedup']:.1f}%")
        criteria_passed += 1
    else:
        print(f"  ❌ Inference speedup <20%: {improvement['single_inference_speedup']:.1f}%")

    # 2. Batch=4 吞吐量提升 2.5-3x (150-200%)
    if 'batch4_throughput_improvement' in improvement:
        criteria_total += 1
        if improvement['batch4_throughput_improvement'] >= 150:
            print(f"  ✅ Batch throughput improvement ≥2.5x: {improvement['batch4_throughput_improvement']:.1f}%")
            criteria_passed += 1
        else:
            print(f"  ⚠️  Batch throughput improvement <2.5x: {improvement['batch4_throughput_improvement']:.1f}%")

    # 3. 长文本推理不 OOM
    criteria_total += 1
    if 'memory_peak_mb' in optimized['long_sequence']:
        print("  ✅ Long sequence inference completed without OOM")
        criteria_passed += 1
    else:
        print("  ⚠️  Memory info not available")

    print(f"\n📈 Overall: {criteria_passed}/{criteria_total} criteria passed")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="KV Cache Comparison Benchmark")
    parser.add_argument('--model_path', type=str, required=True, help="Model path or NPZ file")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda/npu/cpu)")
    parser.add_argument('--lora_path', type=str, default=None, help="LoRA weights path")
    parser.add_argument('--output', type=str, default='comparison_results.json', help="Output file")

    args = parser.parse_args()

    run_comparison_benchmark(
        model_path=args.model_path,
        device=args.device,
        lora_path=args.lora_path,
        output_file=args.output
    )


if __name__ == '__main__':
    main()

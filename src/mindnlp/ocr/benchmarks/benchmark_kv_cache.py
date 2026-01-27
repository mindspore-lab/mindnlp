"""
KV Cache 和 Flash Attention 性能基准测试
测试不同配置下的推理性能、内存占用和吞吐量
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mindnlp.ocr.models.qwen2vl import Qwen2VLModel  # pylint: disable=wrong-import-position
from mindnlp.ocr.utils.cache_manager import CacheConfig  # pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def create_test_image(width: int = 800, height: int = 600) -> Image.Image:
    """
    创建测试图像

    Args:
        width: 图像宽度
        height: 图像高度

    Returns:
        PIL图像对象
    """
    # 创建包含文本的测试图像
    from PIL import ImageDraw, ImageFont  # pylint: disable=reimported

    # 创建白色背景
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # 添加一些文本（模拟OCR场景）
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    text_lines = [
        "Invoice #12345",
        "Date: 2024-01-24",
        "Customer: Test Company",
        "Total Amount: $1,234.56",
        "Payment Status: Paid",
    ]

    y_position = 50
    for line in text_lines:
        draw.text((50, y_position), line, fill='black', font=font)
        y_position += 40

    return img


def measure_memory_usage():
    """
    测量内存使用情况

    Returns:
        内存使用字典 (MB)
    """
    import torch
    import gc

    gc.collect()

    memory_info = {}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_info['cuda_allocated'] = torch.cuda.memory_allocated() / (1024**2)
        memory_info['cuda_reserved'] = torch.cuda.memory_reserved() / (1024**2)
        memory_info['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / (1024**2)

    try:
        import torch_npu  # pylint: disable=import-error
        memory_info['npu_allocated'] = torch_npu.npu.memory_allocated() / (1024**2)
        memory_info['npu_reserved'] = torch_npu.npu.memory_reserved() / (1024**2)
    except:
        pass

    return memory_info


def benchmark_single_inference(model: Qwen2VLModel,
                               test_image: Image.Image,
                               num_runs: int = 10) -> Dict[str, Any]:
    """
    单图推理性能测试

    Args:
        model: 模型实例
        test_image: 测试图像
        num_runs: 测试次数

    Returns:
        性能统计字典
    """
    logger.info(f"Running single inference benchmark ({num_runs} runs)...")

    # Warmup
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "Extract all text from this invoice."}
            ]
        }
    ]
    _ = model.infer(messages)

    # 重置缓存统计
    model.reset_cache_stats()

    # 重置内存统计
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 记录开始内存
    mem_before = measure_memory_usage()

    # 执行测试
    latencies = []
    for i in range(num_runs):
        start_time = time.time()
        result = model.infer(messages)
        latency = time.time() - start_time
        latencies.append(latency)

        if (i + 1) % 5 == 0:
            logger.info(f"Progress: {i+1}/{num_runs}, avg latency: {np.mean(latencies):.3f}s")

    # 记录结束内存
    mem_after = measure_memory_usage()

    # 获取缓存统计
    cache_stats = model.get_cache_stats()

    # 计算统计数据
    results = {
        'num_runs': num_runs,
        'latency_mean_ms': np.mean(latencies) * 1000,
        'latency_std_ms': np.std(latencies) * 1000,
        'latency_min_ms': np.min(latencies) * 1000,
        'latency_max_ms': np.max(latencies) * 1000,
        'latency_p50_ms': np.percentile(latencies, 50) * 1000,
        'latency_p90_ms': np.percentile(latencies, 90) * 1000,
        'latency_p99_ms': np.percentile(latencies, 99) * 1000,
        'memory_before_mb': mem_before,
        'memory_after_mb': mem_after,
        'cache_stats': cache_stats,
    }

    # 计算内存增长
    if 'cuda_max_allocated' in mem_after:
        results['memory_peak_mb'] = mem_after['cuda_max_allocated']
    elif 'npu_allocated' in mem_after:
        results['memory_peak_mb'] = mem_after['npu_allocated']

    logger.info(f"Single inference benchmark complete: "
               f"mean={results['latency_mean_ms']:.2f}ms, "
               f"std={results['latency_std_ms']:.2f}ms")

    return results


def benchmark_batch_inference(model: Qwen2VLModel,
                              test_image: Image.Image,
                              batch_sizes: List[int] = [1, 2, 4, 8]) -> Dict[str, Any]:
    """
    批量推理性能测试

    Args:
        model: 模型实例
        test_image: 测试图像
        batch_sizes: 要测试的批量大小列表

    Returns:
        性能统计字典
    """
    logger.info(f"Running batch inference benchmark (batch_sizes={batch_sizes})...")

    results = {}

    for batch_size in batch_sizes:
        logger.info(f"Testing batch_size={batch_size}...")

        # 准备批量消息
        batch_messages = []
        for i in range(batch_size):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image},
                        {"type": "text", "text": f"Extract text from invoice #{i+1}."}
                    ]
                }
            ]
            batch_messages.append(messages)

        # Warmup
        _ = model.batch_generate(batch_messages, max_new_tokens=256)

        # 重置统计
        model.reset_cache_stats()
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 记录开始内存
        mem_before = measure_memory_usage()

        # 执行测试 (3 runs)
        latencies = []
        for _ in range(3):
            start_time = time.time()
            outputs = model.batch_generate(batch_messages, max_new_tokens=256)
            latency = time.time() - start_time
            latencies.append(latency)

        # 记录结束内存
        mem_after = measure_memory_usage()
        cache_stats = model.get_cache_stats()

        # 计算统计
        mean_latency = np.mean(latencies)
        throughput = batch_size / mean_latency  # 图片/秒

        results[f'batch_{batch_size}'] = {
            'batch_size': batch_size,
            'latency_mean_s': mean_latency,
            'latency_std_s': np.std(latencies),
            'throughput_images_per_sec': throughput,
            'latency_per_image_ms': (mean_latency / batch_size) * 1000,
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'cache_stats': cache_stats,
        }

        if 'cuda_max_allocated' in mem_after:
            results[f'batch_{batch_size}']['memory_peak_mb'] = mem_after['cuda_max_allocated']
        elif 'npu_allocated' in mem_after:
            results[f'batch_{batch_size}']['memory_peak_mb'] = mem_after['npu_allocated']

        logger.info(f"Batch {batch_size}: "
                   f"latency={mean_latency:.2f}s, "
                   f"throughput={throughput:.2f} img/s, "
                   f"per_image={results[f'batch_{batch_size}']['latency_per_image_ms']:.2f}ms")

    return results


def benchmark_long_sequence(model: Qwen2VLModel,
                            test_image: Image.Image,
                            max_tokens: int = 2048) -> Dict[str, Any]:
    """
    长序列生成测试（测试 KV Cache 效果）

    Args:
        model: 模型实例
        test_image: 测试图像
        max_tokens: 最大生成 token 数

    Returns:
        性能统计字典
    """
    logger.info(f"Running long sequence benchmark (max_tokens={max_tokens})...")

    # 使用较长的提示来触发更多 token 生成
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "Extract all text from this document and provide a detailed description of the layout, formatting, and content structure."}
            ]
        }
    ]

    # Warmup
    _ = model.infer(messages, max_new_tokens=512)

    # 重置统计
    model.reset_cache_stats()
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 记录开始内存
    mem_before = measure_memory_usage()

    # 执行测试
    start_time = time.time()
    result = model.infer(messages, max_new_tokens=max_tokens)
    latency = time.time() - start_time

    # 记录结束内存
    mem_after = measure_memory_usage()
    cache_stats = model.get_cache_stats()

    # 计算生成的 token 数（粗略估计）
    generated_text = result[0] if isinstance(result, list) else result
    approx_tokens = len(generated_text.split())

    results = {
        'max_tokens': max_tokens,
        'latency_s': latency,
        'approx_generated_tokens': approx_tokens,
        'tokens_per_second': approx_tokens / latency if latency > 0 else 0,
        'memory_before_mb': mem_before,
        'memory_after_mb': mem_after,
        'cache_stats': cache_stats,
        'generated_text_length': len(generated_text),
    }

    if 'cuda_max_allocated' in mem_after:
        results['memory_peak_mb'] = mem_after['cuda_max_allocated']
    elif 'npu_allocated' in mem_after:
        results['memory_peak_mb'] = mem_after['npu_allocated']

    logger.info(f"Long sequence benchmark complete: "
               f"latency={latency:.2f}s, "
               f"~{approx_tokens} tokens, "
               f"{results['tokens_per_second']:.2f} tokens/s")

    return results


def run_comprehensive_benchmark(model_path: str,
                                device: str,
                                lora_path: str = None,
                                enable_flash_attention: bool = False,
                                output_file: str = "benchmark_results.json") -> Dict[str, Any]:
    """
    运行完整的性能基准测试

    Args:
        model_path: 模型路径
        device: 设备
        lora_path: LoRA权重路径（可选）
        enable_flash_attention: 是否启用 Flash Attention
        output_file: 输出文件路径

    Returns:
        完整的测试结果字典
    """
    logger.info("="*80)
    logger.info("KV Cache and Flash Attention Performance Benchmark")
    logger.info("="*80)

    # 创建测试图像
    test_image = create_test_image()

    # 配置 KV Cache
    cache_config = CacheConfig(
        enable_kv_cache=True,
        max_cache_size_mb=2048.0,
        enable_lru=True,
        cache_ttl_seconds=300.0,
        enable_flash_attention=enable_flash_attention,
        auto_detect_flash_attention=True,
    )

    # 加载模型
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Device: {device}")
    logger.info(f"LoRA: {lora_path if lora_path else 'None'}")
    logger.info(f"Flash Attention: {enable_flash_attention}")

    # pylint: disable=unexpected-keyword-arg
    model = Qwen2VLModel(
        model_name=model_path,
        device=device,
        lora_weights_path=lora_path,
        cache_config=cache_config
    )

    # 获取模型信息
    model_info = model.get_model_info()
    logger.info(f"Model info: {json.dumps(model_info, indent=2)}")

    # 运行测试
    all_results = {
        'model_info': model_info,
        'test_config': {
            'model_path': model_path,
            'device': device,
            'lora_path': lora_path,
            'enable_flash_attention': enable_flash_attention,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 1. 单图推理测试
    logger.info("%s", "\n" + "="*80)
    logger.info("Test 1: Single Image Inference")
    logger.info("%s", "="*80)
    all_results['single_inference'] = benchmark_single_inference(model, test_image, num_runs=10)

    # 2. 批量推理测试
    logger.info("%s", "\n" + "="*80)
    logger.info("Test 2: Batch Inference")
    logger.info("%s", "="*80)
    all_results['batch_inference'] = benchmark_batch_inference(model, test_image, batch_sizes=[1, 2, 4])

    # 3. 长序列测试
    logger.info("%s", "\n" + "="*80)
    logger.info("Test 3: Long Sequence Generation")
    logger.info("%s", "="*80)
    all_results['long_sequence'] = benchmark_long_sequence(model, test_image, max_tokens=1024)

    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ Benchmark completed! Results saved to: {output_path}")

    # 输出摘要
    print_summary(all_results)

    return all_results


def print_summary(results: Dict[str, Any]):
    """打印测试摘要"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # 模型信息
    print(f"\nModel: {results['test_config']['model_path']}")
    print(f"Device: {results['test_config']['device']}")
    print(f"Flash Attention: {results['model_info'].get('flash_attention_enabled', False)}")
    print(f"KV Cache: {results['model_info'].get('kv_cache_enabled', False)}")

    # 单图推理
    single = results['single_inference']
    print(f"\n📊 Single Inference ({single['num_runs']} runs):")
    print(f"  Mean Latency: {single['latency_mean_ms']:.2f} ms (±{single['latency_std_ms']:.2f} ms)")
    print(f"  P50 / P90 / P99: {single['latency_p50_ms']:.2f} / {single['latency_p90_ms']:.2f} / {single['latency_p99_ms']:.2f} ms")
    if 'memory_peak_mb' in single:
        print(f"  Peak Memory: {single['memory_peak_mb']:.2f} MB")

    # 批量推理
    print("\n📊 Batch Inference:")
    batch = results['batch_inference']
    for key, data in batch.items():
        print(f"  Batch {data['batch_size']}: "
              f"{data['throughput_images_per_sec']:.2f} img/s, "
              f"{data['latency_per_image_ms']:.2f} ms/img")

    # 长序列
    long_seq = results['long_sequence']
    print(f"\n📊 Long Sequence (max_tokens={long_seq['max_tokens']}):")
    print(f"  Latency: {long_seq['latency_s']:.2f} s")
    print(f"  Throughput: {long_seq['tokens_per_second']:.2f} tokens/s")
    if 'memory_peak_mb' in long_seq:
        print(f"  Peak Memory: {long_seq['memory_peak_mb']:.2f} MB")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="KV Cache and Flash Attention Benchmark")
    parser.add_argument('--model_path', type=str, required=True, help="Model path or NPZ file")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda/npu/cpu)")
    parser.add_argument('--lora_path', type=str, default=None, help="LoRA weights path")
    parser.add_argument('--flash_attention', action='store_true', help="Enable Flash Attention")
    parser.add_argument('--output', type=str, default='benchmark_results.json', help="Output file")

    args = parser.parse_args()

    run_comprehensive_benchmark(
        model_path=args.model_path,
        device=args.device,
        lora_path=args.lora_path,
        enable_flash_attention=args.flash_attention,
        output_file=args.output
    )


if __name__ == '__main__':
    main()

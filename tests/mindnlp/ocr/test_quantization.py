"""
量化性能测试脚本 (Issue #2377)
测试不同量化模式的推理速度、显存占用和精度
"""

import os
import time
import torch
import psutil
from pathlib import Path
from PIL import Image

# 设置环境变量
os.environ['OCR_USE_MOCK_ENGINE'] = 'false'

from mindnlp.ocr.core.engine import VLMOCREngine
from mindnlp.ocr.api.schemas.request import OCRRequest
from mindnlp.ocr.utils.logger import get_logger

logger = get_logger(__name__)


def get_memory_usage():
    """获取当前进程内存使用量 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage(device):
    """获取GPU/NPU显存使用量 (MB)"""
    if "cuda" in device:
        return torch.cuda.memory_allocated() / 1024 / 1024
    elif "npu" in device:
        import torch_npu
        return torch_npu.npu.memory_allocated() / 1024 / 1024
    return 0


def benchmark_quantization(image_path: str, device: str = "npu:0", 
                           num_runs: int = 3, warmup_runs: int = 1):
    """
    对比不同量化模式的性能
    
    Args:
        image_path: 测试图像路径
        device: 运行设备
        num_runs: 测试运行次数
        warmup_runs: 预热运行次数
    """
    # 量化模式配置
    quantization_modes = {
        "FP16": {
            "mode": "fp16",
            "config": {}
        },
        "INT8": {
            "mode": "int8",
            "config": {
                "int8_threshold": 6.0,
            }
        },
        "INT4 (NF4)": {
            "mode": "int4",
            "config": {
                "int4_compute_dtype": "float16",
                "int4_quant_type": "nf4",
                "int4_use_double_quant": True,
            }
        },
    }
    
    # 加载测试图像
    logger.info(f"Loading test image: {image_path}")
    test_image = Image.open(image_path)
    
    results = []
    
    for quant_name, quant_config in quantization_modes.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing quantization: {quant_name}")
        logger.info(f"{'='*60}")
        
        try:
            # 初始化引擎
            logger.info(f"Initializing engine with {quant_name}...")
            mem_before = get_memory_usage()
            gpu_mem_before = get_gpu_memory_usage(device)
            
            engine = VLMOCREngine(
                model_name="Qwen/Qwen2-VL-7B-Instruct",
                device=device,
                enable_monitoring=False,
                quantization_mode=quant_config["mode"],
                quantization_config=quant_config["config"]
            )
            
            mem_after_load = get_memory_usage()
            gpu_mem_after_load = get_gpu_memory_usage(device)
            
            mem_used = mem_after_load - mem_before
            gpu_mem_used = gpu_mem_after_load - gpu_mem_before
            
            logger.info(f"Model loaded - Memory: {mem_used:.2f}MB, GPU Memory: {gpu_mem_used:.2f}MB")
            
            # 预热运行
            logger.info(f"Warming up ({warmup_runs} runs)...")
            for i in range(warmup_runs):
                request = OCRRequest(
                    image=test_image,
                    task_type="general_ocr",
                    output_format="text"
                )
                _ = engine.predict(request)
                logger.info(f"Warmup run {i+1}/{warmup_runs} completed")
            
            # 基准测试
            logger.info(f"Running benchmark ({num_runs} runs)...")
            inference_times = []
            
            for i in range(num_runs):
                request = OCRRequest(
                    image=test_image,
                    task_type="general_ocr",
                    output_format="text"
                )
                
                start_time = time.time()
                response = engine.predict(request)
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                logger.info(f"Run {i+1}/{num_runs}: {inference_time:.2f}s")
                logger.info(f"OCR Result preview: {response.text[:100]}...")
            
            # 计算统计信息
            avg_time = sum(inference_times) / len(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            
            result = {
                "quantization": quant_name,
                "mode": quant_config["mode"],
                "avg_inference_time": avg_time,
                "min_inference_time": min_time,
                "max_inference_time": max_time,
                "memory_used_mb": mem_used,
                "gpu_memory_used_mb": gpu_mem_used,
                "ocr_text": response.text
            }
            
            results.append(result)
            
            logger.info(f"\n{quant_name} Results:")
            logger.info(f"  Avg Inference Time: {avg_time:.2f}s")
            logger.info(f"  Min/Max Time: {min_time:.2f}s / {max_time:.2f}s")
            logger.info(f"  Memory Used: {mem_used:.2f}MB")
            logger.info(f"  GPU Memory Used: {gpu_mem_used:.2f}MB")
            
            # 清理
            del engine
            if "cuda" in device:
                torch.cuda.empty_cache()
            elif "npu" in device:
                import torch_npu
                torch_npu.npu.empty_cache()
            
        except Exception as e:
            logger.error(f"Error testing {quant_name}: {e}", exc_info=True)
            results.append({
                "quantization": quant_name,
                "mode": quant_config["mode"],
                "error": str(e)
            })
    
    # 打印对比报告
    print_comparison_report(results)
    
    return results


def print_comparison_report(results):
    """打印量化性能对比报告"""
    logger.info(f"\n{'='*80}")
    logger.info("QUANTIZATION PERFORMANCE COMPARISON REPORT")
    logger.info(f"{'='*80}")
    
    # 表头
    print(f"\n{'Quantization':<15} {'Avg Time (s)':<15} {'Memory (MB)':<15} {'GPU Mem (MB)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    # 基准（FP16）
    baseline_time = None
    baseline_memory = None
    
    for result in results:
        if "error" in result:
            print(f"{result['quantization']:<15} ERROR: {result['error']}")
            continue
        
        quant = result['quantization']
        avg_time = result['avg_inference_time']
        mem = result['memory_used_mb']
        gpu_mem = result['gpu_memory_used_mb']
        
        # 计算加速比
        if baseline_time is None:
            baseline_time = avg_time
            baseline_memory = mem
            speedup = "1.00x"
        else:
            speedup_ratio = baseline_time / avg_time
            speedup = f"{speedup_ratio:.2f}x"
        
        print(f"{quant:<15} {avg_time:<15.2f} {mem:<15.2f} {gpu_mem:<15.2f} {speedup:<10}")
    
    # 内存节省
    logger.info(f"\n{'='*80}")
    logger.info("MEMORY SAVINGS")
    logger.info(f"{'='*80}")
    
    if baseline_memory:
        for result in results:
            if "error" not in result:
                quant = result['quantization']
                mem = result['memory_used_mb']
                gpu_mem = result['gpu_memory_used_mb']
                mem_saving = (1 - mem / baseline_memory) * 100
                
                print(f"{quant:<15} Memory Saving: {mem_saving:>6.1f}%")
    
    logger.info(f"\n{'='*80}")


def test_accuracy(image_path: str, device: str = "npu:0"):
    """
    测试不同量化模式的OCR准确率
    
    Args:
        image_path: 测试图像路径
        device: 运行设备
    """
    logger.info(f"\n{'='*80}")
    logger.info("ACCURACY COMPARISON TEST")
    logger.info(f"{'='*80}")
    
    test_image = Image.open(image_path)
    
    quantization_modes = {
        "FP16": {"mode": "fp16", "config": {}},
        "INT8": {"mode": "int8", "config": {}},
        "INT4": {"mode": "int4", "config": {}},
    }
    
    results = {}
    
    for quant_name, quant_config in quantization_modes.items():
        try:
            logger.info(f"\nTesting {quant_name}...")
            
            engine = VLMOCREngine(
                model_name="Qwen/Qwen2-VL-7B-Instruct",
                device=device,
                enable_monitoring=False,
                quantization_mode=quant_config["mode"],
                quantization_config=quant_config["config"]
            )
            
            request = OCRRequest(
                image=test_image,
                task_type="general_ocr",
                output_format="text"
            )
            
            response = engine.predict(request)
            results[quant_name] = response.text
            
            logger.info(f"{quant_name} OCR Result:\n{response.text}\n")
            
            del engine
            if "cuda" in device:
                torch.cuda.empty_cache()
            elif "npu" in device:
                import torch_npu
                torch_npu.npu.empty_cache()
                
        except Exception as e:
            logger.error(f"Error testing {quant_name}: {e}")
            results[quant_name] = f"ERROR: {e}"
    
    # 字符级差异对比
    logger.info(f"\n{'='*80}")
    logger.info("CHARACTER-LEVEL DIFFERENCES")
    logger.info(f"{'='*80}")
    
    baseline_text = results.get("FP16", "")
    for quant_name, text in results.items():
        if quant_name != "FP16" and not text.startswith("ERROR"):
            diff_chars = sum(1 for a, b in zip(baseline_text, text) if a != b)
            total_chars = max(len(baseline_text), len(text))
            diff_ratio = diff_chars / total_chars * 100 if total_chars > 0 else 0
            
            logger.info(f"{quant_name}: {diff_chars}/{total_chars} chars different ({diff_ratio:.2f}%)")
    
    return results


if __name__ == "__main__":
    # 测试图像路径（请替换为实际路径）
    test_image_path = "path/to/your/test/image.jpg"
    
    if not Path(test_image_path).exists():
        logger.error(f"Test image not found: {test_image_path}")
        logger.info("Please specify a valid image path")
    else:
        # 运行性能基准测试
        logger.info("Starting quantization benchmark...")
        benchmark_results = benchmark_quantization(
            image_path=test_image_path,
            device="npu:0",  # 修改为实际设备
            num_runs=3,
            warmup_runs=1
        )
        
        # 运行准确率测试
        logger.info("\nStarting accuracy test...")
        accuracy_results = test_accuracy(
            image_path=test_image_path,
            device="npu:0"
        )
        
        logger.info("\n✅ All tests completed!")

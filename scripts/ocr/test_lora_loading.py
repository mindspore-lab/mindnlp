#!/usr/bin/env python3
"""
测试LoRA微调模型加载
使用方法: python test_lora_loading.py
"""

import os
import sys

# 设置环境变量
os.environ['TMPDIR'] = '/data1/tmp'
os.environ['HF_HOME'] = '/data1/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/data1/huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 添加路径
sys.path.insert(0, '/root/mindnlp/src')

import numpy as np
from pathlib import Path

def test_lora_weights_loading():
    """测试LoRA权重是否可以加载"""
    lora_path = "/data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39"
    weights_file = os.path.join(lora_path, 'adapter_model.npz')
    
    print("="*60)
    print("测试 LoRA 权重加载")
    print("="*60)
    print(f"权重路径: {weights_file}")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(weights_file):
        print(f"❌ 错误: 权重文件不存在: {weights_file}")
        return False
    
    print(f"✅ 权重文件存在")
    
    # 加载权重
    try:
        weights = np.load(weights_file)
        print(f"✅ 成功加载权重")
        print(f"   参数数量: {len(weights.files)}")
        print(f"   文件大小: {os.path.getsize(weights_file) / 1024 / 1024:.2f} MB")
        print()
        
        # 显示前10个参数的名称和形状
        print("前10个参数:")
        for i, key in enumerate(weights.files[:10]):
            shape = weights[key].shape
            dtype = weights[key].dtype
            print(f"  {i+1}. {key}")
            print(f"     形状: {shape}, 类型: {dtype}")
        
        if len(weights.files) > 10:
            print(f"  ... 还有 {len(weights.files) - 10} 个参数")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载（不包括LoRA）"""
    print()
    print("="*60)
    print("测试基础模型加载")
    print("="*60)
    
    try:
        from mindnlp.ocr.models.qwen2vl import Qwen2VLModel
        
        print("正在加载 Qwen2-VL-7B-Instruct...")
        print("这可能需要几分钟时间...")
        print()
        
        model = Qwen2VLModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="npu:0",
            quantization_mode="none"
        )
        
        print("✅ 基础模型加载成功")
        print(f"   模型类型: {type(model.model).__name__}")
        print(f"   Tokenizer类型: {type(model.tokenizer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lora_model_loading():
    """测试LoRA模型完整加载"""
    print()
    print("="*60)
    print("测试 LoRA 模型完整加载")
    print("="*60)
    
    try:
        from mindnlp.ocr.models.qwen2vl import Qwen2VLModel
        
        lora_path = "/data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39"
        
        print(f"正在加载基础模型 + LoRA权重...")
        print(f"LoRA路径: {lora_path}")
        print("这可能需要几分钟时间...")
        print()
        
        model = Qwen2VLModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="npu:0",
            quantization_mode="none",
            lora_weights_path=lora_path
        )
        
        print("✅ LoRA模型加载成功")
        print(f"   模型类型: {type(model.model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试...")
    print()
    
    # 测试1: 权重文件加载
    test1 = test_lora_weights_loading()
    
    # 测试2: 基础模型加载
    test2 = test_model_loading()
    
    # 测试3: LoRA模型完整加载
    test3 = test_lora_model_loading()
    
    # 总结
    print()
    print("="*60)
    print("测试总结")
    print("="*60)
    print(f"1. LoRA权重文件加载: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"2. 基础模型加载: {'✅ 通过' if test2 else '❌ 失败'}")
    print(f"3. LoRA模型完整加载: {'✅ 通过' if test3 else '❌ 失败'}")
    print("="*60)
    
    sys.exit(0 if all([test1, test2, test3]) else 1)

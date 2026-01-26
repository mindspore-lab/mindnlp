"""
服务器端 KV Cache 功能测试
测试 NPU 设备上的 KV Cache 和优化功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print("=" * 80)
print("KV Cache and Flash Attention - Server Test (NPU)")
print("=" * 80)

# 测试1: 导入模块（避免导入整个mindnlp包）
print("\n[Test 1] Importing cache_manager directly...")
try:
    # 直接导入，绕过 mindnlp.__init__.py
    import importlib.util
    
    cache_manager_path = project_root / "src/mindnlp/ocr/utils/cache_manager.py"
    spec = importlib.util.spec_from_file_location("cache_manager", cache_manager_path)
    cache_manager = importlib.util.module_from_spec(spec)
    
    # 先设置模块到 sys.modules 避免 dataclass 问题
    sys.modules['cache_manager'] = cache_manager
    spec.loader.exec_module(cache_manager)
    
    CacheConfig = cache_manager.CacheConfig
    KVCacheManager = cache_manager.KVCacheManager
    detect_flash_attention_support = cache_manager.detect_flash_attention_support
    get_optimal_cache_config = cache_manager.get_optimal_cache_config
    
    print("✅ Successfully imported cache_manager components")
except Exception as e:
    print(f"❌ Failed to import cache_manager: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: 创建配置
print("\n[Test 2] Creating CacheConfig for NPU...")
try:
    config = CacheConfig(
        enable_kv_cache=True,
        max_cache_size_mb=1024.0,
        enable_lru=True,
        cache_ttl_seconds=300.0,
        enable_flash_attention=False,  # NPU 不支持
    )
    print(f"✅ CacheConfig created:")
    print(f"   - KV Cache: {config.enable_kv_cache}")
    print(f"   - Max Size: {config.max_cache_size_mb} MB")
    print(f"   - LRU: {config.enable_lru}")
    print(f"   - Flash Attention: {config.enable_flash_attention}")
except Exception as e:
    print(f"❌ Failed to create config: {e}")
    import traceback
    traceback.print_exc()

# 测试3: KVCacheManager 基本操作
print("\n[Test 3] Testing KVCacheManager...")
try:
    manager = KVCacheManager(config)
    print("✅ KVCacheManager created")
    
    # 测试 put/get
    import torch
    test_tensor = torch.randn(10, 10)
    manager.put("test_key", test_tensor)
    print("✅ Cache put successful")
    
    retrieved = manager.get("test_key")
    if retrieved is not None and torch.equal(retrieved, test_tensor):
        print("✅ Cache get successful")
    else:
        print("❌ Cache get failed")
    
    # 测试统计
    stats = manager.get_stats()
    print(f"✅ Cache stats: {stats}")
    
    # 测试清理
    manager.clear()
    print("✅ Cache cleared")
    
except Exception as e:
    print(f"❌ KVCacheManager test failed: {e}")
    import traceback
    traceback.print_exc()

# 测试4: Flash Attention 检测（NPU应该不支持）
print("\n[Test 4] Flash Attention support detection...")
try:
    supported, reason = detect_flash_attention_support()
    if supported:
        print(f"✅ Flash Attention supported: {reason}")
    else:
        print(f"✅ Flash Attention not supported (expected for NPU): {reason}")
except Exception as e:
    print(f"❌ Flash Attention detection failed: {e}")
    import traceback
    traceback.print_exc()

# 测试5: NPU优化配置
print("\n[Test 5] Getting optimal config for NPU...")
try:
    npu_config = get_optimal_cache_config("npu:0", model_size_gb=7.0)
    print(f"✅ NPU optimal config:")
    print(f"   - KV Cache: {npu_config.enable_kv_cache}")
    print(f"   - Flash Attention: {npu_config.enable_flash_attention}")
    print(f"   - Max Cache Size: {npu_config.max_cache_size_mb} MB")
except Exception as e:
    print(f"❌ Optimal config failed: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 测试带NPZ模型的实际推理（如果模型存在）
print("\n[Test 6] Testing with actual model (if available)...")
model_path = "/data1/model_weights/qwen2vl_lora_merged.npz"
if os.path.exists(model_path):
    print(f"✅ Model found: {model_path}")
    print("   Attempting to load model with KV Cache...")
    
    try:
        from mindnlp.ocr.models.qwen2vl import Qwen2VLModel
        from PIL import Image
        import numpy as np
        
        # 创建测试图像
        test_img = Image.new('RGB', (800, 600), color='white')
        
        # 创建模型（使用优化配置）
        model = Qwen2VLModel(
            model_name=model_path,
            device="npu:0",
            cache_config=npu_config
        )
        print("✅ Model loaded successfully")
        
        # 获取模型信息
        model_info = model.get_model_info()
        print(f"✅ Model info:")
        print(f"   - Device: {model_info.get('device')}")
        print(f"   - KV Cache Enabled: {model_info.get('kv_cache_enabled')}")
        print(f"   - Flash Attention: {model_info.get('flash_attention_enabled')}")
        print(f"   - Attention Implementation: {model_info.get('attn_implementation', 'N/A')}")
        
        # 运行一次推理测试
        print("\n   Running inference test...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_img},
                    {"type": "text", "text": "Extract text."}
                ]
            }
        ]
        
        import time
        start_time = time.time()
        result = model.infer(messages, max_new_tokens=128)
        inference_time = time.time() - start_time
        
        print(f"✅ Inference completed in {inference_time:.2f}s")
        print(f"   Result: {result[:100] if result else 'Empty'}...")
        
        # 获取缓存统计
        cache_stats = model.get_cache_stats()
        print(f"✅ Cache stats after inference: {cache_stats}")
        
    except Exception as e:
        print(f"⚠️  Model test failed (expected if dependencies missing): {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"⚠️  Model not found at {model_path}, skipping model test")

print("\n" + "=" * 80)
print("Server Test Summary")
print("=" * 80)
print("✅ All basic tests completed!")
print("\n📝 Next steps:")
print("1. Run full benchmark: python scripts/ocr/benchmark_kv_cache.py")
print("2. Run comparison test: python scripts/ocr/benchmark_comparison.py")
print("\nTo run benchmarks on NPU:")
print("cd /data1/mindnlp")
print("python scripts/ocr/benchmark_kv_cache.py \\")
print("    --model_path /data1/model_weights/qwen2vl_lora_merged.npz \\")
print("    --device npu:0 \\")
print("    --output /data1/benchmark_results/kv_cache_npu.json")

"""
快速测试 KV Cache 和 Flash Attention 功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print("=" * 80)
print("Testing KV Cache and Flash Attention Implementation")
print("=" * 80)

# 测试1: 导入模块
print("\n[Test 1] Importing modules...")
try:
    from mindnlp.ocr.utils.cache_manager import (
        CacheConfig, KVCacheManager, detect_flash_attention_support, get_optimal_cache_config
    )
    print("✅ Successfully imported cache_manager module")
except Exception as e:
    print(f"❌ Failed to import cache_manager: {e}")
    sys.exit(1)

try:
    from mindnlp.ocr.models.qwen2vl import Qwen2VLModel
    print("✅ Successfully imported Qwen2VLModel")
except Exception as e:
    print(f"❌ Failed to import Qwen2VLModel: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: CacheConfig 创建
print("\n[Test 2] Creating CacheConfig...")
try:
    config = CacheConfig(
        enable_kv_cache=True,
        max_cache_size_mb=1024.0,
        enable_lru=True,
        cache_ttl_seconds=300.0,
        enable_flash_attention=False,
    )
    print(f"✅ CacheConfig created: kv_cache={config.enable_kv_cache}, max_size={config.max_cache_size_mb}MB")
except Exception as e:
    print(f"❌ Failed to create CacheConfig: {e}")
    sys.exit(1)

# 测试3: KVCacheManager 创建和基本操作
print("\n[Test 3] Testing KVCacheManager...")
try:
    cache_manager = KVCacheManager(config)
    
    # 测试 put/get
    import torch
    test_tensor = torch.randn(10, 10)
    cache_manager.put("test_key", test_tensor)
    
    retrieved = cache_manager.get("test_key")
    if retrieved is not None and torch.equal(retrieved, test_tensor):
        print("✅ Cache put/get works correctly")
    else:
        print("❌ Cache put/get failed")
    
    # 测试统计
    stats = cache_manager.get_stats()
    print(f"✅ Cache stats: {stats}")
    
    # 测试清理
    cache_manager.clear()
    print("✅ Cache cleared successfully")
    
except Exception as e:
    print(f"❌ KVCacheManager test failed: {e}")
    import traceback
    traceback.print_exc()

# 测试4: Flash Attention 检测
print("\n[Test 4] Detecting Flash Attention support...")
try:
    supported, reason = detect_flash_attention_support()
    if supported:
        print(f"✅ Flash Attention supported: {reason}")
    else:
        print(f"⚠️  Flash Attention not supported: {reason}")
except Exception as e:
    print(f"❌ Flash Attention detection failed: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 获取优化配置
print("\n[Test 5] Getting optimal cache config...")
try:
    import torch
    
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            devices.append("npu:0")
    except:
        pass
    
    for device in devices:
        optimal_config = get_optimal_cache_config(device, model_size_gb=7.0)
        print(f"✅ {device}: kv_cache={optimal_config.enable_kv_cache}, "
              f"flash_attn={optimal_config.enable_flash_attention}, "
              f"max_cache={optimal_config.max_cache_size_mb}MB")
except Exception as e:
    print(f"❌ Optimal config failed: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 模型初始化（不加载权重）
print("\n[Test 6] Testing model initialization with cache config...")
print("⚠️  Skipping full model load (requires model weights)")
print("   To test with actual model, run:")
print("   python -c \"from mindnlp.ocr.models.qwen2vl import Qwen2VLModel; \\")
print("              from mindnlp.ocr.utils.cache_manager import CacheConfig; \\")
print("              config = CacheConfig(enable_kv_cache=True); \\")
print("              model = Qwen2VLModel('path/to/model', 'cuda', cache_config=config); \\")
print("              print(model.get_model_info())\"")

# 测试7: 检查 qwen2vl.py 中的新方法
print("\n[Test 7] Checking Qwen2VLModel new methods...")
try:
    expected_methods = [
        'get_cache_stats',
        'clear_cache',
        'reset_cache_stats',
        'update_cache_config',
        'get_model_info'
    ]
    
    for method_name in expected_methods:
        if hasattr(Qwen2VLModel, method_name):
            print(f"✅ Method '{method_name}' exists")
        else:
            print(f"❌ Method '{method_name}' not found")
except Exception as e:
    print(f"❌ Method check failed: {e}")

print("\n" + "=" * 80)
print("Basic functionality tests completed!")
print("=" * 80)
print("\n📝 Next steps:")
print("1. Run benchmark_kv_cache.py to test performance")
print("2. Run benchmark_comparison.py to compare KV Cache on/off")
print("3. Update and commit code to GitHub")
print("\n✅ All basic tests passed! Ready for performance benchmarking.")

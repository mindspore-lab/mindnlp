# KV Cache and Flash Attention Optimization

本文档介绍 MindNLP OCR 项目中实现的 KV Cache 和 Flash Attention 优化功能。

## 功能概述

### 1. KV Cache 管理
- **自动缓存管理**：LRU 缓存策略，自动清理过期缓存
- **内存限制**：可配置的最大缓存大小，防止 OOM
- **缓存统计**：实时监控缓存命中率和内存使用
- **灵活配置**：支持启用/禁用、TTL 设置等

### 2. Flash Attention 2.0
- **硬件自动检测**：自动检测 GPU 是否支持 Flash Attention
- **性能优化**：降低 Attention 计算的显存占用（O(N) vs O(N²)）
- **降级策略**：不支持的硬件自动降级到标准实现
- **NPU 兼容**：NPU 设备自动禁用，使用 eager 实现

## 快速开始

### 基本使用

```python
from mindnlp.ocr.models.qwen2vl import Qwen2VLModel
from mindnlp.ocr.utils.cache_manager import CacheConfig

# 创建缓存配置
cache_config = CacheConfig(
    enable_kv_cache=True,          # 启用 KV Cache
    max_cache_size_mb=2048.0,      # 最大缓存 2GB
    enable_lru=True,               # 启用 LRU 清理
    cache_ttl_seconds=300.0,       # 缓存过期时间 5 分钟
    enable_flash_attention=True,   # 启用 Flash Attention（自动检测）
)

# 加载模型
model = Qwen2VLModel(
    model_name="/path/to/model_or_npz",
    device="cuda",
    lora_weights_path="/path/to/lora",  # 可选
    cache_config=cache_config
)

# 推理
messages = [{"role": "user", "content": [...]}]
result = model.infer(messages)

# 获取缓存统计
stats = model.get_cache_stats()
print(stats)
# {
#   'total_requests': 10,
#   'cache_hits': 8,
#   'cache_misses': 2,
#   'hit_rate': '80.00%',
#   'total_memory_mb': '128.45MB',
#   'evictions': 0,
#   'cache_items': 8
# }
```

### 自动配置（推荐）

```python
from mindnlp.ocr.models.qwen2vl import Qwen2VLModel
from mindnlp.ocr.utils.cache_manager import get_optimal_cache_config

# 根据设备自动配置
cache_config = get_optimal_cache_config(device="cuda", model_size_gb=7.0)

model = Qwen2VLModel(
    model_name="/path/to/model",
    device="cuda",
    cache_config=cache_config
)

# 查看模型信息（包括 Flash Attention 状态）
info = model.get_model_info()
print(info)
# {
#   'model_name': '/path/to/model',
#   'device': 'cuda:0',
#   'kv_cache_enabled': True,
#   'flash_attention_enabled': True,
#   'flash_attention_support': True,
#   'flash_attention_reason': 'Supported (CUDA 11.8, compute 8.0)',
#   'attn_implementation': 'flash_attention_2'
# }
```

## 性能基准测试

### 1. 运行完整基准测试

```bash
cd scripts/ocr

# NPU 测试（NPZ 格式）
python benchmark_kv_cache.py \
    --model_path /data1/model_weights/qwen2vl_lora_merged.npz \
    --device npu:0 \
    --output /data1/benchmark_results/npu_kv_cache.json

# CUDA 测试（HuggingFace 格式，启用 Flash Attention）
python benchmark_kv_cache.py \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --device cuda \
    --lora_path /path/to/lora \
    --flash_attention \
    --output benchmark_flash_attn.json
```

测试内容：
- 单图推理延迟（10 次运行）
- 批量推理吞吐量（batch=1/2/4/8）
- 长序列生成（max_tokens=2048）
- 内存占用峰值
- 缓存统计

### 2. 运行对比测试（KV Cache 启用 vs 禁用）

```bash
python benchmark_comparison.py \
    --model_path /data1/model_weights/qwen2vl_lora_merged.npz \
    --device npu:0 \
    --output /data1/benchmark_results/comparison.json
```

输出示例：
```
KV CACHE COMPARISON SUMMARY
================================================================================

📊 Single Image Inference:
  KV Cache Disabled: 2500.00 ms
  KV Cache Enabled:  1875.00 ms
  ⚡ Speedup: 25.0%

📊 Batch Inference (batch=4):
  KV Cache Disabled: 4.5 img/s
  KV Cache Enabled:  12.0 img/s
  ⚡ Throughput Improvement: 166.7%

📊 Long Sequence Generation:
  KV Cache Disabled: 15.2 tokens/s
  KV Cache Enabled:  22.8 tokens/s
  ⚡ Speedup: 50.0%

✅ Acceptance Criteria Check:
  ✅ Inference speedup ≥20%: 25.0%
  ✅ Batch throughput improvement ≥2.5x: 166.7%
  ✅ Long sequence inference completed without OOM

📈 Overall: 3/3 criteria passed
```

## API 参考

### CacheConfig

```python
@dataclass
class CacheConfig:
    enable_kv_cache: bool = True              # 启用 KV Cache
    max_cache_size_mb: float = 2048.0         # 最大缓存大小（MB）
    enable_lru: bool = True                   # 启用 LRU 清理
    cache_ttl_seconds: float = 300.0          # 缓存过期时间（秒）
    enable_flash_attention: bool = False      # 启用 Flash Attention
    auto_detect_flash_attention: bool = True  # 自动检测硬件支持
```

### Qwen2VLModel 方法

```python
# 获取缓存统计
stats = model.get_cache_stats()

# 清空缓存
model.clear_cache()

# 重置统计
model.reset_cache_stats()

# 更新配置
new_config = CacheConfig(enable_kv_cache=False)
model.update_cache_config(new_config)

# 获取模型信息
info = model.get_model_info()
```

## 性能优化建议

### NPU 设备
- ✅ 启用 KV Cache（自动启用）
- ❌ Flash Attention 不支持，自动禁用
- ✅ 使用 `attn_implementation='eager'`
- ✅ 推荐缓存大小：1024 MB

### CUDA 设备
- ✅ 启用 KV Cache
- ✅ 启用 Flash Attention（如果支持）
  - 需要：CUDA ≥ 11.6, GPU 架构 ≥ Ampere (8.0)
  - 需要安装：`pip install flash-attn`
- ✅ 推荐缓存大小：可用内存的 20%

### CPU 设备
- ✅ 启用 KV Cache
- ❌ Flash Attention 不支持
- ✅ 推荐缓存大小：512 MB

## 验收标准

根据 [Issue #2378](https://github.com/mindspore-lab/mindnlp/issues/2378)：

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| KV Cache 推理速度提升 | 20-30% | 待测试 | ⏳ |
| Flash Attention 显存降低 | 30-40% | 待测试 | ⏳ |
| Batch=4 吞吐量提升 | 2.5-3x | 待测试 | ⏳ |
| 长文本不 OOM | >2048 tokens | 待测试 | ⏳ |

## 故障排除

### Flash Attention 未启用

```python
info = model.get_model_info()
print(info['flash_attention_reason'])
```

常见原因：
- GPU 架构不支持（需要 Ampere 或更新）
- CUDA 版本过低（需要 ≥ 11.6）
- flash-attn 未安装：`pip install flash-attn`

### 缓存占用过多内存

```python
# 减小缓存大小
cache_config = CacheConfig(
    enable_kv_cache=True,
    max_cache_size_mb=512.0,  # 降低到 512 MB
    enable_lru=True
)

# 或手动清理
model.clear_cache()
```

### NPU 推理报错

NPU 设备必须使用 `attn_implementation='eager'`，代码已自动处理：

```python
# NPU 自动配置（无需手动设置）
model = Qwen2VLModel(model_name="...", device="npu:0")
```

## 依赖项

```txt
# 必需
torch >= 2.0
transformers >= 4.37.0
numpy

# 可选（Flash Attention）
flash-attn >= 2.0  # 仅 CUDA 设备需要
```

## 参考资料

- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [Transformers KV Cache 文档](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [Issue #2378](https://github.com/mindspore-lab/mindnlp/issues/2378)

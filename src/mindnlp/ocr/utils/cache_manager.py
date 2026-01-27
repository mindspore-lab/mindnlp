"""
KV Cache 管理器
实现 LRU 缓存策略和内存管理
"""

from __future__ import annotations

import gc
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """KV Cache 配置"""
    enable_kv_cache: bool = True  # 是否启用 KV Cache
    max_cache_size_mb: float = 2048.0  # 最大缓存大小（MB）
    enable_lru: bool = True  # 是否启用 LRU 清理
    cache_ttl_seconds: float = 300.0  # 缓存过期时间（秒）
    enable_flash_attention: bool = False  # 是否启用 Flash Attention
    auto_detect_flash_attention: bool = True  # 自动检测硬件支持


@dataclass
class CacheStats:
    """缓存统计信息"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_memory_mb: float = 0.0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests * 100

    def reset(self):
        """重置统计"""
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_memory_mb = 0.0
        self.evictions = 0


class KVCacheManager:
    """KV Cache 管理器"""

    def __init__(self, config: CacheConfig = None):
        """
        初始化 Cache 管理器

        Args:
            config: Cache 配置
        """
        self.config = config or CacheConfig()
        self.cache: OrderedDict = OrderedDict()
        self.stats = CacheStats()
        self.access_times: Dict[str, float] = {}

        logger.info(f"KVCacheManager initialized: enable={self.config.enable_kv_cache}, "
                   f"max_size={self.config.max_cache_size_mb}MB, "
                   f"lru={self.config.enable_lru}, ttl={self.config.cache_ttl_seconds}s")

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项

        Args:
            key: 缓存键

        Returns:
            缓存值或 None
        """
        self.stats.total_requests += 1

        if not self.config.enable_kv_cache:
            self.stats.cache_misses += 1
            return None

        # 检查是否存在
        if key not in self.cache:
            self.stats.cache_misses += 1
            return None

        # 检查是否过期
        if self.config.cache_ttl_seconds > 0:
            access_time = self.access_times.get(key, 0)
            if time.time() - access_time > self.config.cache_ttl_seconds:
                logger.debug(f"Cache expired: {key}")
                self.remove(key)
                self.stats.cache_misses += 1
                return None

        # 命中：更新访问时间和LRU顺序
        self.cache.move_to_end(key)
        self.access_times[key] = time.time()
        self.stats.cache_hits += 1

        logger.debug(f"Cache hit: {key} (hit_rate={self.stats.hit_rate:.2f}%)")
        return self.cache[key]

    def put(self, key: str, value: Any):
        """
        添加缓存项

        Args:
            key: 缓存键
            value: 缓存值
        """
        if not self.config.enable_kv_cache:
            return

        # 更新或添加
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value

        self.access_times[key] = time.time()

        # 检查内存使用
        current_memory_mb = self._estimate_cache_size()
        self.stats.total_memory_mb = current_memory_mb

        # 如果超过限制，执行LRU清理
        if self.config.enable_lru and current_memory_mb > self.config.max_cache_size_mb:
            self._evict_lru()

        logger.debug(f"Cache put: {key} (size={current_memory_mb:.2f}MB)")

    def remove(self, key: str):
        """
        删除缓存项

        Args:
            key: 缓存键
        """
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            logger.debug(f"Cache removed: {key}")

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cache cleared")

    def _estimate_cache_size(self) -> float:
        """
        估算缓存大小（MB）

        Returns:
            缓存大小（MB）
        """
        total_bytes = 0

        for value in self.cache.values():
            if isinstance(value, torch.Tensor):
                total_bytes += value.element_size() * value.nelement()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        total_bytes += item.element_size() * item.nelement()

        return total_bytes / (1024 * 1024)  # 转换为 MB

    def _evict_lru(self):
        """LRU 清理策略"""
        while len(self.cache) > 0:
            current_size = self._estimate_cache_size()
            if current_size <= self.config.max_cache_size_mb * 0.8:  # 清理到80%
                break

            # 删除最旧的项
            oldest_key = next(iter(self.cache))
            self.remove(oldest_key)
            self.stats.evictions += 1

        gc.collect()
        logger.info(f"LRU eviction completed: {self.stats.evictions} items evicted, "
                   f"current_size={self._estimate_cache_size():.2f}MB")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_requests': self.stats.total_requests,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'hit_rate': f"{self.stats.hit_rate:.2f}%",
            'total_memory_mb': f"{self.stats.total_memory_mb:.2f}MB",
            'evictions': self.stats.evictions,
            'cache_items': len(self.cache),
        }

    def reset_stats(self):
        """重置统计信息"""
        self.stats.reset()
        logger.info("Cache stats reset")


def detect_flash_attention_support() -> tuple:
    """
    检测硬件是否支持 Flash Attention

    Returns:
        (是否支持, 原因说明)
    """
    try:
        # 检查是否有 CUDA
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # 检查 CUDA 版本
        cuda_version = torch.version.cuda
        if cuda_version is None:
            return False, "CUDA version not found"

        cuda_major, cuda_minor = map(int, cuda_version.split('.')[:2])
        if cuda_major < 11 or (cuda_major == 11 and cuda_minor < 6):
            return False, f"CUDA {cuda_version} < 11.6 (required)"

        # 检查 GPU 架构（需要 Ampere 或更新）
        device_capability = torch.cuda.get_device_capability()
        compute_capability = device_capability[0] * 10 + device_capability[1]

        # Ampere (8.0+), Ada (8.9+), Hopper (9.0+)
        if compute_capability < 80:
            return False, f"Compute capability {compute_capability/10:.1f} < 8.0 (Ampere required)"

        # 尝试导入 flash_attn
        try:
            import flash_attn
            flash_version = getattr(flash_attn, '__version__', 'unknown')
            logger.info(f"Flash Attention {flash_version} detected")
            return True, f"Supported (CUDA {cuda_version}, compute {compute_capability/10:.1f})"
        except ImportError:
            return False, "flash-attn package not installed (pip install flash-attn)"

    except Exception as e:
        return False, f"Detection error: {e}"


def get_optimal_cache_config(device: str, model_size_gb: float = 7.0) -> CacheConfig:
    """
    获取优化的缓存配置

    Args:
        device: 设备类型 (cuda, npu, cpu)
        model_size_gb: 模型大小（GB）

    Returns:
        优化的 CacheConfig
    """
    config = CacheConfig()

    if "cuda" in device:
        # CUDA 设备
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # 缓存大小设为可用内存的 20%
            config.max_cache_size_mb = (total_memory_gb - model_size_gb) * 0.2 * 1024

        # 检测 Flash Attention
        if config.auto_detect_flash_attention:
            supported, reason = detect_flash_attention_support()
            config.enable_flash_attention = supported
            logger.info(f"Flash Attention: {supported} ({reason})")

    elif "npu" in device:
        # NPU 设备（通常不支持 Flash Attention）
        config.enable_flash_attention = False
        config.max_cache_size_mb = 1024.0  # 保守设置
        logger.info("NPU detected: Flash Attention disabled, cache limited to 1GB")

    else:
        # CPU 设备
        config.enable_kv_cache = True
        config.enable_flash_attention = False
        config.max_cache_size_mb = 512.0  # CPU 内存较大，但速度慢
        logger.info("CPU detected: Flash Attention disabled, cache limited to 512MB")

    return config

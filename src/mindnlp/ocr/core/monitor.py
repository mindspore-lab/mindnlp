"""
性能监控系统
收集和统计 OCR 系统的性能指标
"""

import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from threading import Lock
import psutil
from mindnlp.ocr.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    inference_time: float  # 推理耗时 (秒)
    image_count: int  # 处理的图像数量
    memory_used: float  # 内存占用 (MB)
    success: bool  # 是否成功
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, max_history: int = 1000):
        """
        初始化性能监控器

        Args:
            max_history: 保留的历史记录数量上限
        """
        self.max_history = max_history
        self._metrics_history: deque = deque(maxlen=max_history)
        self._lock = Lock()

        # 统计数据
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_images = 0
        self._total_inference_time = 0.0

        logger.info(f"PerformanceMonitor initialized (max_history={max_history})")

    def record_inference(
        self,
        inference_time: float,
        image_count: int = 1,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        记录一次推理的性能指标

        Args:
            inference_time: 推理耗时 (秒)
            image_count: 处理的图像数量
            success: 是否成功
            error_message: 错误信息（如果失败）
        """
        with self._lock:
            # 获取当前内存使用
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_used = memory_info.rss / 1024 / 1024  # 转换为 MB

            # 创建指标记录
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                inference_time=inference_time,
                image_count=image_count,
                memory_used=memory_used,
                success=success,
                error_message=error_message
            )

            # 添加到历史记录
            self._metrics_history.append(metrics)

            # 更新统计数据
            self._total_requests += 1
            if success:
                self._successful_requests += 1
                self._total_images += image_count
                self._total_inference_time += inference_time
            else:
                self._failed_requests += 1

            logger.debug(
                f"Recorded metrics: time={inference_time:.3f}s, "
                f"images={image_count}, success={success}, memory={memory_used:.1f}MB"
            )

    def get_statistics(self) -> Dict:
        """
        获取性能统计数据

        Returns:
            Dict: 包含各项统计指标的字典
        """
        with self._lock:
            if self._total_requests == 0:
                return {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "success_rate": 0.0,
                    "total_images": 0,
                    "average_inference_time": 0.0,
                    "throughput": 0.0,
                    "current_memory_mb": 0.0
                }

            # 计算平均推理时间
            avg_inference_time = (
                self._total_inference_time / self._successful_requests
                if self._successful_requests > 0 else 0.0
            )

            # 计算吞吐量 (images/second)
            throughput = (
                self._total_images / self._total_inference_time
                if self._total_inference_time > 0 else 0.0
            )

            # 成功率
            success_rate = self._successful_requests / self._total_requests

            # 当前内存使用
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024

            return {
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "success_rate": success_rate,
                "total_images": self._total_images,
                "average_inference_time": avg_inference_time,
                "throughput": throughput,
                "current_memory_mb": current_memory
            }

    def get_recent_metrics(self, count: int = 10) -> List[Dict]:
        """
        获取最近的性能指标记录

        Args:
            count: 返回的记录数量

        Returns:
            List[Dict]: 最近的性能指标列表
        """
        with self._lock:
            recent = list(self._metrics_history)[-count:]
            return [m.to_dict() for m in recent]

    def get_time_window_stats(self, window_seconds: int = 60) -> Dict:
        """
        获取时间窗口内的统计数据

        Args:
            window_seconds: 时间窗口大小（秒）

        Returns:
            Dict: 时间窗口内的统计数据
        """
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds

            # 过滤时间窗口内的记录
            window_metrics = [
                m for m in self._metrics_history
                if m.timestamp >= cutoff_time
            ]

            if not window_metrics:
                return {
                    "window_seconds": window_seconds,
                    "request_count": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "success_rate": 0.0,
                    "total_images": 0,
                    "average_inference_time": 0.0,
                    "throughput": 0.0,
                    "min_inference_time": 0.0,
                    "max_inference_time": 0.0,
                    "average_memory_mb": 0.0
                }

            # 统计指标
            successful = [m for m in window_metrics if m.success]
            failed = [m for m in window_metrics if not m.success]

            total_images = sum(m.image_count for m in successful)
            total_time = sum(m.inference_time for m in successful)

            inference_times = [m.inference_time for m in successful]
            memory_usage = [m.memory_used for m in window_metrics]

            return {
                "window_seconds": window_seconds,
                "request_count": len(window_metrics),
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "success_rate": len(successful) / len(window_metrics) if window_metrics else 0.0,
                "total_images": total_images,
                "average_inference_time": total_time / len(successful) if successful else 0.0,
                "throughput": total_images / total_time if total_time > 0 else 0.0,
                "min_inference_time": min(inference_times) if inference_times else 0.0,
                "max_inference_time": max(inference_times) if inference_times else 0.0,
                "average_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0.0
            }

    def reset(self):
        """重置所有统计数据"""
        with self._lock:
            self._metrics_history.clear()
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._total_images = 0
            self._total_inference_time = 0.0
            logger.info("Performance monitor reset")


# 全局性能监控器实例
_global_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """
    获取全局性能监控器实例（单例模式）

    Returns:
        PerformanceMonitor: 性能监控器实例
    """
    global _global_monitor

    if _global_monitor is None:
        with _monitor_lock:
            if _global_monitor is None:
                _global_monitor = PerformanceMonitor()

    return _global_monitor

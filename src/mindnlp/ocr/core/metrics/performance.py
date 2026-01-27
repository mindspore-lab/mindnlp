"""
性能监控模块 - 收集和导出 Prometheus 指标
"""

import time
from typing import Dict, Any
from collections import deque
from dataclasses import dataclass
from mindnlp.ocr.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyMetrics:
    """延迟指标"""
    p50: float = 0.0  # 中位数
    p95: float = 0.0  # 95分位
    p99: float = 0.0  # 99分位
    avg: float = 0.0  # 平均值
    min: float = 0.0  # 最小值
    max: float = 0.0  # 最大值


class PerformanceMonitor:
    """
    性能监控器

    功能:
    - QPS统计
    - 延迟分析 (P50/P95/P99)
    - GPU利用率监控 (如果可用)
    - 队列长度监控
    - 批处理命中率
    """

    def __init__(self, max_history: int = 1000):
        """
        初始化监控器

        Args:
            max_history: 保留的历史数据点数
        """
        self.max_history = max_history
        self.latencies = deque(maxlen=max_history)
        self.start_time = time.time()

        # 计数器
        self.counters = {
            "total_requests": 0,
            "success_requests": 0,
            "failed_requests": 0,
            "rejected_requests": 0,  # 限流拒绝
        }

        # 时间窗口计数 (用于计算QPS)
        self.window_size = 60  # 60秒窗口
        self.request_times = deque(maxlen=max_history)

        logger.info(f"PerformanceMonitor initialized (max_history={max_history})")

    def record_request(self, latency: float, success: bool = True):
        """
        记录请求

        Args:
            latency: 延迟(秒)
            success: 是否成功
        """
        now = time.time()

        self.counters["total_requests"] += 1
        if success:
            self.counters["success_requests"] += 1
            self.latencies.append(latency)
        else:
            self.counters["failed_requests"] += 1

        self.request_times.append(now)

    def record_rejection(self):
        """记录被拒绝的请求"""
        self.counters["rejected_requests"] += 1

    def get_qps(self) -> float:
        """
        计算当前QPS (过去60秒)

        Returns:
            QPS值
        """
        if not self.request_times:
            return 0.0

        now = time.time()
        cutoff_time = now - self.window_size

        # 统计时间窗口内的请求数
        recent_requests = sum(
            1 for t in self.request_times if t >= cutoff_time
        )

        return recent_requests / self.window_size

    def get_latency_metrics(self) -> LatencyMetrics:
        """
        计算延迟指标

        Returns:
            LatencyMetrics对象
        """
        if not self.latencies:
            return LatencyMetrics()

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        return LatencyMetrics(
            p50=sorted_latencies[int(n * 0.50)] if n > 0 else 0.0,
            p95=sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            p99=sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
            avg=sum(sorted_latencies) / n if n > 0 else 0.0,
            min=sorted_latencies[0] if n > 0 else 0.0,
            max=sorted_latencies[-1] if n > 0 else 0.0,
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        获取完整的监控摘要

        Returns:
            监控指标字典
        """
        latency_metrics = self.get_latency_metrics()
        qps = self.get_qps()

        # 计算成功率
        total = self.counters["total_requests"]
        if total > 0:
            success_rate = self.counters["success_requests"] / total
        else:
            success_rate = 0.0

        # 运行时间
        uptime_s = time.time() - self.start_time

        return {
            "uptime_seconds": uptime_s,
            "qps": qps,
            "counters": self.counters.copy(),
            "success_rate": success_rate,
            "latency": {
                "p50_ms": latency_metrics.p50 * 1000,
                "p95_ms": latency_metrics.p95 * 1000,
                "p99_ms": latency_metrics.p99 * 1000,
                "avg_ms": latency_metrics.avg * 1000,
                "min_ms": latency_metrics.min * 1000,
                "max_ms": latency_metrics.max * 1000,
            },
        }

    def reset(self):
        """重置所有统计"""
        self.latencies.clear()
        self.request_times.clear()
        self.counters = {
            "total_requests": 0,
            "success_requests": 0,
            "failed_requests": 0,
            "rejected_requests": 0,
        }
        self.start_time = time.time()
        logger.info("PerformanceMonitor reset")


class PrometheusExporter:
    """
    Prometheus 指标导出器

    将性能指标导出为 Prometheus 格式
    """

    def __init__(self, monitor: PerformanceMonitor):
        """
        初始化导出器

        Args:
            monitor: 性能监控器实例
        """
        self.monitor = monitor
        logger.info("PrometheusExporter initialized")

    def export_metrics(self) -> str:
        """
        导出 Prometheus 格式的指标

        Returns:
            Prometheus metrics 文本
        """
        summary = self.monitor.get_summary()
        latency = summary["latency"]
        counters = summary["counters"]

        metrics = []

        # QPS
        metrics.append("# HELP ocr_qps Current queries per second")
        metrics.append("# TYPE ocr_qps gauge")
        metrics.append(f"ocr_qps {summary['qps']:.2f}")
        metrics.append("")

        # 总请求数
        metrics.append("# HELP ocr_requests_total Total number of requests")
        metrics.append("# TYPE ocr_requests_total counter")
        metrics.append(f"ocr_requests_total {counters['total_requests']}")
        metrics.append("")

        # 成功请求数
        metrics.append("# HELP ocr_success_total Total number of successful requests")
        metrics.append("# TYPE ocr_success_total counter")
        metrics.append(f"ocr_success_total {counters['success_requests']}")
        metrics.append("")

        # 失败请求数
        metrics.append("# HELP ocr_failed_total Total number of failed requests")
        metrics.append("# TYPE ocr_failed_total counter")
        metrics.append(f"ocr_failed_total {counters['failed_requests']}")
        metrics.append("")

        # 被拒绝请求数
        metrics.append("# HELP ocr_rejected_total Total number of rejected requests")
        metrics.append("# TYPE ocr_rejected_total counter")
        metrics.append(f"ocr_rejected_total {counters['rejected_requests']}")
        metrics.append("")

        # 成功率
        metrics.append("# HELP ocr_success_rate Request success rate")
        metrics.append("# TYPE ocr_success_rate gauge")
        metrics.append(f"ocr_success_rate {summary['success_rate']:.4f}")
        metrics.append("")

        # 延迟指标
        for percentile, value in latency.items():
            metric_name = f"ocr_latency_{percentile}"
            metrics.append(f"# HELP {metric_name} Request latency {percentile}")
            metrics.append(f"# TYPE {metric_name} gauge")
            metrics.append(f"{metric_name} {value:.2f}")
            metrics.append("")

        # 运行时间
        metrics.append("# HELP ocr_uptime_seconds Service uptime in seconds")
        metrics.append("# TYPE ocr_uptime_seconds counter")
        metrics.append(f"ocr_uptime_seconds {summary['uptime_seconds']:.0f}")
        metrics.append("")

        return "\n".join(metrics)

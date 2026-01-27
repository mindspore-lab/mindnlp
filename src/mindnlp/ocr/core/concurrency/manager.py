"""
并发处理管理器 - 整合批处理、限流和监控
"""

import asyncio
import uuid
from typing import Any, Optional
from fastapi import HTTPException  # pylint: disable=import-error
from mindnlp.ocr.utils.logger import get_logger
from mindnlp.ocr.config.settings import get_settings
from mindnlp.ocr.core.batch import DynamicBatcher, BatchConfig
from mindnlp.ocr.core.limiter import TokenBucketRateLimiter, CircuitBreaker, RateLimitConfig
from mindnlp.ocr.core.metrics import PerformanceMonitor

logger = get_logger(__name__)
settings = get_settings()


class ConcurrencyManager:
    """
    并发处理管理器

    功能:
    - 动态批处理
    - 限流和熔断
    - 性能监控
    - 请求队列管理
    """

    def __init__(self, engine):
        """
        初始化并发管理器

        Args:
            engine: OCR引擎实例
        """
        self.engine = engine

        # 批处理配置
        batch_config = BatchConfig(
            max_batch_size=settings.max_batch_size,
            wait_timeout_ms=settings.batch_wait_timeout_ms,
            max_queue_size=settings.queue_maxsize,
            enable_dynamic_batching=True,
        )

        # 限流配置
        rate_limit_config = RateLimitConfig(
            qps=settings.qps_limit,
            burst=20,  # 允许20个请求的突发
            enable_circuit_breaker=True,
            circuit_breaker_threshold=10,
            circuit_breaker_timeout_s=30,
        )

        # 初始化组件
        self.batcher = DynamicBatcher(batch_config, self._batch_predict)
        self.rate_limiter = TokenBucketRateLimiter(rate_limit_config)
        self.circuit_breaker = CircuitBreaker(rate_limit_config)
        self.monitor = PerformanceMonitor(max_history=1000)

        self.is_running = False

        logger.info("ConcurrencyManager initialized")

    async def start(self):
        """启动并发管理器"""
        if self.is_running:
            logger.warning("ConcurrencyManager already running")
            return

        await self.batcher.start()
        self.is_running = True
        logger.info("ConcurrencyManager started")

    async def stop(self):
        """停止并发管理器"""
        if not self.is_running:
            return

        await self.batcher.stop()
        self.is_running = False
        logger.info("ConcurrencyManager stopped")

    async def process_request(
        self,
        request: Any,
        priority: int = 0,
        timeout: float = 150.0  # 增加到150秒，适应批处理（4图像×33秒）
    ) -> Any:
        """
        处理单个请求 (带限流、批处理、监控)

        Args:
            request: OCR请求对象
            priority: 优先级 (0=普通, 1=高, 2=紧急)
            timeout: 超时时间(秒)

        Returns:
            OCR响应对象

        Raises:
            HTTPException: 限流拒绝、熔断打开、处理失败等
        """
        import time
        start_time = time.time()

        # 限流检查
        if not await self.rate_limiter.acquire():
            self.monitor.record_rejection()
            logger.warning("Request rejected by rate limiter")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": 1.0,
                }
            )

        # 生成请求ID
        request_id = str(uuid.uuid4())

        try:
            # 通过熔断器和批处理器处理请求
            result = await self.circuit_breaker.call(
                self.batcher.submit,
                request_id,
                request,
                priority,
                timeout
            )

            # 记录成功
            latency = time.time() - start_time
            self.monitor.record_request(latency, success=True)

            return result

        except RuntimeError as e:
            # 熔断器打开
            logger.error(f"Circuit breaker rejected request: {e}")
            self.monitor.record_request(time.time() - start_time, success=False)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service temporarily unavailable",
                    "reason": "Circuit breaker is open",
                }
            )

        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timeout")
            self.monitor.record_request(time.time() - start_time, success=False)
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "Request timeout",
                    "timeout": timeout,
                }
            )

        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}", exc_info=True)
            self.monitor.record_request(time.time() - start_time, success=False)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "message": str(e),
                }
            )

    async def _batch_predict(self, requests: list) -> list:
        """
        批量推理 (由 DynamicBatcher 调用)

        Args:
            requests: 请求列表

        Returns:
            结果列表
        """
        if len(requests) == 1:
            # 单个请求,直接调用
            result = self.engine.predict(requests[0])
            return [result]
        else:
            # 批量请求,使用 process_batch (适配批处理器接口)
            results = self.engine.process_batch(requests)
            return results

    def get_stats(self) -> dict:
        """
        获取所有组件的统计信息

        Returns:
            统计信息字典
        """
        return {
            "batch": self.batcher.get_stats(),
            "rate_limiter": self.rate_limiter.get_stats(),
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "monitor": self.monitor.get_summary(),
        }

    def reset_stats(self):
        """重置所有统计"""
        self.batcher.reset_stats()
        self.rate_limiter.reset_stats()
        self.monitor.reset()
        logger.info("All statistics reset")


# 全局并发管理器实例 (延迟初始化)
_concurrency_manager: Optional[ConcurrencyManager] = None


def get_concurrency_manager() -> Optional[ConcurrencyManager]:
    """获取并发管理器实例"""
    return _concurrency_manager


def init_concurrency_manager(engine):
    """
    初始化并发管理器

    Args:
        engine: OCR引擎实例
    """
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = ConcurrencyManager(engine)
        logger.info("Global ConcurrencyManager initialized")
    return _concurrency_manager
